"""
Overall survival analysis — two PSM-matched comparisons:
  Comparison A: TORS alone  vs RT alone
  Comparison B: TORS + RT   vs CT/CRT
Time origin: first_tx_date
Event: all-cause death (MBSF DEATH_DT)
Censoring: December 31 of each patient's last enrollment year in mbsf_all

Reads from pre-built SQL tables:
  opscc_survival    — death date + Dec-31 censor per patient (step 11)
  opscc_propensity  — PSM match flags (updated by step 13)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

DB_PATH = r"F:\CMS\cms_data.duckdb"

COMPARISONS = [
    ('A', 'psm_matched_A', 'psm_match_id_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'psm_match_id_B', 'TORS + RT',  'CT/CRT'),
]


def survival_at(kmf, t_days):
    """KM survival estimate at a given time, with 95% CI."""
    try:
        sf = kmf.survival_function_at_times(t_days).iloc[0]
        ci = kmf.confidence_interval_survival_function_
        lo = np.interp(t_days, ci.index, ci.iloc[:, 0])
        hi = np.interp(t_days, ci.index, ci.iloc[:, 1])
        return sf, lo, hi
    except Exception:
        return np.nan, np.nan, np.nan


def median_followup(df):
    """Median follow-up via reverse KM (censoring as event)."""
    kmf = KaplanMeierFitter()
    kmf.fit(df['t_days'], 1 - df['event'])
    return kmf.median_survival_time_


con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

for comp, match_col, match_id_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*70}")
    print(f"  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}")
    print(f"{'#'*70}")

    # ── 1. Pull matched cohort from pre-built survival table ──────────────────
    survival = con.execute(f"""
        SELECT s.DSYSRTKY, s.tx_group, s.first_tx_date,
               s.van_walraven_score, s.age_at_dx,
               s.death_date, s.censor_date, s.event, s.t_days,
               p.{match_id_col} AS psm_match_id
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
          AND s.t_days >= 0
    """).df()

    print(f"Loaded {len(survival):,} patients  |  Deaths: {survival['event'].sum():,}")

    tors_s = survival[survival['tx_group'] == tors_label]
    ctrl_s = survival[survival['tx_group'] == ctrl_label]

    print(f"\nMedian follow-up:")
    print(f"  {tors_label}: {median_followup(tors_s)/365.25:.1f} yrs")
    print(f"  {ctrl_label}: {median_followup(ctrl_s)/365.25:.1f} yrs")

    # ── 3. KM curves ──────────────────────────────────────────────────────────
    kmf_t = KaplanMeierFitter(label=tors_label)
    kmf_c = KaplanMeierFitter(label=ctrl_label)
    kmf_t.fit(tors_s['t_days'], tors_s['event'])
    kmf_c.fit(ctrl_s['t_days'], ctrl_s['event'])

    print(f"\n-- Overall Survival (Kaplan-Meier) -----------------------------------")
    print(f"{'Timepoint':<14} {tors_label+' %':>14} {'95% CI':>16}   {ctrl_label+' %':>14} {'95% CI':>16}")
    print("-" * 80)
    for yrs, days in [(1, 365), (2, 730), (3, 1095), (5, 1825)]:
        ts, tl, th = survival_at(kmf_t, days)
        cs, cl, ch = survival_at(kmf_c, days)
        print(f"{yrs}-year{'':<8} {100*ts:>13.1f}%  ({100*tl:.1f}%\u2013{100*th:.1f}%)    "
              f"{100*cs:>13.1f}%  ({100*cl:.1f}%\u2013{100*ch:.1f}%)")

    med_t = kmf_t.median_survival_time_
    med_c = kmf_c.median_survival_time_
    print(f"\nMedian OS:")
    print(f"  {tors_label}: {med_t/365.25:.2f} yrs  ({med_t:.0f} days)")
    print(f"  {ctrl_label}: {med_c/365.25:.2f} yrs  ({med_c:.0f} days)")

    # ── 4. Log-rank ────────────────────────────────────────────────────────────
    lr = logrank_test(tors_s['t_days'], ctrl_s['t_days'],
                      tors_s['event'],  ctrl_s['event'])
    print(f"\n-- Log-rank test -------------------------------------------------------")
    print(f"  Test statistic: {lr.test_statistic:.3f}")
    print(f"  p-value:        {lr.p_value:.4f}")

    # ── 5. Cox PH ──────────────────────────────────────────────────────────────
    cox_df = survival[['t_days','event','tx_group','age_at_dx',
                        'van_walraven_score','psm_match_id']].copy()
    cox_df['tors'] = (cox_df['tx_group'] == tors_label).astype(int)

    cph = CoxPHFitter()
    cph.fit(cox_df[['t_days','event','tors','age_at_dx','van_walraven_score']],
            duration_col='t_days', event_col='event')

    print(f"\n-- Cox PH (adjusted for age + van Walraven) ---------------------------")
    summary = cph.summary[['coef','exp(coef)','exp(coef) lower 95%',
                            'exp(coef) upper 95%','p']]
    summary.columns = ['log HR','HR','HR lower 95%','HR upper 95%','p']
    print(summary.round(4).to_string())

    print(f"\n-- Crude Mortality Rates -----------------------------------------------")
    for grp, df_g in [(tors_label, tors_s), (ctrl_label, ctrl_s)]:
        n      = len(df_g)
        deaths = df_g['event'].sum()
        py     = df_g['t_days'].sum() / 365.25
        print(f"  {grp:<14}  N={n:,}  Deaths={deaths:,}  Person-years={py:.0f}  "
              f"Rate={100*deaths/py:.1f}/100 PY  Crude={100*deaths/n:.1f}%")

con.close()
