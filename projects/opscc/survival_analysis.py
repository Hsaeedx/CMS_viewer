"""
Overall survival analysis: TORS only vs CT/CRT only
Propensity score-matched cohort (N = 2,030)
Time origin: first_tx_date
Event: all-cause death (MBSF DEATH_DT)
Censoring: December 31 of each patient's last enrollment year in mbsf_all
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter

DB_PATH = r"E:\CMS\cms_data.duckdb"

# ── 1. Pull matched cohort with survival data ─────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)

# Last enrollment year per patient (for censoring date)
# Death date: take the non-null DEATH_DT from any year's mbsf row
survival = con.execute("""
    WITH matched AS (
        SELECT DSYSRTKY, tx_group, first_tx_date, psm_match_id,
               van_walraven_score, age_at_dx
        FROM opscc_propensity
        WHERE psm_matched = TRUE
    ),
    mbsf_summary AS (
        SELECT
            m.DSYSRTKY,
            MAX(CAST(m.RFRNC_YR AS INTEGER))                         AS last_enrl_year,
            MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                     THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END)   AS death_date
        FROM mbsf_all m
        JOIN matched p ON m.DSYSRTKY = p.DSYSRTKY
        GROUP BY m.DSYSRTKY
    )
    SELECT
        p.DSYSRTKY,
        p.tx_group,
        p.first_tx_date,
        p.psm_match_id,
        p.van_walraven_score,
        p.age_at_dx,
        s.death_date,
        s.last_enrl_year,
        -- Censoring date: Dec 31 of last enrollment year
        make_date(s.last_enrl_year, 12, 31) AS censor_date
    FROM matched p
    JOIN mbsf_summary s ON p.DSYSRTKY = s.DSYSRTKY
""").df()

con.close()
print(f"Loaded {len(survival):,} patients in matched cohort")
print(f"Deaths observed: {survival['death_date'].notna().sum():,}")

# ── 2. Build time-to-event variables ──────────────────────────────────────────
survival['event_date'] = survival['death_date'].combine_first(survival['censor_date'])
survival['event']      = survival['death_date'].notna().astype(int)

# Time in days from first treatment to event/censor
survival['t_days'] = (survival['event_date'] - survival['first_tx_date']).dt.days

# Drop any with missing tx date or negative follow-up
survival = survival[survival['first_tx_date'].notna() & (survival['t_days'] >= 0)].copy()
print(f"After dropping missing tx date / negative follow-up: {len(survival):,}")

tors  = survival[survival['tx_group'] == 'TORS only']
ctcrt = survival[survival['tx_group'] == 'CT/CRT only']

# Median follow-up (reverse KM of censoring)
def median_followup(df):
    """Median follow-up via reverse KM (censoring as event)."""
    kmf = KaplanMeierFitter()
    kmf.fit(df['t_days'], 1 - df['event'])
    return kmf.median_survival_time_

print(f"\nMedian follow-up:")
print(f"  TORS:   {median_followup(tors)/365.25:.1f} years ({median_followup(tors):.0f} days)")
print(f"  CT/CRT: {median_followup(ctcrt)/365.25:.1f} years ({median_followup(ctcrt):.0f} days)")

# ── 3. Overall survival summary ───────────────────────────────────────────────
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

kmf_tors  = KaplanMeierFitter(label="TORS only")
kmf_ctcrt = KaplanMeierFitter(label="CT/CRT only")

kmf_tors.fit(tors['t_days'],   tors['event'])
kmf_ctcrt.fit(ctcrt['t_days'], ctcrt['event'])

print(f"\n-- Overall Survival (Kaplan-Meier) ------------------------------------")
print(f"{'Timepoint':<14} {'TORS %':>8} {'95% CI':>16}   {'CT/CRT %':>10} {'95% CI':>16}")
print("-" * 70)
for yrs, days in [(1, 365), (2, 730), (3, 1095), (5, 1825)]:
    ts, tl, th = survival_at(kmf_tors,  days)
    cs, cl, ch = survival_at(kmf_ctcrt, days)
    print(f"{yrs}-year{'':<8} {100*ts:>7.1f}%  ({100*tl:.1f}%-{100*th:.1f}%)    "
          f"{100*cs:>9.1f}%  ({100*cl:.1f}%-{100*ch:.1f}%)")

print(f"\nMedian OS:")
print(f"  TORS:   {kmf_tors.median_survival_time_/365.25:.2f} yrs  ({kmf_tors.median_survival_time_:.0f} days)")
print(f"  CT/CRT: {kmf_ctcrt.median_survival_time_/365.25:.2f} yrs  ({kmf_ctcrt.median_survival_time_:.0f} days)")

# ── 4. Log-rank test ───────────────────────────────────────────────────────────
lr = logrank_test(
    tors['t_days'],  ctcrt['t_days'],
    tors['event'],   ctcrt['event']
)
print(f"\n-- Log-rank test -------------------------------------------------------")
print(f"  Test statistic: {lr.test_statistic:.3f}")
print(f"  p-value:        {lr.p_value:.4f}")

# ── 5. Cox proportional hazards (matched pairs, adjusted) ─────────────────────
# Covariates: age, van_walraven_score (residual confounders)
cox_df = survival[['t_days','event','tx_group','age_at_dx','van_walraven_score',
                   'psm_match_id']].copy()
cox_df['tors'] = (cox_df['tx_group'] == 'TORS only').astype(int)

cph = CoxPHFitter()
cph.fit(cox_df[['t_days','event','tors','age_at_dx','van_walraven_score']],
        duration_col='t_days', event_col='event')

print(f"\n-- Cox Proportional Hazards (adjusted for age + van Walraven) ---------")
summary = cph.summary[['coef','exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']]
summary.columns = ['log HR','HR','HR lower 95%','HR upper 95%','p']
print(summary.round(4).to_string())

print(f"\nSchoenfeld residuals (PH assumption test):")
cph.check_assumptions(cox_df[['t_days','event','tors','age_at_dx','van_walraven_score']],
                      p_value_threshold=0.05, show_plots=False)

# ── 6. Mortality rates ─────────────────────────────────────────────────────────
print(f"\n-- Crude Mortality Rates -----------------------------------------------")
for grp, df_g in [("TORS only", tors), ("CT/CRT only", ctcrt)]:
    n         = len(df_g)
    deaths    = df_g['event'].sum()
    py        = df_g['t_days'].sum() / 365.25
    rate_100  = 100 * deaths / py
    print(f"  {grp:<12}  N={n:,}  Deaths={deaths:,}  Person-years={py:.0f}  "
          f"Rate={rate_100:.1f}/100 PY  Crude mortality={100*deaths/n:.1f}%")
