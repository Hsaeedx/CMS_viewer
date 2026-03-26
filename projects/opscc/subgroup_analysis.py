"""
Subgroup survival analysis — two PSM-matched comparisons:
  Comparison A: TORS alone  vs RT alone
  Comparison B: TORS + RT   vs CT/CRT

Subgroups:
  1. Age < 75
  2. Age >= 75
  3. Elixhauser van Walraven tertiles (Low / Mid / High)

C77 (nodal disease) patients excluded from all analyses.
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

DB_PATH = r"F:\CMS\cms_data.duckdb"

COMPARISONS = [
    ('A', 'psm_matched_A', 'psm_match_id_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'psm_match_id_B', 'TORS + RT',  'CT/CRT'),
]

# ── C77 exclusion CTE (reused across both comparisons) ────────────────────────
C77_CTE = """
    c77_patients AS (
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN inp_claimsk_all i ON i.DSYSRTKY = o.DSYSRTKY,
        UNNEST([i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
                i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,
                i.ICD_DGNS_CD4,  i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,
                i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,  i.ICD_DGNS_CD9,
                i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
                i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN out_claimsk_all oc ON oc.DSYSRTKY = o.DSYSRTKY,
        UNNEST([oc.PRNCPAL_DGNS_CD,
                oc.ICD_DGNS_CD1, oc.ICD_DGNS_CD2, oc.ICD_DGNS_CD3,
                oc.ICD_DGNS_CD4, oc.ICD_DGNS_CD5, oc.ICD_DGNS_CD6,
                oc.ICD_DGNS_CD7, oc.ICD_DGNS_CD8, oc.ICD_DGNS_CD9,
                oc.ICD_DGNS_CD10]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN car_linek_all cl ON cl.DSYSRTKY = o.DSYSRTKY
        WHERE cl.LINE_ICD_DGNS_CD LIKE 'C77%'
          AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
    )
"""


def run_analysis(label, subset, tors_label, ctrl_label, match_id_col):
    tors_s = subset[subset['tors'] == 1]
    ctrl_s = subset[subset['tors'] == 0]

    if len(tors_s) < 10 or len(ctrl_s) < 10:
        print(f"\n  {label}: insufficient n ({tors_label}={len(tors_s)}, {ctrl_label}={len(ctrl_s)})")
        return

    kmf_t = KaplanMeierFitter()
    kmf_c = KaplanMeierFitter()
    kmf_t.fit(tors_s['t_days'], tors_s['event'])
    kmf_c.fit(ctrl_s['t_days'], ctrl_s['event'])

    def km_at(kmf, days):
        try:
            return kmf.survival_function_at_times(days).iloc[0]
        except Exception:
            return np.nan

    lr = logrank_test(tors_s['t_days'], ctrl_s['t_days'],
                      tors_s['event'],  ctrl_s['event'])

    cph_cols = ['t_days','event','tors','age_at_dx','van_walraven_score', match_id_col]
    subset_cph = subset[cph_cols].rename(columns={match_id_col: 'psm_match_id'})
    cph = CoxPHFitter()
    try:
        cph.fit(subset_cph, duration_col='t_days', event_col='event',
                strata=['psm_match_id'])
        row   = cph.summary.loc['tors']
        hr    = row['exp(coef)']
        hr_lo = row['exp(coef) lower 95%']
        hr_hi = row['exp(coef) upper 95%']
        p_cox = row['p']
    except Exception:
        hr = hr_lo = hr_hi = p_cox = np.nan

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N: {tors_label}={len(tors_s):,}  {ctrl_label}={len(ctrl_s):,}  "
          f"Deaths: {tors_label}={int(tors_s['event'].sum())} ({100*tors_s['event'].mean():.1f}%)  "
          f"{ctrl_label}={int(ctrl_s['event'].sum())} ({100*ctrl_s['event'].mean():.1f}%)")
    print(f"  {'Timepoint':<10} {tors_label+' OS':>12} {ctrl_label+' OS':>12}")
    print(f"  {'-'*36}")
    for yrs, days in [(1,365),(2,730),(3,1095),(5,1825)]:
        ts = km_at(kmf_t, days)
        cs = km_at(kmf_c, days)
        print(f"  {yrs}-year    {100*ts:>11.1f}%  {100*cs:>11.1f}%")
    med_t = kmf_t.median_survival_time_
    med_c = kmf_c.median_survival_time_
    med_t_str = f"{med_t/365.25:.1f} yr" if not np.isinf(med_t) else "NR"
    med_c_str = f"{med_c/365.25:.1f} yr" if not np.isinf(med_c) else "NR"
    print(f"  Median OS   {med_t_str:>11}  {med_c_str:>11}")
    print(f"  Log-rank p = {lr.p_value:.4f}")
    print(f"  HR ({tors_label} vs {ctrl_label}): {hr:.3f}  95% CI ({hr_lo:.3f}\u2013{hr_hi:.3f})  p={p_cox:.4f}")


con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

for comp, match_col, match_id_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*70}")
    print(f"  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}  (C77 excluded)")
    print(f"{'#'*70}")

    df = con.execute(f"""
        WITH {C77_CTE},
        matched AS (
            SELECT DSYSRTKY, tx_group, first_tx_date, {match_id_col},
                   van_walraven_score, age_at_dx
            FROM opscc_propensity
            WHERE {match_col} = TRUE
              AND tx_group IN ('{tors_label}', '{ctrl_label}')
              AND DSYSRTKY NOT IN (SELECT DSYSRTKY FROM c77_patients)
        ),
        mbsf_summary AS (
            SELECT m.DSYSRTKY,
                MAX(CAST(m.RFRNC_YR AS INTEGER))                         AS last_enrl_year,
                MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                         THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END)   AS death_date
            FROM mbsf_all m
            JOIN matched p ON m.DSYSRTKY = p.DSYSRTKY
            GROUP BY m.DSYSRTKY
        )
        SELECT p.*, s.death_date,
               make_date(s.last_enrl_year, 12, 31) AS censor_date
        FROM matched p
        JOIN mbsf_summary s ON p.DSYSRTKY = s.DSYSRTKY
    """).df()

    df['event_date'] = df['death_date'].combine_first(df['censor_date'])
    df['event']      = df['death_date'].notna().astype(int)
    df['t_days']     = (df['event_date'] - df['first_tx_date']).dt.days
    df = df[(df['first_tx_date'].notna()) & (df['t_days'] >= 0)].copy()
    df['tors']       = (df['tx_group'] == tors_label).astype(int)

    print(f"N (C77 excluded): {len(df):,}  |  "
          f"{tors_label} deaths: {df[df.tors==1]['event'].sum()}  "
          f"{ctrl_label} deaths: {df[df.tors==0]['event'].sum()}")

    t1 = df['van_walraven_score'].quantile(1/3)
    t2 = df['van_walraven_score'].quantile(2/3)
    df['vw_group'] = pd.cut(df['van_walraven_score'],
                             bins=[-np.inf, t1, t2, np.inf],
                             labels=['Low', 'Mid', 'High'])
    print(f"Elixhauser tertiles: Low ≤{t1:.0f} | Mid {t1:.0f}–{t2:.0f} | High >{t2:.0f}")

    run_analysis("FULL MATCHED COHORT (all ages)", df, tors_label, ctrl_label, match_id_col)
    run_analysis("AGE < 75",  df[df['age_at_dx'] < 75], tors_label, ctrl_label, match_id_col)
    run_analysis("AGE \u2265 75", df[df['age_at_dx'] >= 75], tors_label, ctrl_label, match_id_col)
    for grp in ['Low', 'Mid', 'High']:
        run_analysis(f"ELIXHAUSER {grp} (VW tertile)",
                     df[df['vw_group'] == grp], tors_label, ctrl_label, match_id_col)

con.close()
