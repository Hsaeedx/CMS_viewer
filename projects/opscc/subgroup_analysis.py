"""
Subgroup survival analysis on PSM-matched cohort:
  1. Age < 75 restriction
  2. Stratified by Elixhauser van Walraven score (low / mid / high tertiles)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

DB_PATH = r"F:\CMS\cms_data.duckdb"

# ── 1. Pull matched cohort + survival ─────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")
df = con.execute("""
    WITH c77_patients AS (
        -- Patients with nodal disease (C77%) within ±90 days of first_hnc_date
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
    ),
    matched AS (
        SELECT DSYSRTKY, tx_group, first_tx_date, psm_match_id,
               van_walraven_score, age_at_dx
        FROM opscc_propensity
        WHERE psm_matched = TRUE
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
con.close()

df['event_date'] = df['death_date'].combine_first(df['censor_date'])
df['event']      = df['death_date'].notna().astype(int)
df['t_days']     = (df['event_date'] - df['first_tx_date']).dt.days
df = df[(df['first_tx_date'].notna()) & (df['t_days'] >= 0)].copy()
df['tors']       = (df['tx_group'] == 'TORS only').astype(int)

print(f"Matched cohort (C77 excluded): {len(df):,}  |  "
      f"TORS deaths: {df[df.tors==1]['event'].sum()}  "
      f"CT/CRT deaths: {df[df.tors==0]['event'].sum()}")

# ── 2. Define Elixhauser tertiles ─────────────────────────────────────────────
t1 = df['van_walraven_score'].quantile(1/3)
t2 = df['van_walraven_score'].quantile(2/3)
df['vw_group'] = pd.cut(
    df['van_walraven_score'],
    bins=[-np.inf, t1, t2, np.inf],
    labels=['Low', 'Mid', 'High']
)
print(f"\nElixhauser tertile cutoffs: Low <= {t1:.0f} | Mid {t1:.0f}-{t2:.0f} | High > {t2:.0f}")
print(df.groupby('vw_group', observed=True)['van_walraven_score'].describe()[['count','min','mean','max']].round(1))

# ── 3. Analysis function ──────────────────────────────────────────────────────
def run_analysis(label, subset):
    tors_s  = subset[subset['tors'] == 1]
    ctcrt_s = subset[subset['tors'] == 0]

    if len(tors_s) < 10 or len(ctcrt_s) < 10:
        print(f"\n  {label}: insufficient n (TORS={len(tors_s)}, CT/CRT={len(ctcrt_s)})")
        return

    # KM
    kmf_t = KaplanMeierFitter()
    kmf_c = KaplanMeierFitter()
    kmf_t.fit(tors_s['t_days'],  tors_s['event'])
    kmf_c.fit(ctcrt_s['t_days'], ctcrt_s['event'])

    def km_at(kmf, days):
        try:
            return kmf.survival_function_at_times(days).iloc[0]
        except Exception:
            return np.nan

    # Log-rank
    lr = logrank_test(tors_s['t_days'], ctcrt_s['t_days'],
                      tors_s['event'],  ctcrt_s['event'])

    # Cox (adjusted for age + van Walraven)
    cph_cols = ['t_days','event','tors','age_at_dx','van_walraven_score','psm_match_id']
    cph = CoxPHFitter()
    try:
        cph.fit(subset[cph_cols], duration_col='t_days', event_col='event',
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
    print(f"  N: TORS={len(tors_s):,}  CT/CRT={len(ctcrt_s):,}  "
          f"Deaths: TORS={int(tors_s['event'].sum())} ({100*tors_s['event'].mean():.1f}%)  "
          f"CT/CRT={int(ctcrt_s['event'].sum())} ({100*ctcrt_s['event'].mean():.1f}%)")
    print(f"  {'Timepoint':<10} {'TORS OS':>9} {'CT/CRT OS':>10}")
    print(f"  {'-'*32}")
    for yrs, days in [(1,365),(2,730),(3,1095),(5,1825)]:
        ts = km_at(kmf_t, days)
        cs = km_at(kmf_c, days)
        print(f"  {yrs}-year    {100*ts:>8.1f}%  {100*cs:>9.1f}%")
    med_t = kmf_t.median_survival_time_
    med_c = kmf_c.median_survival_time_
    med_t_str = f"{med_t/365.25:.1f} yr" if not np.isinf(med_t) else "NR"
    med_c_str = f"{med_c/365.25:.1f} yr" if not np.isinf(med_c) else "NR"
    print(f"  Median OS   {med_t_str:>9}  {med_c_str:>9}")
    print(f"  Log-rank p = {lr.p_value:.4f}")
    print(f"  HR (TORS vs CT/CRT): {hr:.3f}  95% CI ({hr_lo:.3f}-{hr_hi:.3f})  p={p_cox:.4f}")

# ── 4. Run subgroups ──────────────────────────────────────────────────────────

# Full matched cohort (reference)
run_analysis("FULL PSM-MATCHED COHORT (all ages)", df)

# Age < 75
run_analysis("AGE < 75", df[df['age_at_dx'] < 75])

# Age >= 75
run_analysis("AGE >= 75", df[df['age_at_dx'] >= 75])

# Elixhauser tertiles
for grp in ['Low', 'Mid', 'High']:
    sub = df[df['vw_group'] == grp]
    t1_val = df['van_walraven_score'].quantile(1/3)
    t2_val = df['van_walraven_score'].quantile(2/3)
    run_analysis(f"ELIXHAUSER {grp} (VW tertile)", sub)
