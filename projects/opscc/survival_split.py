import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import duckdb, pandas as pd, numpy as np
from lifelines import CoxPHFitter

DB_PATH = r"E:\CMS\cms_data.duckdb"
con = duckdb.connect(DB_PATH, read_only=True)

survival = con.execute("""
    WITH matched AS (
        SELECT DSYSRTKY, tx_group, first_tx_date, psm_match_id,
               van_walraven_score, age_at_dx
        FROM opscc_propensity WHERE psm_matched = TRUE
    ),
    mbsf_summary AS (
        SELECT m.DSYSRTKY,
            MAX(CAST(m.RFRNC_YR AS INTEGER)) AS last_enrl_year,
            MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                     THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END) AS death_date
        FROM mbsf_all m JOIN matched p ON m.DSYSRTKY = p.DSYSRTKY
        GROUP BY m.DSYSRTKY
    )
    SELECT p.*, s.death_date, make_date(s.last_enrl_year, 12, 31) AS censor_date
    FROM matched p JOIN mbsf_summary s ON p.DSYSRTKY = s.DSYSRTKY
""").df()
con.close()

survival['event_date'] = survival['death_date'].combine_first(survival['censor_date'])
survival['event']      = survival['death_date'].notna().astype(int)
survival['t_days']     = (survival['event_date'] - survival['first_tx_date']).dt.days
survival = survival[(survival['first_tx_date'].notna()) & (survival['t_days'] >= 0)].copy()
survival['tors']       = (survival['tx_group'] == 'TORS only').astype(int)

# Time-split Cox: early vs late HR
print('-- Time-split HR (landmark analysis) -----------------------------------')
for label, t_start, t_end in [('0-1 yr', 0, 365), ('1-3 yr', 365, 1095), ('>3 yr', 1095, 9999)]:
    sub = survival[survival['t_days'] > t_start].copy()
    sub['t_adj']     = np.minimum(sub['t_days'], t_end) - t_start
    sub['event_adj'] = np.where(sub['t_days'] <= t_end, sub['event'], 0)
    sub = sub[sub['t_adj'] > 0]
    cph = CoxPHFitter()
    cph.fit(sub[['t_adj','event_adj','tors','age_at_dx','van_walraven_score']],
            duration_col='t_adj', event_col='event_adj')
    row = cph.summary.loc['tors']
    print(f"  {label:<8}: HR={row['exp(coef)']:.3f}  "
          f"95% CI ({row['exp(coef) lower 95%']:.3f}-{row['exp(coef) upper 95%']:.3f})  "
          f"p={row['p']:.4f}  n={len(sub):,}  events={int(sub['event_adj'].sum())}")

# Stratified Cox (strata=match pair, accounts for paired design)
print('')
print('-- Stratified Cox (strata=psm_match_id, adjusted for age + van Walraven)')
cph2 = CoxPHFitter()
cph2.fit(
    survival[['t_days','event','tors','age_at_dx','van_walraven_score','psm_match_id']],
    duration_col='t_days', event_col='event', strata=['psm_match_id']
)
row2 = cph2.summary.loc['tors']
print(f"  TORS HR={row2['exp(coef)']:.3f}  "
      f"95% CI ({row2['exp(coef) lower 95%']:.3f}-{row2['exp(coef) upper 95%']:.3f})  "
      f"p={row2['p']:.6f}")
