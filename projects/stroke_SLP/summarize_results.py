import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb, numpy as np, pandas as pd
from lifelines import CoxPHFitter

DB_PATH = Path(os.getenv("duckdb_database", "cms_data.duckdb"))

con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT p.DSYSRTKY, p.age_at_adm, p.sex, p.stroke_type,
           p.index_los, p.mech_vent, p.peg_placed, p.trach_placed,
           p.dschg_status, p.dysphagia_poa, p.aspiration_poa,
           p.prior_stroke, p.dementia, p.van_walraven_score,
           p.slp_outpt_any_90d, p.days_to_slp_outpt,
           o.days_to_death, o.days_to_aspiration,
           o.days_to_gtube, o.days_to_recur_stroke
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()

LANDMARK, MAX_FOLLOW_LM = 30, 1795

lm = df[df['days_to_death'].isna() | (df['days_to_death'] > LANDMARK)].copy()
lm_slp = lm[lm['slp_outpt_any_90d'] == 1].copy()

def slp_group(d):
    if pd.isna(d): return None
    if d <= 14: return '0-14d'
    if d <= 30: return '15-30d'
    return '31-90d'

lm_slp['grp'] = lm_slp['days_to_slp_outpt'].apply(slp_group)

def map_dschg(c):
    c = str(c).strip() if not pd.isna(c) else ''
    if c in ('01','08'): return 'home'
    if c == '06': return 'hha'
    if c in ('03','61'): return 'snf'
    if c == '62': return 'irf'
    return 'other'

lm_slp['dschg_group'] = lm_slp['dschg_status'].apply(map_dschg)

print("=== TIMING GROUP SIZES ===")
vc = lm_slp['grp'].value_counts().sort_index()
print(vc)
print(f"Total: {len(lm_slp):,}")

print("\n=== DISCHARGE DESTINATION BY TIMING GROUP (%) ===")
ct = pd.crosstab(lm_slp['grp'], lm_slp['dschg_group'], normalize='index').round(3)*100
print(ct.to_string())

print("\n=== BASELINE BY TIMING GROUP ===")
for g in ['0-14d','15-30d','31-90d']:
    sub = lm_slp[lm_slp['grp']==g]
    print(f"{g}: N={len(sub):,}  age={sub['age_at_adm'].mean():.1f}  "
          f"VW={sub['van_walraven_score'].mean():.1f}  "
          f"LOS={sub['index_los'].mean():.1f}  "
          f"dysphagia_poa={sub['dysphagia_poa'].mean()*100:.1f}%  "
          f"dementia={sub['dementia'].mean()*100:.1f}%")

for col in ['days_to_death','days_to_aspiration','days_to_gtube','days_to_recur_stroke']:
    lm_slp[col] = lm_slp[col].astype('float64')
    lm_slp[col+'_lm'] = np.where(
        lm_slp[col].notna() & (lm_slp[col] > LANDMARK),
        (lm_slp[col] - LANDMARK).clip(lower=0.5), np.nan)

lm_slp['censor_lm'] = np.where(
    lm_slp['days_to_death_lm'].notna(),
    lm_slp['days_to_death_lm'].clip(upper=MAX_FOLLOW_LM), MAX_FOLLOW_LM)
lm_slp['died_lm']  = lm_slp['days_to_death_lm'].notna().astype(int)
lm_slp['asp_lm']   = lm_slp['days_to_aspiration_lm'].notna().astype(int)
lm_slp['gtube_lm'] = lm_slp['days_to_gtube_lm'].notna().astype(int)
lm_slp['recur_lm'] = lm_slp['days_to_recur_stroke_lm'].notna().astype(int)

for ev, dur in [('asp_lm','days_to_aspiration_lm'),
                ('gtube_lm','days_to_gtube_lm'),
                ('recur_lm','days_to_recur_stroke_lm')]:
    lm_slp[dur+'_filled'] = lm_slp[dur].fillna(lm_slp['censor_lm']).clip(lower=0.5)

lm_slp['slp_15_30d'] = (lm_slp['grp'] == '15-30d').astype(int)
lm_slp['slp_31_90d'] = (lm_slp['grp'] == '31-90d').astype(int)

dschg_d = pd.get_dummies(lm_slp['dschg_group'], prefix='dschg', drop_first=True)
dschg_d = dschg_d.loc[:, dschg_d.std() > 0.01]
sex_d   = pd.get_dummies(lm_slp['sex'],          prefix='sex',   drop_first=True)
strk_d  = pd.get_dummies(lm_slp['stroke_type'],  prefix='stroke',drop_first=True)
lm_slp  = pd.concat([lm_slp, dschg_d, sex_d, strk_d], axis=1)

COVS = (['slp_15_30d','slp_31_90d','age_at_adm','van_walraven_score','index_los',
         'mech_vent','trach_placed','prior_stroke','dementia','dysphagia_poa','aspiration_poa']
        + list(dschg_d.columns) + list(sex_d.columns) + list(strk_d.columns))

print("\n=== COX PH RESULTS ===")
models = [
    ('Aspiration PNA', 'asp_lm',    'days_to_aspiration_lm_filled',  False, 0.0),
    ('PEG/G-tube',     'gtube_lm',  'days_to_gtube_lm_filled',       True,  0.1),
    ('Mortality',      'died_lm',   'censor_lm',                     False, 0.0),
    ('Recurrent stroke','recur_lm', 'days_to_recur_stroke_lm_filled', False, 0.0),
]
for label, ev, dur, excl_peg, pen in models:
    sub = lm_slp[lm_slp['peg_placed']==0].copy() if excl_peg else lm_slp.copy()
    covs = list(COVS)
    if not excl_peg and 'peg_placed' not in covs:
        covs = ['peg_placed'] + covs
    cdf = sub[[dur,ev]+covs].rename(columns={dur:'duration',ev:'event'}).dropna()
    n_ev = int(cdf['event'].sum())
    try:
        cph = CoxPHFitter(penalizer=pen)
        cph.fit(cdf, duration_col='duration', event_col='event', show_progress=False)
        print(f'\n{label}  (N={len(cdf):,}, events={n_ev:,}, {100*n_ev/len(cdf):.1f}%)')
        for tv, comp in [('slp_15_30d','15-30d vs 0-14d'),('slp_31_90d','31-90d vs 0-14d')]:
            if tv not in cph.summary.index: continue
            r = cph.summary.loc[tv]
            hr = np.exp(r['coef']); lo = np.exp(r['coef lower 95%']); hi = np.exp(r['coef upper 95%'])
            p = r['p']
            ev_val = hr + np.sqrt(hr * abs(hr-1))
            ps = '<0.0001' if p < 0.0001 else f'{p:.4f}'
            print(f'  {comp}: HR={hr:.2f} [{lo:.2f},{hi:.2f}]  p={ps}  E={ev_val:.2f}')
    except Exception as e:
        print(f'\n{label}: ERROR - {e}')
