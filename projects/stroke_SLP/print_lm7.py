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
           o.days_to_gtube, o.days_to_recur_stroke,
           o.pre_stroke_tube
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

sex_d   = pd.get_dummies(lm_slp['sex'],         prefix='sex',    drop_first=True)
strk_d  = pd.get_dummies(lm_slp['stroke_type'], prefix='stroke', drop_first=True)
lm_slp  = pd.concat([lm_slp, sex_d, strk_d], axis=1)

BASE_COVS = (['slp_15_30d','slp_31_90d','age_at_adm','van_walraven_score','index_los',
              'mech_vent','trach_placed','prior_stroke','dementia',
              'dysphagia_poa','aspiration_poa']
             + list(sex_d.columns) + list(strk_d.columns))

MODELS = [
    ('Aspiration PNA', 'asp_lm',   'days_to_aspiration_lm_filled',  False, 0.0),
    ('PEG/G-tube',     'gtube_lm', 'days_to_gtube_lm_filled',       True,  0.1),
    ('Mortality',      'died_lm',  'censor_lm',                     False, 0.0),
    ('Recur stroke',   'recur_lm', 'days_to_recur_stroke_lm_filled', False, 0.0),
]

STRATA = [
    ('home', 'Home'),
    ('snf',  'SNF'),
    ('irf',  'IRF'),
    ('hha',  'HHA'),
]

print("=== LM7: COX PH STRATIFIED BY DISCHARGE DESTINATION ===\n")
print(f"{'Destination':<8} {'Outcome':<16} {'Comparison':<20} {'N':>7} {'Ev':>6} "
      f"{'HR':>6} {'95% CI':<16} {'p':>8} {'E':>5}")
print("-"*100)

for dschg_code, dschg_label in STRATA:
    d_sub = lm_slp[lm_slp['dschg_group'] == dschg_code]
    n_d   = len(d_sub)
    n_by_grp = {g: int((d_sub['grp'] == g).sum()) for g in ['0-14d','15-30d','31-90d']}
    print(f"\n{'='*8} {dschg_label} (N={n_d:,})  "
          f"0-14d:{n_by_grp['0-14d']:,}  15-30d:{n_by_grp['15-30d']:,}  31-90d:{n_by_grp['31-90d']:,}")

    for label, ev_col, dur_col, excl_peg, pen in MODELS:
        sub = d_sub.copy()
        if excl_peg:
            sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
        cov_use = list(BASE_COVS)
        if not excl_peg and 'peg_placed' not in cov_use:
            cov_use = ['peg_placed'] + cov_use

        raw = sub[[dur_col, ev_col] + cov_use].rename(
            columns={dur_col: 'duration', ev_col: 'event'}).dropna()
        cov_ok = [c for c in cov_use if raw[c].std() > 0.01]
        cox_df = raw[['duration', 'event'] + cov_ok]
        n_ev   = int(cox_df['event'].sum())

        if n_ev < 20:
            print(f"  {label:<16} (too few events: {n_ev})")
            continue
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(cox_df, duration_col='duration', event_col='event', show_progress=False)
            for tv, comp in [('slp_15_30d','15-30d vs 0-14d'),('slp_31_90d','31-90d vs 0-14d')]:
                if tv not in cph.summary.index: continue
                r  = cph.summary.loc[tv]
                hr = np.exp(r['coef']); lo = np.exp(r['coef lower 95%']); hi = np.exp(r['coef upper 95%'])
                p  = r['p']
                ev_v = hr + np.sqrt(hr * abs(hr-1))
                ps = '<.0001' if p < 0.0001 else f'{p:.4f}'
                print(f"  {label:<16} {comp:<20} {len(cox_df):>7,} {n_ev:>6,} "
                      f"{hr:>6.2f} [{lo:.2f},{hi:.2f}]     {ps:>8} {ev_v:>5.2f}")
        except Exception as e:
            print(f"  {label}: ERROR {e}")
