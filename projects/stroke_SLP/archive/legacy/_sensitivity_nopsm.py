"""
_sensitivity_nopsm.py

Sensitivity: primary bins (1-14d / 15-30d / 31-90d) with covariate-adjusted
TV Cox on the full UNMATCHED cohort — no PSM.

Compares results against the PSM-based primary analysis.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb, numpy as np, pandas as pd
from lifelines import CoxTimeVaryingFitter

DB_PATH    = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
MAX_FOLLOW = 365
TV_COVARIATES = ['age_at_adm', 'van_walraven_score', 'index_los']

COMPARISONS = [
    ('A', '1-14d',  '31-90d'),
    ('B', '15-30d', '31-90d'),
]
OUTCOMES = [
    ('Aspiration PNA', 'days_to_aspiration', 'days_to_death', False),
    ('PEG/G-tube',     'days_to_gtube',      'days_to_death', True),
    ('Mortality',      'days_to_death',       None,            False),
]

PRIMARY_HRS = {
    ('A', 'Aspiration PNA'): '0.85 [0.71-1.03] p=0.094',
    ('A', 'PEG/G-tube'):     '0.69 [0.52-0.91] p=0.010',
    ('A', 'Mortality'):      '1.07 [0.96-1.19] p=0.226',
    ('B', 'Aspiration PNA'): '0.79 [0.66-0.95] p=0.010',
    ('B', 'PEG/G-tube'):     '0.70 [0.54-0.92] p=0.011',
    ('B', 'Mortality'):      '0.98 [0.89-1.08] p=0.745',
}

print("Loading data...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_all = con.execute("""
    SELECT p.DSYSRTKY, p.slp_timing_group, p.days_to_slp_outpt,
           p.age_at_adm, p.index_los, p.van_walraven_score,
           p.dysphagia_poa, p.peg_placed,
           o.days_to_death, o.days_to_aspiration,
           o.days_to_gtube, o.pre_stroke_tube
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
    WHERE p.slp_timing_group IN ('1-14d', '15-30d', '31-90d')
""").df()
con.close()
counts = df_all['slp_timing_group'].value_counts()
print(f"  Loaded {len(df_all):,}  |  " +
      "  ".join(f"{g}:{counts.get(g,0):,}" for g in ['1-14d','15-30d','31-90d']))


def build_tv_df(df, event_col, competing_col, treat_grp):
    records = []
    for _, row in df.iterrows():
        slp_day  = float(row['days_to_slp_outpt'])
        ev_day   = row[event_col]
        comp_day = (row[competing_col]
                    if competing_col and pd.notna(row[competing_col]) else np.nan)
        candidates = [float(MAX_FOLLOW)]
        if pd.notna(ev_day):   candidates.append(float(ev_day))
        if pd.notna(comp_day): candidates.append(float(comp_day))
        end_time = min(candidates)
        final_event = int(pd.notna(ev_day) and float(ev_day) <= MAX_FOLLOW
                          and float(ev_day) == end_time)
        group_flag = 1 if row['slp_timing_group'] == treat_grp else 0
        base = {c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in TV_COVARIATES}
        if end_time <= slp_day:
            records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                            'stop': max(end_time, 0.5),
                            'trt': 0, 'event': final_event, **base})
        else:
            if slp_day > 0:
                records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                                'stop': slp_day, 'trt': 0, 'event': 0, **base})
            records.append({'id': row['DSYSRTKY'], 'start': slp_day,
                            'stop': max(end_time, slp_day + 0.5),
                            'trt': group_flag, 'event': final_event, **base})
    return pd.DataFrame(records)


def run_tv_cox(df, event_col, competing_col, treat_grp):
    tv = build_tv_df(df, event_col, competing_col, treat_grp)
    tv = tv.dropna(subset=TV_COVARIATES)
    for col in ['age_at_adm', 'van_walraven_score', 'index_los']:
        sd = tv[col].std()
        if sd > 0:
            tv[col] = (tv[col] - tv[col].mean()) / sd
    n_pts, n_evts = tv['id'].nunique(), int(tv['event'].sum())
    if n_evts < 10:
        return None, n_pts, n_evts
    try:
        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(tv, id_col='id', start_col='start', stop_col='stop',
                event_col='event', show_progress=False)
        r = ctv.summary.loc['trt']
        hr, lo, hi, p = (np.exp(r['coef']), np.exp(r['coef lower 95%']),
                         np.exp(r['coef upper 95%']), r['p'])
        return (hr, lo, hi, p), n_pts, n_evts
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, n_pts, n_evts


print()
rows = []
for comp_label, treat_grp, ctrl_grp in COMPARISONS:
    df_comp = df_all[df_all['slp_timing_group'].isin([treat_grp, ctrl_grp])].copy()
    n_treat = (df_comp['slp_timing_group'] == treat_grp).sum()
    n_ctrl  = (df_comp['slp_timing_group'] == ctrl_grp).sum()
    print(f"Comparison {comp_label}: {treat_grp} vs {ctrl_grp} (ref)  "
          f"[treated={n_treat:,}  ref={n_ctrl:,}]")
    for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
        sub = df_comp.copy()
        if excl_peg:
            sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
        res, n, n_ev = run_tv_cox(sub, ev_col, comp_col, treat_grp)
        if res:
            hr, lo, hi, p = res
            p_str = '<0.0001' if p < 0.0001 else f'{p:.4f}'
            sens  = f"HR={hr:.2f} [{lo:.2f}-{hi:.2f}] p={p_str}"
        else:
            sens = "failed"
        rows.append((comp_label, out_label, sens))
        print(f"  {out_label}: n={n:,} ev={n_ev:,}  {sens}")

print()
print("="*90)
print("COMPARISON: PSM-based (primary) vs No-PSM (sensitivity)")
print("="*90)
print(f"{'Comp':<4} {'Outcome':<18} {'PSM — primary':<38} {'No PSM — sensitivity'}")
print("-"*90)
for comp_label, out_label, sens in rows:
    primary = PRIMARY_HRS.get((comp_label, out_label), 'N/A')
    print(f"{comp_label:<4} {out_label:<18} {primary:<38} {sens}")
