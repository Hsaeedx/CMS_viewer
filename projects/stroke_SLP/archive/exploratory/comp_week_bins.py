"""
comp_week_bins.py

Compares two study designs (week 0 = days 1-7 always excluded):

Design A — Week-by-week vs Week 6+ reference:
  Wk1 (8-14d)   vs  Wk6+ (43-90d) ref
  Wk2 (15-21d)  vs  Wk6+ (43-90d) ref
  Wk3 (22-28d)  vs  Wk6+ (43-90d) ref
  Wk4 (29-35d)  vs  Wk6+ (43-90d) ref

Design B — Early (first 4 post-discharge weeks) vs Week 5+ reference:
  Early (8-35d)  vs  Wk5+ (36-90d) ref

PSM fitted per comparison (global). TV Cox for each outcome.
Days 1-7 excluded from all analyses.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
import warnings
warnings.filterwarnings('ignore')

DB_PATH       = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
RANDOM_SEED   = 42
MAX_FOLLOW    = 365
TV_COVARIATES = ['age_at_adm', 'van_walraven_score', 'index_los']
CONT_VARS     = ['age_at_adm', 'index_los', 'van_walraven_score', 'adm_year']
BINARY_VARS   = ['afib', 'hypertension', 'mech_vent', 'prior_stroke', 'dual_eligible']
CAT_VARS      = ['sex', 'race', 'stroke_type', 'drg_group', 'adm_source', 'rucc_group']

OUTCOMES = [
    ('Aspiration-related PNA', 'days_to_asp_related', 'days_to_death', False),
    ('PEG/G-tube',             'days_to_gtube',       'days_to_death', True),
    ('Mortality',              'days_to_death',        None,            False),
]

# Design A: week-by-week, reference = Week 6+ (days 43-90)
DESIGN_A = {
    'name': 'Design A — Week-by-week (ref: Wk6+ days 43-90)',
    'ref_lo': 43, 'ref_hi': 90,
    'comparisons': [
        ('Wk1 (8-14d)',  8, 14),
        ('Wk2 (15-21d)', 15, 21),
        ('Wk3 (22-28d)', 22, 28),
        ('Wk4 (29-35d)', 29, 35),
    ]
}

# Design B: early (weeks 1-4) vs Week 5+ reference (days 36-90)
DESIGN_B = {
    'name': 'Design B — Early vs Late (ref: Wk5+ days 36-90)',
    'ref_lo': 36, 'ref_hi': 90,
    'comparisons': [
        ('Early (8-35d)', 8, 35),
    ]
}


def bucket_drg(d):
    if pd.isna(d): return 'Other'
    try: n = int(str(d).strip())
    except: return 'Other'
    if 61 <= n <= 69: return 'Medical_stroke'
    if 20 <= n <= 38: return 'Neurosurgical'
    if 52 <= n <= 60: return 'Spinal'
    if 70 <= n <= 74: return 'TIA_headache'
    return 'Other'


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_prop = con.execute("""
    SELECT DSYSRTKY, days_to_slp_outpt,
           age_at_adm, index_los, van_walraven_score, adm_year,
           sex, race, stroke_type, DRG_CD AS drg_cd, adm_source,
           mech_vent, peg_placed, prior_stroke, afib, hypertension,
           rucc_group, dual_eligible
    FROM stroke_propensity
    WHERE days_to_slp_outpt BETWEEN 8 AND 90  -- week 0 excluded
""").df()
df_out = con.execute("""
    SELECT DSYSRTKY, days_to_death, days_to_aspiration, days_to_gtube,
           days_to_pneumonia, first_pneumonia_code, pre_stroke_tube
    FROM stroke_outcomes
""").df()
con.close()

for col in CONT_VARS:
    df_prop[col] = df_prop[col].fillna(df_prop[col].median())
df_prop['drg_group']    = df_prop['drg_cd'].apply(bucket_drg)
df_prop['adm_source']   = df_prop['adm_source'].fillna('Unknown').astype(str)
df_prop['adm_year']     = df_prop['adm_year'].fillna(df_prop['adm_year'].median()).astype(int)
df_prop['rucc_group']   = df_prop['rucc_group'].fillna('Unknown').astype(str)
df_prop['dual_eligible'] = df_prop['dual_eligible'].fillna(0).astype(int)
df_all = df_prop.merge(df_out, on='DSYSRTKY', how='inner')
# J18+J69 composite: aspiration-related pneumonia
df_all['days_to_asp_related'] = np.where(
    df_all['first_pneumonia_code'].isin(['J18', 'J69']),
    df_all['days_to_pneumonia'],
    np.nan
)
print(f"  {len(df_all):,} rows after excluding week 0 (days 1-7)")


# ── PSM ────────────────────────────────────────────────────────────────────────
def run_psm(df_treat, df_ctrl):
    df = pd.concat([df_treat.assign(_treated=1), df_ctrl.assign(_treated=0)])
    n_treat = int(df['_treated'].sum())
    dummies  = pd.get_dummies(df[CAT_VARS], drop_first=True)
    X        = pd.concat([df[CONT_VARS + BINARY_VARS].astype(float), dummies], axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr.fit(X_scaled, df['_treated'])
    ps       = np.clip(lr.predict_proba(X_scaled)[:, 1], 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps / (1 - ps))
    df = df.reset_index(drop=True)
    df['logit_ps'] = logit_ps
    caliper  = 0.2 * logit_ps.std()
    treated  = df[df['_treated'] == 1].reset_index(drop=True)
    control  = df[df['_treated'] == 0].reset_index(drop=True)
    tree     = cKDTree(control['logit_ps'].values.reshape(-1, 1))
    rng      = np.random.default_rng(RANDOM_SEED)
    matched_pairs, used_ctrl = [], set()
    for i in rng.permutation(len(treated)):
        dists, idxs = tree.query([[treated.loc[i, 'logit_ps']]], k=min(50, len(control)))
        for dist, idx in zip(dists[0], idxs[0]):
            if dist > caliper: break
            cid = control.loc[idx, 'DSYSRTKY']
            if cid not in used_ctrl:
                matched_pairs.append((treated.loc[i, 'DSYSRTKY'], cid))
                used_ctrl.add(cid)
                break
    n_matched = len(matched_pairs)
    pct = 100 * n_matched / n_treat
    return {pid for pair in matched_pairs for pid in pair}, n_matched, n_treat, pct, caliper


# ── TV Cox ─────────────────────────────────────────────────────────────────────
def build_tv_df(df, event_col, competing_col, treat_label):
    records = []
    for _, row in df.iterrows():
        slp_day  = float(row['days_to_slp_outpt'])
        ev_day   = row[event_col]
        comp_day = row[competing_col] if competing_col and pd.notna(row[competing_col]) else np.nan
        end_time = min([float(MAX_FOLLOW)]
                       + ([float(ev_day)]   if pd.notna(ev_day)   else [])
                       + ([float(comp_day)] if pd.notna(comp_day) else []))
        final_event = int(pd.notna(ev_day) and float(ev_day) <= MAX_FOLLOW
                          and float(ev_day) == end_time)
        grp_flag = 1 if row['_treat_grp'] == treat_label else 0
        base = {c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in TV_COVARIATES}
        if end_time <= slp_day:
            records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                            'stop': max(end_time, 0.5), 'trt': 0,
                            'event': final_event, **base})
        else:
            if slp_day > 0:
                records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                                'stop': slp_day, 'trt': 0, 'event': 0, **base})
            records.append({'id': row['DSYSRTKY'], 'start': slp_day,
                            'stop': max(end_time, slp_day + 0.5),
                            'trt': grp_flag, 'event': final_event, **base})
    return pd.DataFrame(records)

def run_tv_cox(df, event_col, competing_col, treat_label):
    tv = build_tv_df(df, event_col, competing_col, treat_label)
    tv = tv.dropna(subset=TV_COVARIATES)
    for col in TV_COVARIATES:
        sd = tv[col].std()
        if sd > 0: tv[col] = (tv[col] - tv[col].mean()) / sd
    n_pts, n_evts = tv['id'].nunique(), int(tv['event'].sum())
    if n_evts < 10:
        return None, n_pts, n_evts
    try:
        ctv = CoxTimeVaryingFitter()
        ctv.fit(tv, id_col='id', start_col='start', stop_col='stop',
                event_col='event', show_progress=False)
        r = ctv.summary.loc['trt']
        hr, lo, hi, p = (np.exp(r['coef']), np.exp(r['coef lower 95%']),
                         np.exp(r['coef upper 95%']), r['p'])
        return (hr, lo, hi, p), n_pts, n_evts
    except Exception as e:
        print(f"      ERROR: {e}")
        return None, n_pts, n_evts


# ── Run both options ───────────────────────────────────────────────────────────
def run_option(opt):
    ref_lo, ref_hi = opt['ref_lo'], opt['ref_hi']
    df_ref = df_all[(df_all['days_to_slp_outpt'] >= ref_lo) &
                    (df_all['days_to_slp_outpt'] <= ref_hi)].copy()
    df_ref['_treat_grp'] = f"ref ({ref_lo}-{ref_hi}d)"

    print(f"\n{'='*75}")
    print(f"  {opt['name']}")
    print(f"  Reference: days {ref_lo}-{ref_hi}  n={len(df_ref):,}")
    print(f"{'='*75}")

    all_rows = []
    for comp_label, lo, hi in opt['comparisons']:
        df_treat = df_all[(df_all['days_to_slp_outpt'] >= lo) &
                          (df_all['days_to_slp_outpt'] <= hi)].copy()
        df_treat['_treat_grp'] = comp_label

        matched_ids, n_matched, n_treat, pct, caliper = run_psm(df_treat, df_ref)
        print(f"\n  {comp_label} vs ref: {n_matched:,}/{n_treat:,} matched "
              f"({pct:.1f}%)  caliper={caliper:.4f}")

        df_comp = df_all[df_all['DSYSRTKY'].isin(matched_ids)].copy()
        df_comp['_treat_grp'] = df_comp['days_to_slp_outpt'].apply(
            lambda d: comp_label if lo <= d <= hi else f"ref ({ref_lo}-{ref_hi}d)")

        print(f"  {'Outcome':<18} {'N':>6} {'Ev':>5} {'Ev%':>5}   HR [95% CI]              p")
        print(f"  {'-'*70}")

        for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
            sub = df_comp.copy()
            if excl_peg:
                sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
            res, n, n_ev = run_tv_cox(sub, ev_col, comp_col, comp_label)
            ev_pct = f"{100*n_ev/n:.1f}%" if n > 0 else "—"
            if res:
                hr, lo2, hi2, p = res
                sig = '***' if p < 0.001 else '** ' if p < 0.01 else '*  ' if p < 0.05 else '   '
                res_str = f"HR={hr:.2f} [{lo2:.2f}-{hi2:.2f}]  p={p:.4f}  {sig}"
            else:
                res_str = "<10 events" if n_ev < 10 else "failed"
            print(f"  {out_label:<18} {n:>6} {n_ev:>5} {ev_pct:>5}   {res_str}")
            all_rows.append({
                'option': opt['name'], 'comparison': comp_label,
                'ref': f"days {ref_lo}-{ref_hi}",
                'outcome': out_label, 'n': n, 'events': n_ev,
                'hr': res[0] if res else None, 'lo': res[1] if res else None,
                'hi': res[2] if res else None, 'p': res[3] if res else None,
            })
    return all_rows


rows_a = run_option(DESIGN_A)
rows_b = run_option(DESIGN_B)

# ── Design A summary table ─────────────────────────────────────────────────────
df_a = pd.DataFrame(rows_a)
df_b = pd.DataFrame(rows_b)

def fmt_hr(r):
    if r is None or pd.isna(r['hr']): return '—'
    sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else ''
    return f"HR={r['hr']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}] p={r['p']:.4f}{sig}"

outcomes_order = ['Aspiration-related PNA', 'PEG/G-tube', 'Mortality']
weeks_order    = ['Wk1 (8-14d)', 'Wk2 (15-21d)', 'Wk3 (22-28d)', 'Wk4 (29-35d)']

print(f"\n\n{'='*90}")
print("  DESIGN A — DOSE-RESPONSE BY WEEK  (ref: Wk6+ days 43-90)")
print(f"{'='*90}")
print(f"  {'Outcome':<18} {'Wk1 (8-14d)':<35} {'Wk2 (15-21d)':<35} {'Wk3 (22-28d)':<35} {'Wk4 (29-35d)'}")
print(f"  {'-'*140}")
for out in outcomes_order:
    row_strs = []
    for wk in weeks_order:
        sub = df_a[(df_a['outcome'] == out) & (df_a['comparison'] == wk)]
        r = sub.iloc[0].to_dict() if not sub.empty else None
        row_strs.append(fmt_hr(r))
    print(f"  {out:<18} {row_strs[0]:<35} {row_strs[1]:<35} {row_strs[2]:<35} {row_strs[3]}")

print(f"\n\n{'='*90}")
print("  DESIGN B — EARLY (8-35d) vs LATE (ref: Wk5+ days 36-90)")
print(f"{'='*90}")
print(f"  {'Outcome':<18} {'N':>6} {'Events':>7}   Result")
print(f"  {'-'*70}")
for out in outcomes_order:
    sub = df_b[(df_b['outcome'] == out) & (df_b['comparison'] == 'Early (8-35d)')]
    if sub.empty: continue
    r = sub.iloc[0].to_dict()
    ev_pct = f"{100*r['events']/r['n']:.1f}%" if r['n'] > 0 else '—'
    print(f"  {out:<18} {r['n']:>6} {r['events']:>7} {ev_pct:>5}   {fmt_hr(r)}")

print("\n* p<0.05  ** p<0.01  *** p<0.001")
print("Week 0 (days 1-7) excluded from all analyses.")
print("Design A: 4 weekly comparisons each vs Wk6+ (days 43-90).")
print("Design B: single comparison, early SLP (days 8-35) vs Wk5+ (days 36-90).")