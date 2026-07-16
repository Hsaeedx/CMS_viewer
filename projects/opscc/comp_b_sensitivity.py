"""
comp_b_sensitivity.py
Sensitivity analyses for Comparison B (TORS+RT vs CT/CRT), testing two
alternative population restrictions on nodal disease (C77):

  Version 1 — TORS+RT all  vs CT/CRT N0 only   (asymmetric: all surgical vs node-neg CRT)
  Version 2 — TORS+RT N0   vs CT/CRT N0 only   (symmetric: both arms node-negative)

Both versions exclude has_nodal_dx from the PS model (no within-arm variance
in CT/CRT once restricted to N0), and are compared against the main analysis:

  Main      — TORS+RT all  vs CT/CRT all  (has_nodal_dx in PS model, from run_pipeline)

Standalone — does not modify opscc_propensity or any pipeline tables.
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

DB_PATH = r"F:\CMS\cms_data.duckdb"

# PS model features — has_nodal_dx excluded (near-zero variance in N0-restricted CT/CRT)
FEATURE_COLS = [
    'age_at_dx', 'male', 'white', 'black', 'hispanic', 'asian_pi',
    'van_walraven_score', 'coag', 'chf', 'cpd', 'rf',
    'dx_year',
    'subsite_c01', 'subsite_c09', 'subsite_c10',
    'region_south', 'region_midwest', 'region_west',
]

CONTINUOUS_VARS = ['age_at_dx', 'van_walraven_score', 'dx_year']
BINARY_VARS = [
    'male', 'white', 'black', 'hispanic', 'asian_pi',
    'coag', 'chf', 'cpd', 'rf',
    'subsite_c01', 'subsite_c09', 'subsite_c10',
    'region_south', 'region_midwest', 'region_west',
]


def smd_continuous(a, b):
    diff = a.mean() - b.mean()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return diff / pooled if pooled > 0 else 0.0

def smd_binary(a, b):
    p1, p2 = a.mean(), b.mean()
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else 0.0


def run_psm(df_raw, label, tors_label='TORS + RT', ctrl_label='CT/CRT'):
    df = df_raw[df_raw['tx_group'].isin([tors_label, ctrl_label])].copy()
    df['treatment'] = (df['tx_group'] == tors_label).astype(int)

    n_tors = (df['treatment'] == 1).sum()
    n_ctrl = (df['treatment'] == 0).sum()
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N: {tors_label}={n_tors:,}   {ctrl_label} (filtered)={n_ctrl:,}")
    print(f"{'='*70}")

    # Feature engineering
    df['male']     = (df['sex']  == 'Male').astype(int)
    df['white']    = (df['race'] == 'White').astype(int)
    df['black']    = (df['race'] == 'Black').astype(int)
    df['hispanic'] = (df['race'] == 'Hispanic').astype(int)
    df['asian_pi'] = (df['race'] == 'Asian/PI').astype(int)
    df['van_walraven_score'] = df['van_walraven_score'].fillna(0)
    for flag in ['coag', 'chf', 'cpd', 'rf']:
        df[flag] = df[flag].fillna(0).astype(int)
    df['dx_year'] = df['dx_year'].fillna(df['dx_year'].median()).astype(int)
    df['subsite_c01'] = (df['subsite'] == 'C01').astype(int)
    df['subsite_c09'] = (df['subsite'] == 'C09').astype(int)
    df['subsite_c10'] = (df['subsite'] == 'C10').astype(int)
    df['region_south']   = (df['census_region'] == 'South').astype(int)
    df['region_midwest'] = (df['census_region'] == 'Midwest').astype(int)
    df['region_west']    = (df['census_region'] == 'West').astype(int)

    X = df[FEATURE_COLS].values
    y = df['treatment'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ps_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    ps_model.fit(X_scaled, y)

    df['ps']       = ps_model.predict_proba(X_scaled)[:, 1]
    df['logit_ps'] = np.log(df['ps'] / (1 - df['ps']))

    auc = roc_auc_score(y, df['ps'])
    print(f"\nPS model C-statistic (AUC): {auc:.3f}")

    caliper = 0.2 * df['logit_ps'].std()
    print(f"Caliper (0.2 × SD logit PS): {caliper:.4f}")

    df = df.reset_index(drop=True)
    tors_idx = df.index[df['treatment'] == 1].tolist()
    ctrl_idx = df.index[df['treatment'] == 0].tolist()

    rng = np.random.default_rng(42)
    rng.shuffle(tors_idx)

    logit_ctrl     = df.loc[ctrl_idx, 'logit_ps'].values
    ctrl_available = np.ones(len(ctrl_idx), dtype=bool)
    matched_pairs  = []

    for ti in tors_idx:
        lp = df.loc[ti, 'logit_ps']
        dists = np.abs(logit_ctrl - lp)
        dists[~ctrl_available] = np.inf
        best = np.argmin(dists)
        if dists[best] <= caliper:
            matched_pairs.append((ti, ctrl_idx[best]))
            ctrl_available[best] = False

    n_matched = len(matched_pairs)
    print(f"\nMatching: {n_matched:,} / {n_tors:,} matched  ({100*n_matched/n_tors:.1f}%)")

    tors_rows = [p[0] for p in matched_pairs]
    ctrl_rows = [p[1] for p in matched_pairs]
    match_ids = list(range(n_matched))

    df_t = df.loc[tors_rows].copy(); df_t['match_id'] = match_ids
    df_c = df.loc[ctrl_rows].copy(); df_c['match_id'] = match_ids
    matched = pd.concat([df_t, df_c]).sort_values(
        ['match_id', 'treatment'], ascending=[True, False]).reset_index(drop=True)

    # Balance
    tors_m   = matched[matched['treatment'] == 1]
    ctrl_m   = matched[matched['treatment'] == 0]
    tors_all = df[df['treatment'] == 1]
    ctrl_all = df[df['treatment'] == 0]

    balance_rows = []
    for var in CONTINUOUS_VARS + BINARY_VARS:
        fn = smd_continuous if var in CONTINUOUS_VARS else smd_binary
        balance_rows.append({
            'variable':   var,
            'smd_before': round(abs(fn(tors_all[var], ctrl_all[var])), 3),
            'smd_after':  round(abs(fn(tors_m[var],   ctrl_m[var])),   3),
        })
    balance = pd.DataFrame(balance_rows)
    imb = balance[balance['smd_after'] >= 0.10]
    print(f"\n{len(imb)} variables imbalanced after matching (SMD >= 0.10)")
    if len(imb):
        print(imb[['variable', 'smd_before', 'smd_after']].to_string(index=False))

    # Nodal balance (informational — not in PS model)
    tors_nodal = tors_m['has_nodal_dx'].fillna(0).astype(int).mean()
    ctrl_nodal = ctrl_m['has_nodal_dx'].fillna(0).astype(int).mean()
    nodal_smd  = abs(smd_binary(tors_m['has_nodal_dx'].fillna(0).astype(int),
                                ctrl_m['has_nodal_dx'].fillna(0).astype(int)))
    print(f"\nhas_nodal_dx (informational, not in PS model):")
    print(f"  {tors_label}: {100*tors_nodal:.1f}%   {ctrl_label}: {100*ctrl_nodal:.1f}%   SMD={nodal_smd:.3f}")

    return matched


def survival_summary(matched, tors_label, ctrl_label, con):
    """Join to opscc_survival and report KM + Cox."""
    dsysrtky_list = matched['DSYSRTKY'].tolist()
    placeholders  = ','.join([f"'{d}'" for d in dsysrtky_list])

    surv = con.execute(f"""
        SELECT s.DSYSRTKY, s.tx_group, s.first_tx_date,
               s.age_at_dx, s.van_walraven_score, s.event, s.t_days
        FROM opscc_survival s
        WHERE s.DSYSRTKY IN ({placeholders})
          AND s.t_days >= 0
    """).df()

    surv = surv.merge(
        matched[['DSYSRTKY', 'match_id', 'treatment']],
        on='DSYSRTKY', how='inner'
    )
    surv['t_years'] = surv['t_days'] / 365.25

    tors_s = surv[surv['treatment'] == 1]
    ctrl_s = surv[surv['treatment'] == 0]

    print(f"\n-- Overall Survival (KM) --------------------------------------")
    print(f"  {tors_label}: N={len(tors_s):,}  Deaths={tors_s['event'].sum()}"
          f"  ({100*tors_s['event'].mean():.1f}%)")
    print(f"  {ctrl_label}: N={len(ctrl_s):,}  Deaths={ctrl_s['event'].sum()}"
          f"  ({100*ctrl_s['event'].mean():.1f}%)")

    kmf_t = KaplanMeierFitter().fit(tors_s['t_years'], tors_s['event'])
    kmf_c = KaplanMeierFitter().fit(ctrl_s['t_years'], ctrl_s['event'])

    print(f"\n  {'Timepoint':<10} {tors_label:>12}  {ctrl_label:>10}")
    print(f"  {'-'*40}")
    for label, yrs in [('1-year', 1), ('3-year', 3), ('5-year', 5)]:
        t_val = kmf_t.survival_function_at_times([yrs]).values[0]
        c_val = kmf_c.survival_function_at_times([yrs]).values[0]
        print(f"  {label:<10} {100*t_val:>11.1f}%  {100*c_val:>9.1f}%")

    lr = logrank_test(tors_s['t_years'], ctrl_s['t_years'],
                      tors_s['event'],   ctrl_s['event'])
    print(f"\n  Log-rank p = {lr.p_value:.4f}")

    try:
        surv['tors'] = surv['treatment']
        cox = CoxPHFitter()
        cox.fit(surv[['t_days', 'event', 'tors', 'age_at_dx', 'van_walraven_score']],
                duration_col='t_days', event_col='event')
        row = cox.summary.loc['tors']
        print(f"  Cox HR ({tors_label} vs {ctrl_label}): "
              f"{row['exp(coef)']:.3f}  "
              f"95% CI ({row['exp(coef) lower 95%']:.3f}–{row['exp(coef) upper 95%']:.3f})  "
              f"p={row['p']:.4f}")
    except Exception as e:
        print(f"  Cox failed: {e}")


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading opscc_propensity...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

df_all = con.execute("SELECT * FROM opscc_propensity").df()
print(f"  Loaded {len(df_all):,} patients")

df_b_base = df_all[df_all['tx_group'].isin(['TORS + RT', 'CT/CRT'])].copy()

# ── Version 1: TORS+RT all  vs  CT/CRT N0 only ────────────────────────────────
print("\n" + "#"*70)
print("  VERSION 1: TORS+RT (all)  vs  CT/CRT (N0 only, has_nodal_dx=FALSE)")
print("#"*70)

df_v1 = df_b_base[
    (df_b_base['tx_group'] == 'TORS + RT') |
    ((df_b_base['tx_group'] == 'CT/CRT') & (df_b_base['has_nodal_dx'] == False))
].copy()

matched_v1 = run_psm(df_v1, "VERSION 1")
survival_summary(matched_v1, 'TORS + RT', 'CT/CRT (N0)', con)

# ── Version 2: TORS+RT N0 only  vs  CT/CRT N0 only ───────────────────────────
print("\n" + "#"*70)
print("  VERSION 2: TORS+RT (N0 only)  vs  CT/CRT (N0 only) — symmetric")
print("#"*70)

df_v2 = df_b_base[
    ((df_b_base['tx_group'] == 'TORS + RT') & (df_b_base['has_nodal_dx'] == False)) |
    ((df_b_base['tx_group'] == 'CT/CRT')    & (df_b_base['has_nodal_dx'] == False))
].copy()

matched_v2 = run_psm(df_v2, "VERSION 2")
survival_summary(matched_v2, 'TORS + RT', 'CT/CRT (N0)', con)

# ── Reference: main analysis results (from pipeline run) ──────────────────────
print("\n" + "#"*70)
print("  REFERENCE: Main analysis (TORS+RT all vs CT/CRT all, has_nodal_dx in PS model)")
print("#"*70)
print("  N: TORS+RT=371  CT/CRT=371  (matched)")
print("  PS C-stat: 0.699")
print("  Imbalanced vars: 3 (dx_year, chf, rf)")
print("  has_nodal_dx: TORS+RT 70.4% vs CT/CRT 47.2% pre-match")
print("  5-yr OS: TORS+RT 83.6%  CT/CRT 65.5%")
print("  Cox HR: 0.395 (0.285–0.547)  p<0.0001")

con.close()
print("\nDone.")
