"""
Propensity Score Matching — two independent comparisons for OPSCC:
  Comparison A: TORS alone vs RT alone
  Comparison B: TORS + RT  vs CT/CRT

Each comparison uses 1:1 nearest-neighbor matching on logit(PS),
caliper = 0.2 × SD(logit PS).

Outputs:
  opscc_psm_A.parquet   — matched pairs for Comparison A
  opscc_psm_B.parquet   — matched pairs for Comparison B
  opscc_propensity columns added/updated:
    psm_matched_A BOOLEAN, psm_match_id_A INTEGER
    psm_matched_B BOOLEAN, psm_match_id_B INTEGER
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DB_PATH = r"F:\CMS\cms_data.duckdb"

FEATURE_COLS = [
    'age_at_dx', 'male', 'white', 'black', 'hispanic', 'asian_pi',
    'van_walraven_score', 'coag', 'chf', 'cpd', 'rf',
    'dx_year',
    'subsite_c01', 'subsite_c09', 'subsite_c10',
    'region_south', 'region_midwest', 'region_west',
]

CONTINUOUS_VARS = ['age_at_dx', 'van_walraven_score', 'dx_year']
BINARY_VARS     = [
    'male', 'white', 'black', 'hispanic', 'asian_pi',
    'coag', 'chf', 'cpd', 'rf',
    'subsite_c01', 'subsite_c09', 'subsite_c10',
    'region_south', 'region_midwest', 'region_west',
]


# ── SMD helpers ───────────────────────────────────────────────────────────────
def smd_continuous(a, b):
    diff   = a.mean() - b.mean()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return diff / pooled if pooled > 0 else 0.0

def smd_binary(a, b):
    p1, p2 = a.mean(), b.mean()
    denom  = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else 0.0


# ── Core matching function ────────────────────────────────────────────────────
def run_psm(df_raw, label, tors_label, ctrl_label, out_parquet, col_suffix):
    """
    Fit logistic PS model and run 1:1 NN matching for one comparison.

    Parameters
    ----------
    df_raw      : full opscc_propensity dataframe (already filtered to non-Other)
    label       : human-readable comparison label for printing
    tors_label  : tx_group value for the TORS/surgical arm (treatment = 1)
    ctrl_label  : tx_group value for the control arm       (treatment = 0)
    out_parquet : output file path for the matched dataset
    col_suffix  : 'A' or 'B' — suffix for psm_matched_X / psm_match_id_X columns
    """
    df = df_raw[df_raw['tx_group'].isin([tors_label, ctrl_label])].copy()
    df['treatment'] = (df['tx_group'] == tors_label).astype(int)

    n_tors  = (df['treatment'] == 1).sum()
    n_ctrl  = (df['treatment'] == 0).sum()
    print(f"\n{'='*70}")
    print(f"  COMPARISON {col_suffix}: {tors_label}  vs  {ctrl_label}")
    print(f"  N: {tors_label}={n_tors:,}   {ctrl_label}={n_ctrl:,}")
    print(f"{'='*70}")

    # ── Features ──────────────────────────────────────────────────────────────
    df['male']     = (df['sex']  == 'Male').astype(int)
    df['white']    = (df['race'] == 'White').astype(int)
    df['black']    = (df['race'] == 'Black').astype(int)
    df['hispanic'] = (df['race'] == 'Hispanic').astype(int)
    df['asian_pi'] = (df['race'] == 'Asian/PI').astype(int)

    df['van_walraven_score'] = df['van_walraven_score'].fillna(0)
    for flag in ['coag', 'chf', 'cpd', 'rf']:
        df[flag] = df[flag].fillna(0).astype(int)

    df['dx_year'] = df['dx_year'].fillna(df['dx_year'].median()).astype(int)

    # Subsite dummies (reference = C14)
    df['subsite_c01'] = (df['subsite'] == 'C01').astype(int)
    df['subsite_c09'] = (df['subsite'] == 'C09').astype(int)
    df['subsite_c10'] = (df['subsite'] == 'C10').astype(int)

    # Census region dummies (reference = Northeast)
    df['region_south']   = (df['census_region'] == 'South').astype(int)
    df['region_midwest'] = (df['census_region'] == 'Midwest').astype(int)
    df['region_west']    = (df['census_region'] == 'West').astype(int)

    X = df[FEATURE_COLS].values
    y = df['treatment'].values

    # ── Propensity model ──────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ps_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    ps_model.fit(X_scaled, y)

    df['ps']       = ps_model.predict_proba(X_scaled)[:, 1]
    df['logit_ps'] = np.log(df['ps'] / (1 - df['ps']))

    auc = roc_auc_score(y, df['ps'])
    print(f"\nPS model C-statistic (AUC): {auc:.3f}")

    # ── 1:1 Nearest-neighbor matching ─────────────────────────────────────────
    caliper = 0.2 * df['logit_ps'].std()
    print(f"Caliper (0.2 × SD logit PS): {caliper:.4f}")

    df = df.reset_index(drop=True)
    tors_idx = df.index[df['treatment'] == 1].tolist()
    ctrl_idx = df.index[df['treatment'] == 0].tolist()

    rng = np.random.default_rng(42)
    rng.shuffle(tors_idx)

    logit_ctrl      = df.loc[ctrl_idx, 'logit_ps'].values
    ctrl_available  = np.ones(len(ctrl_idx), dtype=bool)
    matched_pairs   = []
    unmatched_tors  = []

    for ti in tors_idx:
        lp_tors = df.loc[ti, 'logit_ps']
        dists   = np.abs(logit_ctrl - lp_tors)
        dists[~ctrl_available] = np.inf
        best = np.argmin(dists)
        if dists[best] <= caliper:
            matched_pairs.append((ti, ctrl_idx[best]))
            ctrl_available[best] = False
        else:
            unmatched_tors.append(ti)

    n_matched = len(matched_pairs)
    print(f"\nMatching results:")
    print(f"  {tors_label} matched:   {n_matched:,} / {n_tors:,}  ({100*n_matched/n_tors:.1f}%)")
    print(f"  {tors_label} unmatched: {len(unmatched_tors):,}")
    print(f"  Matched pairs:  {n_matched:,}  (total N = {2*n_matched:,})")

    tors_rows  = [p[0] for p in matched_pairs]
    ctrl_rows  = [p[1] for p in matched_pairs]
    match_ids  = list(range(n_matched))

    df_t = df.loc[tors_rows].copy(); df_t['match_id'] = match_ids
    df_c = df.loc[ctrl_rows].copy(); df_c['match_id'] = match_ids
    matched = pd.concat([df_t, df_c]).sort_values(
        ['match_id','treatment'], ascending=[True, False]).reset_index(drop=True)

    # ── Balance ───────────────────────────────────────────────────────────────
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
        print(imb[['variable','smd_before','smd_after']].to_string(index=False))

    # ── Save parquet ──────────────────────────────────────────────────────────
    matched.to_parquet(out_parquet, index=False)
    print(f"\nSaved: {out_parquet}")

    return matched[['DSYSRTKY', 'match_id']].rename(columns={'match_id': f'psm_match_id_{col_suffix}'})


# ── 1. Load full propensity table ─────────────────────────────────────────────
print("Loading opscc_propensity...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_all = con.execute("SELECT * FROM opscc_propensity").df()
con.close()
print(f"  Loaded {len(df_all):,} patients")
for grp, n in df_all.groupby('tx_group').size().items():
    print(f"    {grp}: {n:,}")

# ── 2. Run both PSM comparisons ───────────────────────────────────────────────
ids_A = run_psm(
    df_all,
    label       = 'Comparison A',
    tors_label  = 'TORS alone',
    ctrl_label  = 'RT alone',
    out_parquet = r"F:\CMS\projects\opscc\opscc_psm_A.parquet",
    col_suffix  = 'A',
)

ids_B = run_psm(
    df_all,
    label       = 'Comparison B',
    tors_label  = 'TORS + RT',
    ctrl_label  = 'CT/CRT',
    out_parquet = r"F:\CMS\projects\opscc\opscc_psm_B.parquet",
    col_suffix  = 'B',
)

# ── 3. Write PSM flags back to opscc_propensity ───────────────────────────────
print("\nWriting PSM flags to opscc_propensity...")
con_rw = duckdb.connect(DB_PATH)
con_rw.execute("SET memory_limit='24GB'; SET threads=12;")

for suffix in ('A', 'B'):
    con_rw.execute(f"ALTER TABLE opscc_propensity ADD COLUMN IF NOT EXISTS psm_matched_{suffix}  BOOLEAN DEFAULT FALSE;")
    con_rw.execute(f"ALTER TABLE opscc_propensity ADD COLUMN IF NOT EXISTS psm_match_id_{suffix} INTEGER;")
    con_rw.execute(f"UPDATE opscc_propensity SET psm_matched_{suffix} = FALSE, psm_match_id_{suffix} = NULL;")

for ids_df, suffix in [(ids_A, 'A'), (ids_B, 'B')]:
    con_rw.register(f'match_ids_{suffix}', ids_df)
    con_rw.execute(f"""
        UPDATE opscc_propensity
        SET psm_matched_{suffix}  = TRUE,
            psm_match_id_{suffix} = m.psm_match_id_{suffix}
        FROM match_ids_{suffix} m
        WHERE opscc_propensity.DSYSRTKY = m.DSYSRTKY
    """)
    n = con_rw.execute(f"SELECT COUNT(*) FROM opscc_propensity WHERE psm_matched_{suffix} = TRUE").fetchone()[0]
    print(f"  Comparison {suffix}: psm_matched_{suffix} = TRUE for {n:,} patients")

con_rw.close()
print("\nDone.")
