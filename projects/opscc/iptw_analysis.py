"""
Propensity Score Matching: TORS only vs CT/CRT only for non-metastatic OPSCC
1:1 nearest-neighbor matching on logit(PS), caliper = 0.2 * SD(logit PS)
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DB_PATH = r"F:\CMS\cms_data.duckdb"

# ── 1. Load data ──────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)

df = con.execute("""
    SELECT *
    FROM opscc_propensity
    WHERE tx_group IN ('TORS only', 'CT/CRT only')
""").df()

con.close()
n_tors   = (df.tx_group == 'TORS only').sum()
n_ctcrt  = (df.tx_group == 'CT/CRT only').sum()
print(f"Loaded {len(df):,} patients  |  TORS: {n_tors:,}  CT/CRT: {n_ctcrt:,}")

# ── 2. Prepare features ───────────────────────────────────────────────────────
df['treatment'] = (df['tx_group'] == 'TORS only').astype(int)   # 1=TORS, 0=CT/CRT

df['male']     = (df['sex']  == 'Male').astype(int)
df['white']    = (df['race'] == 'White').astype(int)
df['black']    = (df['race'] == 'Black').astype(int)
df['hispanic'] = (df['race'] == 'Hispanic').astype(int)
df['asian_pi'] = (df['race'] == 'Asian/PI').astype(int)

elixhauser_flags = [
    'chf','carit','valv','pcd','pvd','hypunc','hypc','para','ond','cpd',
    'diabunc','diabc','hypothy','rf','ld','pud','aids','lymph','metacanc',
    'solidtum','rheumd','coag','obes','wloss','fed','blane','dane',
    'alcohol','drug','psycho','depre'
]

df[elixhauser_flags] = df[elixhauser_flags].fillna(0).astype(int)
df['van_walraven_score'] = df['van_walraven_score'].fillna(0)

feature_cols = ['age_at_dx', 'male', 'white', 'black', 'hispanic', 'asian_pi'] + elixhauser_flags

X = df[feature_cols].values
y = df['treatment'].values

# ── 3. Fit propensity score model ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
ps_model.fit(X_scaled, y)

df['ps']       = ps_model.predict_proba(X_scaled)[:, 1]
df['logit_ps'] = np.log(df['ps'] / (1 - df['ps']))

auc = roc_auc_score(y, df['ps'])
print(f"\nPropensity model C-statistic (AUC): {auc:.3f}")

# ── 4. 1:1 Nearest-neighbor matching (without replacement) ───────────────────
# Caliper = 0.2 * SD(logit PS) — standard Cochran & Rubin rule
caliper = 0.2 * df['logit_ps'].std()
print(f"Caliper (0.2 x SD logit PS): {caliper:.4f}")

# Reset index so positional lookup is clean
df = df.reset_index(drop=True)

tors_idx  = df.index[df['treatment'] == 1].tolist()
ctcrt_idx = df.index[df['treatment'] == 0].tolist()

# Shuffle TORS to avoid systematic ordering bias
rng = np.random.default_rng(42)
rng.shuffle(tors_idx)

logit_ctcrt = df.loc[ctcrt_idx, 'logit_ps'].values
ctcrt_available = np.ones(len(ctcrt_idx), dtype=bool)   # track unmatched controls

matched_pairs   = []   # [(tors_idx, ctcrt_idx), ...]
unmatched_tors  = []

for ti in tors_idx:
    lp_tors = df.loc[ti, 'logit_ps']
    dists   = np.abs(logit_ctcrt - lp_tors)

    # Mask already-matched controls
    dists[~ctcrt_available] = np.inf
    best = np.argmin(dists)

    if dists[best] <= caliper:
        matched_pairs.append((ti, ctcrt_idx[best]))
        ctcrt_available[best] = False
    else:
        unmatched_tors.append(ti)

n_matched = len(matched_pairs)
print(f"\nMatching results:")
print(f"  TORS matched:   {n_matched:,} / {n_tors:,}  ({100*n_matched/n_tors:.1f}%)")
print(f"  TORS unmatched: {len(unmatched_tors):,}  (outside caliper)")
print(f"  Matched pairs:  {n_matched:,}  (total N = {2*n_matched:,})")

# Build matched dataset
tors_rows  = [p[0] for p in matched_pairs]
ctcrt_rows = [p[1] for p in matched_pairs]
match_id   = list(range(n_matched))

df_tors  = df.loc[tors_rows].copy();  df_tors['match_id']  = match_id
df_ctcrt = df.loc[ctcrt_rows].copy(); df_ctcrt['match_id'] = match_id
matched  = pd.concat([df_tors, df_ctcrt]).sort_values(['match_id','treatment'], ascending=[True,False])
matched  = matched.reset_index(drop=True)

# ── 5. Covariate balance ──────────────────────────────────────────────────────
continuous_vars = ['age_at_dx', 'van_walraven_score']
binary_vars     = ['male', 'white', 'black', 'hispanic', 'asian_pi'] + elixhauser_flags

def smd_continuous(a, b):
    diff = a.mean() - b.mean()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return diff / pooled if pooled > 0 else 0.0

def smd_binary(a, b):
    p1, p2 = a.mean(), b.mean()
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else 0.0

tors_m  = matched[matched['treatment'] == 1]
ctcrt_m = matched[matched['treatment'] == 0]

# Unweighted (pre-match) pools
tors_all  = df[df['treatment'] == 1]
ctcrt_all = df[df['treatment'] == 0]

balance_rows = []
for var in continuous_vars + binary_vars:
    fn = smd_continuous if var in continuous_vars else smd_binary
    smd_before = abs(fn(tors_all[var],  ctcrt_all[var]))
    smd_after  = abs(fn(tors_m[var],    ctcrt_m[var]))
    balance_rows.append({
        'variable':   var,
        'tors_pre':   tors_all[var].mean(),
        'ctcrt_pre':  ctcrt_all[var].mean(),
        'tors_post':  tors_m[var].mean(),
        'ctcrt_post': ctcrt_m[var].mean(),
        'smd_before': round(smd_before, 3),
        'smd_after':  round(smd_after,  3),
        'balanced':   smd_after < 0.10
    })

balance = pd.DataFrame(balance_rows)

print("\n-- Covariate Balance (SMD < 0.10 = well-balanced) ----------------------")
print(f"{'Variable':<20} {'TORS pre':>9} {'CTRL pre':>9} {'TORS post':>10} {'CTRL post':>10} {'SMD pre':>8} {'SMD post':>9} {'OK':>4}")
print("-" * 84)
for _, row in balance.iterrows():
    flag = "Y" if row['balanced'] else "X"
    print(f"{row['variable']:<20} {row['tors_pre']:>9.3f} {row['ctcrt_pre']:>9.3f} "
          f"{row['tors_post']:>10.3f} {row['ctcrt_post']:>10.3f} "
          f"{row['smd_before']:>8.3f} {row['smd_after']:>9.3f} {flag:>4}")

imbalanced = balance[~balance['balanced']]
print(f"\n{len(imbalanced)} variables still imbalanced after matching (SMD >= 0.10)")
if len(imbalanced):
    print(imbalanced[['variable','smd_before','smd_after']].to_string(index=False))

# ── 6. PS distribution in matched sample ─────────────────────────────────────
print("\n-- Propensity Score Distribution (matched sample) -----------------------")
ps_summary = matched.groupby('tx_group')['ps'].describe().round(3)
print(ps_summary.to_string())

print("\n-- Logit PS: mean difference within matched pairs -----------------------")
pair_diff = (matched[matched['treatment']==1]['logit_ps'].values
           - matched[matched['treatment']==0]['logit_ps'].values)
print(f"Mean |logit PS diff| within pairs: {np.abs(pair_diff).mean():.4f}")
print(f"Max  |logit PS diff| within pairs: {np.abs(pair_diff).max():.4f}")

# ── 7. Save matched dataset & write psm_matched back to DB ───────────────────
matched.to_parquet(r"F:\CMS\opscc_psm_matched.parquet", index=False)
print(f"\nMatched dataset saved to F:\\CMS\\opscc_psm_matched.parquet")
print(f"Columns include: DSYSRTKY, match_id, treatment, ps, logit_ps")
print(f"Final matched cohort: {len(matched):,} patients ({n_matched:,} TORS + {n_matched:,} CT/CRT)")

# Write psm_matched and psm_match_id back to opscc_propensity
print("\nWriting psm_matched flags to opscc_propensity...")
con_rw = duckdb.connect(DB_PATH)
con_rw.execute("SET memory_limit='24GB'; SET threads=12;")
con_rw.execute("ALTER TABLE opscc_propensity ADD COLUMN IF NOT EXISTS psm_matched  BOOLEAN DEFAULT FALSE;")
con_rw.execute("ALTER TABLE opscc_propensity ADD COLUMN IF NOT EXISTS psm_match_id INTEGER;")
con_rw.execute("UPDATE opscc_propensity SET psm_matched = FALSE, psm_match_id = NULL;")
df_ids = matched[['DSYSRTKY', 'match_id']].rename(columns={'match_id': 'psm_match_id'})
con_rw.register('match_ids', df_ids)
con_rw.execute("""
    UPDATE opscc_propensity
    SET psm_matched  = TRUE,
        psm_match_id = m.psm_match_id
    FROM match_ids m
    WHERE opscc_propensity.DSYSRTKY = m.DSYSRTKY
""")
n_written = con_rw.execute("SELECT COUNT(*) FROM opscc_propensity WHERE psm_matched = TRUE").fetchone()[0]
con_rw.close()
print(f"  psm_matched = TRUE for {n_written:,} patients")
