"""
stroke_psm.py

Propensity score matching for the stroke + SLP study.

Algorithm:
  1. Load stroke_propensity from DuckDB
  2. Fit logistic regression: P(SLP=1 | covariates)
  3. 1:1 greedy nearest-neighbor matching without replacement
     Caliper = 0.2 * SD(logit(PS))
  4. Write psm_matched + psm_match_id back to stroke_propensity
  5. Print covariate balance (pre- and post-match SMDs)

Covariates in PS model:
  age_at_adm, index_los, van_walraven_score, adm_year (continuous)
  sex, race, stroke_type, adm_source (categorical)
  dysphagia_poa, aspiration_poa (binary)
  drg_group (bucketed DRG)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

DB_PATH = r"F:\CMS\cms_data.duckdb"
RANDOM_SEED = 42

# ── Covariate lists ────────────────────────────────────────────────────────────

CONT_VARS   = ['age_at_adm', 'index_los', 'van_walraven_score', 'adm_year']
BINARY_VARS = ['dysphagia_poa', 'aspiration_poa', 'afib', 'hypertension',
               'mech_vent', 'peg_placed', 'trach_placed', 'prior_stroke']
CAT_VARS    = ['sex', 'race', 'stroke_type', 'drg_group', 'adm_source']


def bucket_drg(drg_cd):
    """Map raw DRG code to a clinical group for use in PS model."""
    if pd.isna(drg_cd):
        return 'Other'
    d = str(drg_cd).strip()
    try:
        n = int(d)
    except ValueError:
        return 'Other'
    if 61 <= n <= 69:   return 'Medical_stroke'      # ischemic/hemorrhagic medical
    if 20 <= n <= 38:   return 'Neurosurgical'        # craniotomy / neurosurgery
    if 52 <= n <= 60:   return 'Spinal'               # spinal procedures
    if 70 <= n <= 74:   return 'TIA_headache'         # TIA, headache
    return 'Other'


def smd(x, treated):
    """Standardized mean difference for a binary vector or continuous vector."""
    x1 = x[treated == 1]
    x0 = x[treated == 0]
    mu1, mu0 = x1.mean(), x0.mean()
    s1, s0 = x1.std(), x0.std()
    pooled_sd = np.sqrt((s1**2 + s0**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(mu1 - mu0) / pooled_sd


def print_balance(df, label=""):
    cols = CONT_VARS + BINARY_VARS
    print(f"\n  Balance check {label}")
    print(f"  {'Variable':<28}  {'SMD':>6}")
    print(f"  {'-'*36}")
    for col in cols:
        if col in df.columns:
            s = smd(df[col].fillna(0).values, df['slp_any_30d'].values)
            flag = '  ***' if s > 0.1 else ''
            print(f"  {col:<28}  {s:>6.3f}{flag}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {DB_PATH} ...")
    con = duckdb.connect(DB_PATH, read_only=False)
    con.execute("SET memory_limit='24GB'; SET threads=12;")

    print("Loading stroke_propensity ...")
    df = con.execute("""
        SELECT
            DSYSRTKY,
            slp_any_30d,
            age_at_adm,
            COALESCE(index_los, 0)          AS index_los,
            van_walraven_score,
            adm_year,
            sex,
            race,
            stroke_type,
            DRG_CD AS drg_cd,
            adm_source,
            dysphagia_poa,
            aspiration_poa,
            mech_vent,
            peg_placed,
            trach_placed,
            prior_stroke,
            afib,
            hypertension
        FROM stroke_propensity
    """).df()

    print(f"  Loaded {len(df):,} rows  "
          f"({df['slp_any_30d'].sum():,.0f} SLP  /  "
          f"{(df['slp_any_30d']==0).sum():,} No SLP)")

    # ── Feature engineering ────────────────────────────────────────────────────
    df['drg_group'] = df['drg_cd'].apply(bucket_drg)
    df['adm_source'] = df['adm_source'].fillna('Unknown').astype(str)
    df['adm_year']   = df['adm_year'].fillna(df['adm_year'].median()).astype(int)

    # Fill missing continuous values with median
    for col in CONT_VARS:
        df[col] = df[col].fillna(df[col].median())

    # One-hot encode categoricals (drop_first to avoid multicollinearity)
    dummies = pd.get_dummies(df[CAT_VARS], drop_first=True)
    X = pd.concat([df[CONT_VARS + BINARY_VARS].astype(float), dummies], axis=1)
    y = df['slp_any_30d'].astype(int)

    print_balance(df, "(pre-match)")

    # ── Propensity score estimation ────────────────────────────────────────────
    print("\nFitting logistic regression ...")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, n_jobs=-1)
    lr.fit(X_scaled, y)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"Convergence: {lr.n_iter_[0]} iterations")

    ps = lr.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps / (1 - ps))
    df['ps']       = ps
    df['logit_ps'] = logit_ps

    print(f"  PS range: [{ps.min():.4f}, {ps.max():.4f}]  "
          f"mean treated={ps[y==1].mean():.3f}  mean control={ps[y==0].mean():.3f}")

    # ── Greedy 1:1 nearest-neighbor caliper matching ───────────────────────────
    caliper = 0.2 * logit_ps.std()
    print(f"\nMatching (caliper = {caliper:.4f}) ...")
    t0 = time.time()

    treated = df[df['slp_any_30d'] == 1].reset_index(drop=True)
    control = df[df['slp_any_30d'] == 0].reset_index(drop=True)

    ctrl_lps = control['logit_ps'].values.reshape(-1, 1)
    ctrl_ids = control['DSYSRTKY'].values
    tree = cKDTree(ctrl_lps)

    # Shuffle treated to randomize match order
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(treated))

    matched_pairs = []   # [(treat_id, ctrl_id), ...]
    used_ctrl = set()

    for i in order:
        t_lps = treated.loc[i, 'logit_ps']
        t_id  = treated.loc[i, 'DSYSRTKY']

        # Query up to k_max neighbors; the first unused one within caliper wins
        k_max = min(50, len(ctrl_ids))
        dists, idxs = tree.query([[t_lps]], k=k_max)
        for dist, idx in zip(dists[0], idxs[0]):
            if dist > caliper:
                break
            cid = ctrl_ids[idx]
            if cid not in used_ctrl:
                matched_pairs.append((t_id, cid))
                used_ctrl.add(cid)
                break

    elapsed = time.time() - t0
    n_matched = len(matched_pairs)
    n_treated = len(treated)
    print(f"  Matched {n_matched:,} pairs of {n_treated:,} treated  "
          f"({100*n_matched/n_treated:.1f}% retention)  in {elapsed:.1f}s")

    # ── Post-match balance ──────────────────────────────────────────────────────
    matched_ids = set()
    for t_id, c_id in matched_pairs:
        matched_ids.add(t_id)
        matched_ids.add(c_id)

    df_match = df[df['DSYSRTKY'].isin(matched_ids)].copy()
    print_balance(df_match, "(post-match)")

    # ── Build match results DataFrame ──────────────────────────────────────────
    # Each patient gets their match partner's DSYSRTKY as psm_match_id
    records = []
    for t_id, c_id in matched_pairs:
        records.append({'DSYSRTKY': t_id, 'psm_matched': True,  'psm_match_id': c_id})
        records.append({'DSYSRTKY': c_id, 'psm_matched': True,  'psm_match_id': t_id})
    match_df = pd.DataFrame(records)

    # PS scores for all patients (for overlap trimming)
    ps_df = df[['DSYSRTKY', 'ps']].rename(columns={'ps': 'prop_score'})

    # ── Write back to DuckDB ──────────────────────────────────────────────────
    print("\nWriting match results and propensity scores to stroke_propensity ...")
    t0 = time.time()

    # Write propensity scores for all patients
    con.register('_ps_scores', ps_df)
    con.execute("""
        UPDATE stroke_propensity
        SET prop_score = r.prop_score
        FROM _ps_scores r
        WHERE stroke_propensity.DSYSRTKY = r.DSYSRTKY
    """)
    con.unregister('_ps_scores')

    # Register DataFrame as a view, then UPDATE via JOIN
    con.register('_psm_results', match_df)
    con.execute("""
        UPDATE stroke_propensity
        SET psm_matched  = r.psm_matched,
            psm_match_id = r.psm_match_id
        FROM _psm_results r
        WHERE stroke_propensity.DSYSRTKY = r.DSYSRTKY
    """)
    con.unregister('_psm_results')
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Verification ───────────────────────────────────────────────────────────
    n_matched_db = con.execute(
        "SELECT COUNT(*) FROM stroke_propensity WHERE psm_matched = TRUE"
    ).fetchone()[0]
    print(f"\n  stroke_propensity: {n_matched_db:,} matched rows  "
          f"({n_matched_db//2:,} pairs)")

    balance = con.execute("""
        SELECT
            slp_group,
            COUNT(*) AS n,
            ROUND(AVG(age_at_adm),1) AS mean_age,
            ROUND(100.0*SUM(CASE WHEN sex='Male' THEN 1 ELSE 0 END)/COUNT(*),1) AS pct_male,
            ROUND(AVG(van_walraven_score),2) AS mean_vw,
            ROUND(100.0*SUM(dysphagia_poa)/COUNT(*),1) AS pct_dysphagia_poa,
            ROUND(AVG(index_los),1) AS mean_los
        FROM stroke_propensity
        WHERE psm_matched = TRUE
        GROUP BY slp_group
        ORDER BY slp_group
    """).df()
    print("\n  Matched cohort summary:")
    print(balance.to_string(index=False))

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
