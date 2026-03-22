"""
stroke_psm.py

Propensity score matching for the stroke + SLP timing study.

Two pairwise PSM comparisons (each 1:1 greedy nearest-neighbor, caliper = 0.2 * SD(logit PS)):
  Comparison A: 0-14d  vs 31-90d  → psm_matched_A / psm_match_id_A / prop_score_A
  Comparison B: 15-30d vs 31-90d  → psm_matched_B / psm_match_id_B / prop_score_B

Covariates in PS model:
  age_at_adm, index_los, van_walraven_score, adm_year (continuous)
  sex, race, stroke_type, adm_source (categorical)
  dysphagia_poa, aspiration_poa, mech_vent, peg_placed, trach_placed,
  prior_stroke, afib, hypertension (binary)
  drg_group (bucketed DRG)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

DB_PATH = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
RANDOM_SEED = 42

# ── Covariate lists ────────────────────────────────────────────────────────────

CONT_VARS   = ['age_at_adm', 'index_los', 'van_walraven_score', 'adm_year']
BINARY_VARS = ['dysphagia_poa', 'aspiration_poa', 'afib', 'hypertension',
               'mech_vent', 'peg_placed', 'trach_placed', 'prior_stroke']
CAT_VARS    = ['sex', 'race', 'stroke_type', 'drg_group', 'adm_source']

COMPARISONS = [
    ('A', '0-14d',  '31-90d'),   # treated='0-14d',  control='31-90d'
    ('B', '15-30d', '31-90d'),   # treated='15-30d', control='31-90d'
]


def bucket_drg(drg_cd):
    """Map raw DRG code to a clinical group for use in PS model."""
    if pd.isna(drg_cd):
        return 'Other'
    d = str(drg_cd).strip()
    try:
        n = int(d)
    except ValueError:
        return 'Other'
    if 61 <= n <= 69:   return 'Medical_stroke'
    if 20 <= n <= 38:   return 'Neurosurgical'
    if 52 <= n <= 60:   return 'Spinal'
    if 70 <= n <= 74:   return 'TIA_headache'
    return 'Other'


def smd(x, treated):
    """Standardized mean difference."""
    x1 = x[treated == 1]
    x0 = x[treated == 0]
    mu1, mu0 = x1.mean(), x0.mean()
    pooled_sd = np.sqrt((x1.std()**2 + x0.std()**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(mu1 - mu0) / pooled_sd


def print_balance(df, treat_flag_col, label=""):
    cols = CONT_VARS + BINARY_VARS
    print(f"\n  Balance check {label}")
    print(f"  {'Variable':<28}  {'SMD':>6}")
    print(f"  {'-'*36}")
    for col in cols:
        if col in df.columns:
            s = smd(df[col].fillna(0).values, df[treat_flag_col].values)
            flag = '  ***' if s > 0.1 else ''
            print(f"  {col:<28}  {s:>6.3f}{flag}")


def run_comparison(df_all, label, treat_grp, ctrl_grp):
    """
    Fit PS model and run 1:1 greedy matching for one pairwise comparison.

    Returns a DataFrame with columns: DSYSRTKY, psm_matched_X, psm_match_id_X, prop_score_X
    where X is `label` (e.g. 'A' or 'B').
    """
    matched_col  = f'psm_matched_{label}'
    match_id_col = f'psm_match_id_{label}'
    ps_col       = f'prop_score_{label}'

    df = df_all[df_all['slp_timing_group'].isin([treat_grp, ctrl_grp])].copy()
    df['_treated'] = (df['slp_timing_group'] == treat_grp).astype(int)
    n_treat = df['_treated'].sum()
    n_ctrl  = (df['_treated'] == 0).sum()
    print(f"\n-- Comparison {label}: {treat_grp!r} vs {ctrl_grp!r}  "
          f"(n={len(df):,}: treated={n_treat:,}, control={n_ctrl:,})")

    # Feature matrix
    dummies = pd.get_dummies(df[CAT_VARS], drop_first=True)
    X = pd.concat([df[CONT_VARS + BINARY_VARS].astype(float), dummies], axis=1)
    y = df['_treated']

    print_balance(df, '_treated', "(pre-match)")

    # ── Propensity score ──────────────────────────────────────────────────────
    print(f"\n  Fitting logistic regression ({label}) ...")
    t0 = time.time()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, n_jobs=-1)
    lr.fit(X_scaled, y)
    print(f"  Done in {time.time()-t0:.1f}s  ({lr.n_iter_[0]} iterations)")

    ps       = np.clip(lr.predict_proba(X_scaled)[:, 1], 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps / (1 - ps))
    df = df.reset_index(drop=True)
    df['ps']       = ps
    df['logit_ps'] = logit_ps

    print(f"  PS range: [{ps.min():.4f}, {ps.max():.4f}]  "
          f"mean treated={ps[y.values==1].mean():.3f}  "
          f"mean control={ps[y.values==0].mean():.3f}")

    # ── Greedy 1:1 nearest-neighbor matching (caliper on logit PS) ───────────────
    # All patients are home-discharged; no exact matching on discharge group needed.
    caliper = 0.2 * logit_ps.std()
    print(f"\n  Matching {label} (caliper = {caliper:.4f}) ...")
    t0 = time.time()

    treated = df[df['_treated'] == 1].reset_index(drop=True)
    control = df[df['_treated'] == 0].reset_index(drop=True)

    ctrl_lps = control['logit_ps'].values.reshape(-1, 1)
    ctrl_ids = control['DSYSRTKY'].values
    tree     = cKDTree(ctrl_lps)

    rng   = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(treated))

    matched_pairs = []
    used_ctrl     = set()

    for i in order:
        t_lps = treated.loc[i, 'logit_ps']
        t_id  = treated.loc[i, 'DSYSRTKY']

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

    n_matched = len(matched_pairs)
    print(f"  Matched {n_matched:,} pairs of {n_treat:,} treated  "
          f"({100*n_matched/n_treat:.1f}% retention)  in {time.time()-t0:.1f}s")

    # Post-match balance
    matched_ids = {pid for pair in matched_pairs for pid in pair}
    df_match    = df[df['DSYSRTKY'].isin(matched_ids)].copy()
    print_balance(df_match, '_treated', f"(post-match {label})")

    # ── Build result DataFrames ───────────────────────────────────────────────
    # Match flags
    records = []
    for t_id, c_id in matched_pairs:
        records.append({'DSYSRTKY': t_id, matched_col: True, match_id_col: c_id})
        records.append({'DSYSRTKY': c_id, matched_col: True, match_id_col: t_id})
    match_df = pd.DataFrame(records)

    # Propensity scores (all rows in this comparison)
    ps_df = df[['DSYSRTKY', 'ps']].rename(columns={'ps': ps_col})

    return match_df, ps_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {DB_PATH} ...")
    con = duckdb.connect(str(DB_PATH), read_only=False)
    con.execute("SET memory_limit='24GB'; SET threads=12;")

    print("Loading stroke_propensity ...")
    df = con.execute("""
        SELECT
            DSYSRTKY,
            slp_timing_group,
            dschg_group,
            age_at_adm,
            COALESCE(index_los, 0)  AS index_los,
            van_walraven_score,
            adm_year,
            sex,
            race,
            stroke_type,
            DRG_CD                  AS drg_cd,
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

    for col in CONT_VARS:
        df[col] = df[col].fillna(df[col].median())
    df['drg_group'] = df['drg_cd'].apply(bucket_drg)
    df['adm_source'] = df['adm_source'].fillna('Unknown').astype(str)
    df['adm_year']   = df['adm_year'].fillna(df['adm_year'].median()).astype(int)

    counts = df['slp_timing_group'].value_counts()
    print(f"  Loaded {len(df):,} rows  |  " +
          "  ".join(f"{g}:{counts.get(g,0):,}" for g in ['0-14d','15-30d','31-90d','No SLP']))

    # ── Run both comparisons ──────────────────────────────────────────────────
    print("\nWriting match results to stroke_propensity ...")
    t_write = time.time()

    for label, treat_grp, ctrl_grp in COMPARISONS:
        match_df, ps_df = run_comparison(df, label, treat_grp, ctrl_grp)

        matched_col  = f'psm_matched_{label}'
        match_id_col = f'psm_match_id_{label}'
        ps_col       = f'prop_score_{label}'

        # Write propensity scores
        con.register(f'_ps_{label}', ps_df)
        con.execute(f"""
            UPDATE stroke_propensity
            SET {ps_col} = r.{ps_col}
            FROM _ps_{label} r
            WHERE stroke_propensity.DSYSRTKY = r.DSYSRTKY
        """)
        con.unregister(f'_ps_{label}')

        # Write match flags
        con.register(f'_match_{label}', match_df)
        con.execute(f"""
            UPDATE stroke_propensity
            SET {matched_col}  = r.{matched_col},
                {match_id_col} = r.{match_id_col}
            FROM _match_{label} r
            WHERE stroke_propensity.DSYSRTKY = r.DSYSRTKY
        """)
        con.unregister(f'_match_{label}')

    print(f"\n  Writes done in {time.time()-t_write:.1f}s")

    # ── Verification ──────────────────────────────────────────────────────────
    for label, treat_grp, ctrl_grp in COMPARISONS:
        matched_col = f'psm_matched_{label}'
        n = con.execute(
            f"SELECT COUNT(*) FROM stroke_propensity WHERE {matched_col} = TRUE"
        ).fetchone()[0]
        print(f"\n  Comparison {label} ({treat_grp} vs {ctrl_grp}): "
              f"{n:,} matched rows  ({n//2:,} pairs)")

        summary = con.execute(f"""
            SELECT
                slp_timing_group,
                COUNT(*) AS n,
                ROUND(AVG(age_at_adm),1)          AS mean_age,
                ROUND(100.0*SUM(CASE WHEN sex='Male' THEN 1 ELSE 0 END)/COUNT(*),1) AS pct_male,
                ROUND(AVG(van_walraven_score),2)   AS mean_vw,
                ROUND(100.0*SUM(dysphagia_poa)/COUNT(*),1) AS pct_dysphagia,
                ROUND(AVG(index_los),1)            AS mean_los
            FROM stroke_propensity
            WHERE {matched_col} = TRUE
            GROUP BY slp_timing_group
            ORDER BY slp_timing_group
        """).df()
        print(f"  Matched summary ({label}):")
        print(summary.to_string(index=False))

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
