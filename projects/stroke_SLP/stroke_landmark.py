"""
stroke_landmark.py

30-day landmark analysis of outpatient SLP timing after acute stroke.

Research question:
  Among stroke survivors who received outpatient SLP within 90 days of
  discharge, is earlier initiation associated with fewer dysphagia-related
  complications?

Analytic cohort:
  - Acute stroke (stroke_cohort)
  - Alive and FFS-enrolled at day 30 post-discharge (landmark)
  - Received outpatient SLP (carrier/outpatient facility, NOT SNF/HHA) within 90 days

SLP timing groups (days from index discharge):
  0-14d  (reference)  — early outpatient / IRF
  15-30d              — standard outpatient referral
  31-90d              — delayed outpatient

Primary outcomes (from day 30 landmark):
  1. PEG/gastrostomy tube placement
     (exclude patients with peg_placed=1 from this outcome — reverse causation)
  2. Inpatient aspiration pneumonia

Secondary outcomes:
  All-cause mortality, recurrent stroke, dysphagia diagnosis
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH  = r"F:\CMS\cms_data.duckdb"
LANDMARK = 30   # days post-discharge
MAX_FOLLOW = 1825  # 5-year max follow-up from discharge

# ── 1. Load cohort ─────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")

df = con.execute("""
    SELECT
        p.DSYSRTKY,
        p.age_at_adm,
        p.sex,
        p.stroke_type,
        p.index_los,
        p.mech_vent,
        p.peg_placed,
        p.trach_placed,
        p.dschg_status,
        p.dysphagia_poa,
        p.aspiration_poa,
        p.prior_stroke,
        p.dementia,
        p.van_walraven_score,
        p.prop_score,

        -- Outpatient SLP timing
        p.slp_outpt_any_90d,
        p.days_to_slp_outpt,
        p.slp_outpt_0_14d,
        p.slp_outpt_15_30d,
        p.slp_outpt_31_90d,

        -- Outcomes
        o.days_to_death,
        o.days_to_aspiration,
        o.has_aspiration,
        o.days_to_gtube,
        o.has_gtube,
        o.days_to_dysphagia,
        o.has_dysphagia,
        o.days_to_recur_stroke,
        o.days_to_readmit

    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()

print(f"Full cohort: {len(df):,}")

# ── 2. Landmark restriction ────────────────────────────────────────────────────
# Alive at day 30 post-discharge
lm = df[
    df['days_to_death'].isna() | (df['days_to_death'] > LANDMARK)
].copy()
print(f"Alive at day-30 landmark: {len(lm):,}  "
      f"(excluded {len(df)-len(lm):,} who died <=30d post-discharge)")

# ── 3. Restrict to outpatient SLP recipients within 90 days ───────────────────
lm_slp = lm[
    lm['slp_outpt_any_90d'] == 1
].copy()
print(f"Received outpatient SLP within 90d: {len(lm_slp):,}  "
      f"({100*len(lm_slp)/len(lm):.1f}% of landmark cohort)")

# ── 4. Classify SLP timing groups ──────────────────────────────────────────────
def slp_group(row):
    d = row['days_to_slp_outpt']
    if pd.isna(d):
        return None
    elif d <= 14:
        return 'SLP 0-14d'
    elif d <= 30:
        return 'SLP 15-30d'
    else:
        return 'SLP 31-90d'

lm_slp['timing_group'] = lm_slp.apply(slp_group, axis=1)

# Group counts
print(f"\nSLP timing group distribution (landmark cohort):")
for grp in ['SLP 0-14d', 'SLP 15-30d', 'SLP 31-90d']:
    n = (lm_slp['timing_group'] == grp).sum()
    print(f"  {grp}: {n:,}  ({100*n/len(lm_slp):.1f}%)")

# ── 5. Shift outcome times to landmark ─────────────────────────────────────────
# Only count events that occur AFTER day 30
# Cast to float first (DuckDB returns nullable Int64 which rejects float clip)
for col in ['days_to_death', 'days_to_aspiration', 'days_to_gtube',
            'days_to_dysphagia', 'days_to_recur_stroke', 'days_to_readmit']:
    lm_slp[col] = lm_slp[col].astype('float64')
    lm_slp[f'{col}_lm'] = np.where(
        lm_slp[col].notna() & (lm_slp[col] > LANDMARK),
        (lm_slp[col] - LANDMARK).clip(lower=0.5),
        np.nan
    )

# Censor at 5 years from landmark (MAX_FOLLOW - 30)
MAX_FOLLOW_LM = MAX_FOLLOW - LANDMARK
lm_slp['censor_lm'] = np.where(
    lm_slp['days_to_death_lm'].notna(),
    lm_slp['days_to_death_lm'].clip(upper=MAX_FOLLOW_LM),
    MAX_FOLLOW_LM
)

# Event indicators (occurred after landmark)
lm_slp['died_lm']       = (lm_slp['days_to_death_lm'].notna()).astype(int)
lm_slp['asp_lm']        = (lm_slp['days_to_aspiration_lm'].notna()).astype(int)
lm_slp['gtube_lm']      = (lm_slp['days_to_gtube_lm'].notna()).astype(int)
lm_slp['dysphagia_lm']  = (lm_slp['days_to_dysphagia_lm'].notna()).astype(int)
lm_slp['recur_lm']      = (lm_slp['days_to_recur_stroke_lm'].notna()).astype(int)

# Fill missing duration with censor time
for ev, dur in [('asp_lm',       'days_to_aspiration_lm'),
                ('gtube_lm',     'days_to_gtube_lm'),
                ('dysphagia_lm', 'days_to_dysphagia_lm'),
                ('recur_lm',     'days_to_recur_stroke_lm')]:
    lm_slp[f'{dur}_filled'] = lm_slp[dur].fillna(lm_slp['censor_lm']).clip(lower=0.5)

# ── 6. Covariates ──────────────────────────────────────────────────────────────
# One-hot timing (reference = 0-14d)
lm_slp['slp_15_30d'] = (lm_slp['timing_group'] == 'SLP 15-30d').astype(int)
lm_slp['slp_31_90d'] = (lm_slp['timing_group'] == 'SLP 31-90d').astype(int)

# Discharge destination — map STUS_CD to clinical groups
# 01/08 = home (reference), 06 = hha, 03/61 = snf, 62 = irf, else = other
def map_dschg(code):
    if pd.isna(code):
        return 'home'
    c = str(code).strip()
    if c in ('01', '08'):  return 'home'
    if c == '06':          return 'hha'
    if c in ('03', '61'):  return 'snf'
    if c == '62':          return 'irf'
    return 'other'

lm_slp['dschg_group'] = lm_slp['dschg_status'].apply(map_dschg)
print(f"\nDischarge destination distribution in outpatient SLP cohort:")
print(lm_slp['dschg_group'].value_counts().to_string())

dschg_dummies = pd.get_dummies(lm_slp['dschg_group'], prefix='dschg', drop_first=True)
dschg_dummies = dschg_dummies.loc[:, dschg_dummies.std() > 0.01]
lm_slp = pd.concat([lm_slp, dschg_dummies], axis=1)

# Sex and stroke_type dummies
sex_dummies    = pd.get_dummies(lm_slp['sex'],         prefix='sex',    drop_first=True)
stroke_dummies = pd.get_dummies(lm_slp['stroke_type'], prefix='stroke', drop_first=True)
lm_slp = pd.concat([lm_slp, sex_dummies, stroke_dummies], axis=1)

COX_COVARIATES = (
    ['slp_15_30d', 'slp_31_90d',
     'age_at_adm', 'van_walraven_score', 'index_los',
     'mech_vent', 'trach_placed',
     'prior_stroke', 'dementia',
     'dysphagia_poa', 'aspiration_poa']
    + list(dschg_dummies.columns)
    + list(sex_dummies.columns)
    + list(stroke_dummies.columns)
)

# ── 7. Cox PH — primary and secondary outcomes ────────────────────────────────
print(f"\n{'='*80}")
print("  Cox PH Models — Outcome from 30-Day Landmark")
print(f"  Reference group: SLP 0-14d (earliest)")
print(f"{'='*80}")

cox_models = [
    # (label, event_col, duration_col, exclude_peg_placed, penalizer)
    ('Aspiration PNA (PRIMARY)',  'asp_lm',   'days_to_aspiration_lm_filled',   False, 0.0),
    ('PEG/G-tube (PRIMARY)',      'gtube_lm', 'days_to_gtube_lm_filled',        True,  0.1),
    ('Mortality (secondary)',     'died_lm',  'censor_lm',                      False, 0.0),
    ('Recurrent stroke (sec)',    'recur_lm', 'days_to_recur_stroke_lm_filled', False, 0.0),
    # Dysphagia Dx excluded — severe detection bias (early SLP codes dysphagia
    # pre-landmark; delayed SLP codes it post-landmark, inflating apparent HR ~4x)
]

for label, ev_col, dur_col, excl_peg, penalizer in cox_models:
    print(f"\n  {label}")
    sub = lm_slp.copy()
    if excl_peg:
        sub = sub[sub['peg_placed'] == 0]
        print(f"    (N={len(sub):,} after excluding index-admission PEG patients)")

    cov_use = [c for c in COX_COVARIATES if c != 'peg_placed' or not excl_peg]
    # Include peg_placed in non-PEG models
    if not excl_peg and 'peg_placed' not in cov_use:
        cov_use = ['peg_placed'] + cov_use

    cox_df = sub[[dur_col, ev_col] + cov_use].rename(
        columns={dur_col: 'duration', ev_col: 'event'}
    ).dropna()

    n_ev = int(cox_df['event'].sum())
    print(f"    N={len(cox_df):,}  Events={n_ev:,}  "
          f"({100*n_ev/len(cox_df):.1f}%)")

    if n_ev < 20:
        print("    Too few events — skip")
        continue

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df, duration_col='duration', event_col='event',
                show_progress=False)

        for timing_var in ['slp_15_30d', 'slp_31_90d']:
            if timing_var not in cph.summary.index:
                continue
            row = cph.summary.loc[timing_var]
            hr   = np.exp(row['coef'])
            lo95 = np.exp(row['coef lower 95%'])
            hi95 = np.exp(row['coef upper 95%'])
            p    = row['p']
            ev_label = '15-30d vs 0-14d' if timing_var == 'slp_15_30d' else '31-90d vs 0-14d'
            p_str = '<0.0001' if p < 0.0001 else f'{p:.4f}'
            # E-value
            ev_val = hr + np.sqrt(hr * abs(hr - 1)) if not np.isnan(hr) else float('nan')
            print(f"    {ev_label:<20}  HR={hr:.2f}  [{lo95:.2f}, {hi95:.2f}]  "
                  f"p={p_str}  E-value={ev_val:.2f}")
    except Exception as e:
        print(f"    ERROR: {e}")


# ── 8. KM curves by timing group ──────────────────────────────────────────────
print(f"\n{'='*80}")
print("  Kaplan-Meier — Cumulative Incidence by Timing Group (from Day-30 Landmark)")
print(f"{'='*80}")

km_outcomes = [
    ('Aspiration PNA',  'asp_lm',    'days_to_aspiration_lm_filled'),
    ('PEG/G-tube',      'gtube_lm',  'days_to_gtube_lm_filled'),
    ('Mortality',       'died_lm',   'censor_lm'),
]

TIME_POINTS = [90, 180, 365, 730]

for label, ev_col, dur_col in km_outcomes:
    print(f"\n  {label}  [cumulative incidence % from landmark]")
    sub = lm_slp.copy()
    if ev_col == 'gtube_lm':
        sub = sub[sub['peg_placed'] == 0]

    groups = {
        'SLP 0-14d':  sub[sub['timing_group'] == 'SLP 0-14d'],
        'SLP 15-30d': sub[sub['timing_group'] == 'SLP 15-30d'],
        'SLP 31-90d': sub[sub['timing_group'] == 'SLP 31-90d'],
    }

    kmfs = {}
    for grp, gdf in groups.items():
        kmf = KaplanMeierFitter()
        kmf.fit(gdf[dur_col].dropna(), gdf.loc[gdf[dur_col].notna(), ev_col], label=grp)
        kmfs[grp] = kmf

    # Log-rank test (3-way: compare all pairwise)
    g0 = groups['SLP 0-14d']
    g2 = groups['SLP 31-90d']
    lr = logrank_test(g0[dur_col].dropna(), g2[dur_col].dropna(),
                      event_observed_A=g0.loc[g0[dur_col].notna(), ev_col],
                      event_observed_B=g2.loc[g2[dur_col].notna(), ev_col])
    p_str = '<0.0001' if lr.p_value < 0.0001 else f'{lr.p_value:.4f}'
    print(f"  Log-rank (0-14d vs 31-90d): p={p_str}")

    header = f"  {'Days':<8}  {'0-14d':>16}  {'15-30d':>16}  {'31-90d':>16}"
    print(header)
    print(f"  {'-'*60}")
    for t in TIME_POINTS:
        row_parts = [f"  {t}d{'':<5}"]
        for grp in ['SLP 0-14d', 'SLP 15-30d', 'SLP 31-90d']:
            kmf = kmfs[grp]
            s = float(kmf.predict(t))
            ci = kmf.confidence_interval_survival_function_
            idx = min(ci.index.searchsorted(t, side='right'), len(ci) - 1)
            lo = float(ci.iloc[idx, 0])
            hi = float(ci.iloc[idx, 1])
            pct = f"{100*(1-s):.1f} [{100*(1-hi):.1f},{100*(1-lo):.1f}]"
            row_parts.append(f"  {pct:>16}")
        print(''.join(row_parts))


# ── 9. Overlap trimming sensitivity ───────────────────────────────────────────
print(f"\n{'='*80}")
print("  Overlap Trimming Sensitivity (PS 0.05–0.95)")
print(f"{'='*80}")

if lm_slp['prop_score'].notna().sum() > 0:
    n_before = len(lm_slp)
    lm_trim = lm_slp[
        (lm_slp['prop_score'] >= 0.05) & (lm_slp['prop_score'] <= 0.95)
    ].copy()
    print(f"  Before trimming: {n_before:,}  After: {len(lm_trim):,}  "
          f"(excluded {n_before-len(lm_trim):,})")

    # Re-run mortality Cox on trimmed cohort
    cox_df_trim = lm_trim[['censor_lm', 'died_lm'] + COX_COVARIATES + ['peg_placed']].dropna()
    cox_df_trim = cox_df_trim.rename(columns={'censor_lm': 'duration', 'died_lm': 'event'})
    if len(cox_df_trim) > 100 and cox_df_trim['event'].sum() > 20:
        try:
            cph2 = CoxPHFitter()
            cph2.fit(cox_df_trim, duration_col='duration', event_col='event',
                     show_progress=False)
            for timing_var in ['slp_15_30d', 'slp_31_90d']:
                if timing_var not in cph2.summary.index:
                    continue
                row = cph2.summary.loc[timing_var]
                hr   = np.exp(row['coef'])
                lo95 = np.exp(row['coef lower 95%'])
                hi95 = np.exp(row['coef upper 95%'])
                p    = row['p']
                ev_label = '15-30d vs 0-14d' if timing_var == 'slp_15_30d' else '31-90d vs 0-14d'
                print(f"  Trimmed mortality — {ev_label}: HR={hr:.2f} [{lo95:.2f},{hi95:.2f}] p={p:.4f}")
        except Exception as e:
            print(f"  Trimmed Cox ERROR: {e}")
else:
    print("  prop_score not available — run stroke_psm.py first to enable trimming")

print("\nDone.")
