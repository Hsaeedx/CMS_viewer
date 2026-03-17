"""
stroke_analysis.py

Analysis of stroke + SLP outcomes on PSM-matched cohort.

Outputs (printed):
  1. Cohort summary (matched vs unmatched)
  2. OR tables for binary outcomes at 180d / 365d / 1095d / 1825d
     (SLP vs No SLP, overall and stratified by stroke type / age)
  3. Cox PH models for time-to-event outcomes (mortality, recurrent stroke,
     aspiration pneumonia) censored at 1825 days
  4. Cost analysis (mean/median 5-year Medicare costs; Mann-Whitney)

Run after:
  stroke_psm.py  (populates psm_matched / psm_match_id in stroke_propensity)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

DB_PATH = r"F:\CMS\cms_data.duckdb"

# ── 1. Load matched cohort ─────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")

df = con.execute("""
    SELECT
        p.DSYSRTKY,
        p.slp_any_30d       AS slp,
        p.slp_group,
        p.age_at_adm,
        p.sex,
        p.race,
        p.stroke_type,
        p.drg_cd,
        p.adm_year,
        p.index_los,
        p.dysphagia_poa,
        p.aspiration_poa,
        p.mech_vent,
        p.peg_placed,
        p.trach_placed,
        p.van_walraven_score,
        p.afib,
        p.hypertension,

        -- Outcomes
        o.death_date,
        o.days_to_death,
        o.first_readmit_date,
        o.days_to_readmit,
        o.n_readmissions_1825d,
        o.first_recur_stroke_date,
        o.days_to_recur_stroke,
        o.first_aspiration_date,
        o.days_to_aspiration,
        o.has_aspiration,
        o.first_dysphagia_date,
        o.days_to_dysphagia,
        o.has_dysphagia,
        o.first_gtube_date,
        o.days_to_gtube,
        o.has_gtube,
        o.first_snf_date,
        o.days_to_snf,
        o.snf_30d,
        o.hha_90d,
        o.total_pmt_1825d,
        o.snf_pmt_1825d,
        o.hha_pmt_1825d,

        -- Index admission cost
        c.index_pmt,
        c.index_dschg_date

    FROM stroke_propensity p
    JOIN stroke_outcomes   o ON o.DSYSRTKY = p.DSYSRTKY
    JOIN stroke_cohort     c ON c.DSYSRTKY = p.DSYSRTKY
    WHERE p.psm_matched = TRUE
""").df()
con.close()

print(f"PSM-matched cohort: {len(df):,}  |  "
      f"SLP: {df['slp'].sum():,}  No SLP: {(df['slp']==0).sum():,}")

# ── 2. Derived fields ──────────────────────────────────────────────────────────
df['death_date']        = pd.to_datetime(df['death_date'])
df['index_dschg_date']  = pd.to_datetime(df['index_dschg_date'])

MAX_FOLLOW = 1825  # 5 years

# Censor at 1825 days (death or administrative end of follow-up)
df['censor_days'] = np.where(
    df['days_to_death'].notna(),
    df['days_to_death'].clip(upper=MAX_FOLLOW),
    MAX_FOLLOW
)

# Mortality at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'died_{label}'] = (df['days_to_death'].notna() & (df['days_to_death'] <= days)).astype(int)

# Readmission at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'readmit_{label}'] = (df['days_to_readmit'].notna() & (df['days_to_readmit'] <= days)).astype(int)

# Recurrent stroke at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'recur_{label}'] = (df['days_to_recur_stroke'].notna() & (df['days_to_recur_stroke'] <= days)).astype(int)

# Aspiration PNA at each time point (has_aspiration = any in 1825d window)
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'asp_{label}'] = (df['days_to_aspiration'].notna() & (df['days_to_aspiration'] <= days)).astype(int)

# Dysphagia at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'dysphagia_{label}'] = (df['days_to_dysphagia'].notna() & (df['days_to_dysphagia'] <= days)).astype(int)

# G-tube at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'gtube_{label}'] = (df['days_to_gtube'].notna() & (df['days_to_gtube'] <= days)).astype(int)

# Age group
df['age_group'] = pd.cut(df['age_at_adm'],
                          bins=[0, 69, 74, 79, 84, 200],
                          labels=['<70', '70-74', '75-79', '80-84', '85+'])

# Elixhauser tertiles
q1 = df['van_walraven_score'].quantile(1/3)
q2 = df['van_walraven_score'].quantile(2/3)
df['elix_grp'] = pd.cut(df['van_walraven_score'],
                          bins=[-np.inf, q1, q2, np.inf],
                          labels=['Low', 'Mid', 'High'])

print(f"Elix tertiles:  Low <= {q1:.0f}  |  Mid {q1:.0f}-{q2:.0f}  |  High > {q2:.0f}")


# ── 3. OR function ─────────────────────────────────────────────────────────────
def compute_or(sub, event_col, days_col, cutoff):
    """
    OR for SLP vs No SLP at cutoff days (or anytime if cutoff=None).
    Excludes patients censored before cutoff without the event.
    """
    valid = sub.copy()

    if cutoff is None:
        elig = valid.copy()
        elig['ev'] = elig[event_col].astype(int)
    else:
        mask = (
            (valid['censor_days'] >= cutoff) |
            (valid[event_col].astype(bool) & (valid[days_col] <= cutoff))
        )
        elig = valid[mask].copy()
        elig['ev'] = (elig[event_col].astype(bool) & (elig[days_col] <= cutoff)).astype(int)

    t_ = elig[elig['slp'] == 1]
    c_ = elig[elig['slp'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    pt = f"{100*et/nt:.1f}" if nt > 0 else "N/A"
    pc = f"{100*ec/nc:.1f}" if nc > 0 else "N/A"

    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return nt, et, pt, nc, ec, pc, float('nan'), float('nan'), float('nan'), float('nan')

    or_v   = (a * d) / (b * c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return nt, et, pt, nc, ec, pc, or_v, lo, hi, p


def print_or_table(label, sub, outcomes_list):
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    header = (f"  {'Outcome':<30}  {'Cutoff':>7}  "
              f"{'n_SLP':>7} {'n_ev_SLP':>8} {'%_SLP':>6}  "
              f"{'n_NoSLP':>7} {'n_ev_NoSLP':>10} {'%_NoSLP':>7}  "
              f"{'OR':>6} {'95%CI':>14} {'p':>8}")
    print(header)
    print(f"  {'-'*98}")

    for label_o, ev_col, days_col, cutoffs in outcomes_list:
        for cutoff in cutoffs:
            cutoff_str = f"{cutoff}d" if cutoff else "any"
            row = compute_or(sub, ev_col, days_col, cutoff)
            nt, et, pt, nc, ec, pc, or_v, lo, hi, p = row
            if np.isnan(or_v):
                ci_str = "N/A"
                or_str = "N/A"
                p_str  = "N/A"
            else:
                ci_str = f"[{lo:.2f}, {hi:.2f}]"
                or_str = f"{or_v:.2f}"
                p_str  = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"  {label_o:<30}  {cutoff_str:>7}  "
                  f"{nt:>7,} {et:>8,} {pt:>6}  "
                  f"{nc:>7,} {ec:>10,} {pc:>7}  "
                  f"{or_str:>6} {ci_str:>14} {p_str:>8}")


# ── 4. Outcomes and strata ─────────────────────────────────────────────────────
TIMEPOINTS = [180, 365, 1095, 1825]

outcomes_list = [
    ('Mortality',       'died_{label}',       'days_to_death',        TIMEPOINTS),
    ('Readmission',     'readmit_{label}',    'days_to_readmit',      TIMEPOINTS),
    ('Recurrent Stroke','recur_{label}',      'days_to_recur_stroke', TIMEPOINTS),
    ('Aspiration PNA',  'asp_{label}',        'days_to_aspiration',   TIMEPOINTS),
    ('Dysphagia Dx',    'dysphagia_{label}',  'days_to_dysphagia',    TIMEPOINTS),
    ('G-tube',          'gtube_{label}',      'days_to_gtube',        TIMEPOINTS),
    ('SNF 30d',         'snf_30d',            'days_to_snf',          [None]),
    ('HHA 90d',         'hha_90d',            'days_to_snf',          [None]),
]

# Expand {label} templates into actual column names
def expand_outcomes(outcomes_list):
    expanded = []
    for name, ev_tmpl, days_col, cutoffs in outcomes_list:
        if '{label}' in ev_tmpl:
            row_cutoffs = []
            row_ev_cols = []
            for c in cutoffs:
                lbl = f"{c}d"
                row_ev_cols.append(ev_tmpl.replace('{label}', lbl))
                row_cutoffs.append(c)
            # emit one entry per timepoint so print_or_table works
            for ev_col, cutoff in zip(row_ev_cols, row_cutoffs):
                expanded.append((name, ev_col, days_col, [cutoff]))
        else:
            expanded.append((name, ev_tmpl, days_col, cutoffs))
    return expanded

outcomes_expanded = expand_outcomes(outcomes_list)

strata = [
    ('All matched',      df),
    ('Ischemic',         df[df['stroke_type'] == 'Ischemic']),
    ('ICH',              df[df['stroke_type'] == 'ICH']),
    ('Age < 75',         df[df['age_at_adm'] < 75]),
    ('Age 75-84',        df[(df['age_at_adm'] >= 75) & (df['age_at_adm'] < 85)]),
    ('Age 85+',          df[df['age_at_adm'] >= 85]),
    ('Dysphagia POA',    df[df['dysphagia_poa'] == 1]),
    ('No Dysphagia POA', df[df['dysphagia_poa'] == 0]),
    # Elixhauser tertiles
    ('Elix Low',         df[df['elix_grp'] == 'Low']),
    ('Elix Mid',         df[df['elix_grp'] == 'Mid']),
    ('Elix High',        df[df['elix_grp'] == 'High']),
    # Age x Elixhauser cross-strata (young vs old x low vs high)
    ('Age<75 + Elix Low',  df[(df['age_at_adm'] < 75)  & (df['elix_grp'] == 'Low')]),
    ('Age<75 + Elix High', df[(df['age_at_adm'] < 75)  & (df['elix_grp'] == 'High')]),
    ('Age75-84 + Elix Low',  df[(df['age_at_adm'] >= 75) & (df['age_at_adm'] < 85) & (df['elix_grp'] == 'Low')]),
    ('Age75-84 + Elix High', df[(df['age_at_adm'] >= 75) & (df['age_at_adm'] < 85) & (df['elix_grp'] == 'High')]),
    ('Age85+ + Elix Low',  df[(df['age_at_adm'] >= 85) & (df['elix_grp'] == 'Low')]),
    ('Age85+ + Elix High', df[(df['age_at_adm'] >= 85) & (df['elix_grp'] == 'High')]),
]

for strat_label, sub in strata:
    print_or_table(strat_label, sub, outcomes_expanded)


# ── 5. Cox PH — time-to-event outcomes (censored at 1825d) ────────────────────
print(f"\n{'='*80}")
print("  Cox PH Models (censored at 1825 days)")
print(f"{'='*80}")

cox_outcomes = [
    ('All-cause mortality',   'died_1825d',  'censor_days'),
    ('Recurrent stroke',      'recur_1825d', 'days_to_recur_stroke'),
    ('Aspiration PNA',        'asp_1825d',   'days_to_aspiration'),
    ('Dysphagia Dx',          'dysphagia_1825d', 'days_to_dysphagia'),
]

covariates = ['slp', 'age_at_adm', 'van_walraven_score', 'dysphagia_poa',
              'aspiration_poa', 'mech_vent', 'peg_placed', 'trach_placed', 'index_los']

for label_cox, event_col, dur_col in cox_outcomes:
    print(f"\n  {label_cox}")

    sub = df.copy()
    if dur_col not in sub.columns:
        print("    (duration column missing — skip)")
        continue

    sub['_dur'] = sub[dur_col].fillna(sub['censor_days']).astype(float)
    sub['_dur'] = sub['_dur'].clip(lower=0.5)
    sub['_ev']  = sub[event_col].astype(int)

    cox_df = sub[['_dur', '_ev'] + covariates].dropna()
    cox_df = cox_df.rename(columns={'_dur': 'duration', '_ev': 'event'})

    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='duration', event_col='event',
                show_progress=False)
        row = cph.summary.loc['slp']
        hr   = np.exp(row['coef'])
        lo95 = np.exp(row['coef lower 95%'])
        hi95 = np.exp(row['coef upper 95%'])
        p    = row['p']
        print(f"    HR={hr:.2f}  95%CI [{lo95:.2f}, {hi95:.2f}]  p={p:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")


# ── 6. KM survival at each time point ─────────────────────────────────────────
print(f"\n{'='*80}")
print("  Kaplan-Meier Survival / Cumulative Incidence at Each Time Point")
print(f"{'='*80}")

def km_estimate(kmf, t):
    """Return (survival_prob, ci_lower, ci_upper) at time t from fitted KMF."""
    s = float(kmf.predict(t))
    ci = kmf.confidence_interval_survival_function_
    idx = min(ci.index.searchsorted(t, side='right'), len(ci) - 1)
    lo = float(ci.iloc[idx, 0])
    hi = float(ci.iloc[idx, 1])
    return s, lo, hi


# (label, event_col, dur_col, survival=True means report survival; False = cumulative incidence)
km_outcomes = [
    ('All-cause mortality',  'died_1825d',       'censor_days',         True),
    ('Recurrent stroke',     'recur_1825d',       'days_to_recur_stroke',False),
    ('Aspiration PNA',       'asp_1825d',         'days_to_aspiration',  False),
    ('Dysphagia Dx',         'dysphagia_1825d',   'days_to_dysphagia',   False),
    ('G-tube',               'gtube_1825d',       'days_to_gtube',       False),
]

for label_km, event_col, dur_col, show_surv in km_outcomes:
    sub = df.copy()
    sub['_dur'] = sub[dur_col].fillna(sub['censor_days']).astype(float).clip(lower=0.5)
    sub['_ev']  = sub[event_col].astype(int)

    slp_s  = sub[sub['slp'] == 1]
    ctrl_s = sub[sub['slp'] == 0]

    kmf_slp  = KaplanMeierFitter()
    kmf_ctrl = KaplanMeierFitter()
    kmf_slp.fit(slp_s['_dur'],  slp_s['_ev'],  label='SLP')
    kmf_ctrl.fit(ctrl_s['_dur'], ctrl_s['_ev'], label='No SLP')

    lr = logrank_test(slp_s['_dur'], ctrl_s['_dur'],
                      event_observed_A=slp_s['_ev'],
                      event_observed_B=ctrl_s['_ev'])
    p_lr = lr.p_value
    p_str = "<0.0001" if p_lr < 0.0001 else f"{p_lr:.4f}"

    metric = "Survival %" if show_surv else "Cum. incidence %"
    print(f"\n  {label_km}  [{metric}]  Log-rank p={p_str}")
    print(f"  {'Time':<8}  {'SLP':>16}  {'No SLP':>16}")
    print(f"  {'-'*44}")
    for t in TIMEPOINTS:
        s_slp,  lo_slp,  hi_slp  = km_estimate(kmf_slp,  t)
        s_ctrl, lo_ctrl, hi_ctrl = km_estimate(kmf_ctrl, t)
        if show_surv:
            pct_slp  = f"{100*s_slp:.1f} [{100*lo_slp:.1f},{100*hi_slp:.1f}]"
            pct_ctrl = f"{100*s_ctrl:.1f} [{100*lo_ctrl:.1f},{100*hi_ctrl:.1f}]"
        else:
            pct_slp  = f"{100*(1-s_slp):.1f} [{100*(1-hi_slp):.1f},{100*(1-lo_slp):.1f}]"
            pct_ctrl = f"{100*(1-s_ctrl):.1f} [{100*(1-hi_ctrl):.1f},{100*(1-lo_ctrl):.1f}]"
        print(f"  {t}d{'':<5}  {pct_slp:>16}  {pct_ctrl:>16}")


# ── 7. Cost analysis ───────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  Medicare Cost Analysis (1825d / 5-year post-discharge)")
print(f"{'='*80}")
print(f"  {'Group':<12}  {'N':>7}  {'Mean $':>10}  {'Median $':>10}  {'MW p':>8}")
print(f"  {'-'*52}")

slp_c   = df[df['slp'] == 1]['total_pmt_1825d'].dropna()
noslp_c = df[df['slp'] == 0]['total_pmt_1825d'].dropna()
stat, mw_p = mannwhitneyu(slp_c, noslp_c, alternative='two-sided')
print(f"  {'SLP':<12}  {len(slp_c):>7,}  {slp_c.mean():>10,.0f}  "
      f"{slp_c.median():>10,.0f}  {'<0.0001' if mw_p < 0.0001 else f'{mw_p:.4f}':>8}")
print(f"  {'No SLP':<12}  {len(noslp_c):>7,}  {noslp_c.mean():>10,.0f}  "
      f"{noslp_c.median():>10,.0f}  {'':>8}")
print(f"\n  Mann-Whitney U test (two-sided): U={stat:.0f}  p={mw_p:.4f}")

# By stroke type
print(f"\n  Cost by stroke type:")
for stype in ['Ischemic', 'ICH', 'SAH']:
    sub = df[df['stroke_type'] == stype]
    s = sub[sub['slp'] == 1]['total_pmt_1825d'].dropna()
    n = sub[sub['slp'] == 0]['total_pmt_1825d'].dropna()
    if len(s) < 10 or len(n) < 10:
        continue
    _, p = mannwhitneyu(s, n, alternative='two-sided')
    print(f"  {stype:<12}  SLP: ${s.mean():,.0f}  No SLP: ${n.mean():,.0f}  "
          f"p={p:.4f}")

print("\nDone.")
