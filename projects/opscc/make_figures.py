"""
make_figures.py
Generates figures for the three-comparison OPSCC study:

  Comparison A — TORS alone  vs RT alone
  Comparison B — TORS + RT   vs CRT
  Comparison C — TORS + CRT  vs CRT

Per comparison:
  1. Overall survival Kaplan-Meier curve   (fig_km_{A,B,C}.png)
  2. Subgroup KM curves (age / Elixhauser) (fig_km_sub_{A,B,C}.png)
  3. Forest plot of canonical-timepoint ORs (fig_forest_{A,B,C}.png)
  4. Love plot of covariate balance        (fig_love_{A,B,C}.png)

Plus two cross-comparison figures:
  5. G-tube placement cumulative incidence across all comparisons
     at intervals 14d / 30d / 90d / 180d / 1yr / 3yr  (fig_cuminc_gtube.png)
  6. SLP visit cumulative incidence across all comparisons
     at intervals 14d / 30d / 90d / 180d / 1yr / 3yr  (fig_cuminc_slp.png)

All cohort assembly and FFS censoring are handled by SQL pipeline steps 1-14.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2_contingency
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
DB_PATH  = os.getenv("duckdb_database",    r"F:\CMS\cms_data.duckdb")
OUT_DIR  = str(Path(os.getenv("analysis_directory", r"C:\Users\hsaee\Desktop\CMS_viewer")) / "projects" / "opscc" / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SCARLET   = '#BA0C2F'
GRAY      = '#A7B1B7'
DARK40    = '#70071C'
DARK60    = '#4A0513'
CI_ALPHA  = 0.15

COMPARISONS = [
    ('A', 'psm_matched_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'TORS + RT',  'CRT'),
    ('C', 'psm_matched_C', 'TORS + CRT', 'CRT'),
]

# Forest plot outcomes — single canonical-timepoint OR per outcome per stratum.
# For G-tube/SLP the multi-interval timeline is shown in fig_cuminc_panel.
# G-tube uses placement-only flags (Z93.1 excluded). SLP uses any-event flags.
GTUBE_INTERVALS = [
    ('14d',   14,   'placement_by_14d'),
    ('30d',   30,   'placement_by_30d'),
    ('90d',   90,   'placement_by_90d'),
    ('180d',  180,  'placement_by_180d'),
    ('1-yr',  365,  'placement_by_365d'),
    ('3-yr',  1095, 'placement_by_1095d'),
]
SLP_INTERVALS = [
    ('14d',   14,   'event_by_14d'),
    ('30d',   30,   'event_by_30d'),
    ('90d',   90,   'event_by_90d'),
    ('180d',  180,  'event_by_180d'),
    ('1-yr',  365,  'event_by_365d'),
    ('3-yr',  1095, 'event_by_1095d'),
]

LEGACY_TIMEPOINTS = [
    ('6-mo',  182),
    ('1-yr',  365),
    ('3-yr', 1095),
    ('Any',   None),
]


def compute_or(sub, has_col, days_col, cutoff):
    """Follow-up-eligibility OR for binary outcomes (dysphagia)."""
    valid = sub[sub[has_col].notna()].copy()
    if cutoff is None:
        elig = valid.copy()
        elig['ev'] = elig[has_col].astype(int)
    else:
        mask = (
            (valid['follow_up_days'] >= cutoff) |
            ((valid[has_col] == True) & (valid[days_col] <= cutoff))
        )
        elig = valid[mask].copy()
        elig['ev'] = ((elig[has_col] == True) & (elig[days_col] <= cutoff)).astype(int)
    t_ = elig[elig['tors'] == 1]; c_ = elig[elig['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    return _or(et, nt, ec, nc)


def compute_or_flag(sub, flag_col):
    """Plain OR for a pre-built binary flag column."""
    valid = sub[sub[flag_col].notna()].copy()
    valid['ev'] = valid[flag_col].astype(int)
    t_ = valid[valid['tors'] == 1]; c_ = valid[valid['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    return _or(et, nt, ec, nc)


def _or(et, nt, ec, nc):
    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return float('nan'), float('nan'), float('nan'), False
    or_v   = (a * d) / (b * c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return or_v, lo, hi, (p < 0.05)


def wilson_band(k, n):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    z = 1.96
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0, centre - half), min(1, centre + half)


# ── SMD helpers for Love plot ─────────────────────────────────────────────────
def _smd_cont(a, b):
    diff   = a.mean() - b.mean()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(diff / pooled) if pooled > 0 else 0.0

def _smd_bin(a, b):
    p1, p2 = a.mean(), b.mean()
    denom  = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return abs((p1 - p2) / denom) if denom > 0 else 0.0

BAL_VARS = [
    ('age_at_dx',          'Age at diagnosis',     'cont'),
    ('van_walraven_score',  'van Walraven score',   'cont'),
    ('dx_year',             'Diagnosis year',       'cont'),
    ('male',                'Male sex',             'bin'),
    ('white',               'White',                'bin'),
    ('black',               'Black',                'bin'),
    ('hispanic',            'Hispanic',             'bin'),
    ('asian_pi',            'Asian/PI',             'bin'),
    ('coag',                'Coagulopathy',         'bin'),
    ('chf',                 'Heart failure',        'bin'),
    ('cpd',                 'Pulm. disease',        'bin'),
    ('rf',                  'Renal failure',        'bin'),
    ('subsite_c01',         'Base of tongue (C01)', 'bin'),
    ('subsite_c09',         'Tonsil (C09)',         'bin'),
    ('subsite_c10',         'Oropharynx (C10)',     'bin'),
    ('region_south',        'Region: South',        'bin'),
    ('region_midwest',      'Region: Midwest',      'bin'),
    ('region_west',         'Region: West',         'bin'),
]
BAL_VARS_NODAL = BAL_VARS + [('has_nodal_dx', 'Nodal disease†', 'bin')]


def plot_love(df_pre, df_post, tors_label, ctrl_label, comp, bal_vars):
    rows = []
    for col, label, vtype in bal_vars:
        t_pre  = df_pre[df_pre['treatment'] == 1][col]
        c_pre  = df_pre[df_pre['treatment'] == 0][col]
        t_post = df_post[df_post['treatment'] == 1][col]
        c_post = df_post[df_post['treatment'] == 0][col]
        fn = _smd_cont if vtype == 'cont' else _smd_bin
        rows.append({'label': label, 'before': fn(t_pre, c_pre), 'after': fn(t_post, c_post)})
    bal = pd.DataFrame(rows).sort_values('before', ascending=True).reset_index(drop=True)
    y = np.arange(len(bal))
    fig_h = max(5.0, len(bal) * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    fig.patch.set_facecolor('white')
    for i in y:
        ax.axhspan(i - 0.5, i + 0.5, color='#F9F9F9' if i % 2 == 0 else 'white', zorder=0)
    for i, row in bal.iterrows():
        ax.plot([row['before'], row['after']], [i, i], color='#cccccc', lw=0.8, zorder=1)
    ax.scatter(bal['before'], y, color=GRAY, s=50, zorder=3, marker='o',
               facecolors='white', edgecolors=GRAY, linewidths=1.5, label='Before matching')
    ax.scatter(bal['after'],  y, color=SCARLET, s=50, zorder=3, marker='o', label='After matching')
    ax.axvline(0.10, color='#888888', lw=1.2, linestyle='--', zorder=2, label='SMD = 0.10')
    ax.axvline(0.0,  color='#cccccc', lw=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(bal['label'], fontsize=8.5)
    ax.set_xlabel('Standardized Mean Difference', fontsize=9)
    ax.set_title(f'Covariate Balance — {tors_label} vs {ctrl_label}',
                 fontsize=10, fontweight='bold', color=DARK40, pad=8)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8); ax.set_xlim(left=-0.02)
    if comp == 'B':
        fig.text(0.12, 0.01, '† Not included in propensity score model',
                 fontsize=7, color='#555555', style='italic')
    plt.tight_layout(rect=[0, 0.03 if comp == 'B' else 0, 1, 1])
    out_path = rf"{OUT_DIR}\fig_love_{comp}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out_path}")


def plot_km(ax, t_df, c_df, tors_label, ctrl_label, title,
            show_table=True, max_yr=5, x_label='Years from treatment'):
    kmf_t = KaplanMeierFitter(label=tors_label)
    kmf_c = KaplanMeierFitter(label=ctrl_label)
    kmf_t.fit(t_df['t_years'], t_df['event'])
    kmf_c.fit(c_df['t_years'], c_df['event'])
    tl = kmf_t.confidence_interval_survival_function_
    cl = kmf_c.confidence_interval_survival_function_
    ax.fill_between(tl.index, tl.iloc[:, 0], tl.iloc[:, 1], alpha=CI_ALPHA, color=SCARLET, step='post')
    ax.fill_between(cl.index, cl.iloc[:, 0], cl.iloc[:, 1], alpha=CI_ALPHA, color=GRAY,    step='post')
    kmf_t.plot_survival_function(ax=ax, color=SCARLET, lw=2.0, ci_show=False)
    kmf_c.plot_survival_function(ax=ax, color=GRAY,    lw=2.0, ci_show=False, linestyle='--')
    lr    = logrank_test(t_df['t_years'], c_df['t_years'], t_df['event'], c_df['event'])
    p_str = 'p < 0.001' if lr.p_value < 0.001 else f'p = {lr.p_value:.3f}'
    ax.set_xlim(0, max_yr); ax.set_ylim(0, 1.05)
    ax.set_xlabel(x_label, fontsize=9); ax.set_ylabel('Overall Survival', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', color=DARK40, pad=6)
    ax.text(0.97, 0.97, f'Log-rank {p_str}', transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    if show_table:
        checkpoints = list(range(0, max_yr + 1))
        t_risk = [int((t_df['t_years'] >= cp).sum()) for cp in checkpoints]
        c_risk = [int((c_df['t_years'] >= cp).sum()) for cp in checkpoints]
        for cp, tr, cr in zip(checkpoints, t_risk, c_risk):
            ax.text(cp, -0.13, str(tr), ha='center', va='top', fontsize=7,
                    color=SCARLET, transform=ax.get_xaxis_transform())
            ax.text(cp, -0.20, str(cr), ha='center', va='top', fontsize=7,
                    color=GRAY,   transform=ax.get_xaxis_transform())
        t_short = tors_label.replace(' alone', '').replace(' + RT', '+RT').replace(' + CRT', '+CRT')
        c_short = ctrl_label.replace(' alone', '').replace(' + RT', '+RT')
        ax.text(-0.5, -0.13, t_short, ha='right', va='top', fontsize=7,
                color=SCARLET, transform=ax.get_xaxis_transform(), fontweight='bold')
        ax.text(-0.5, -0.20, c_short, ha='right', va='top', fontsize=7,
                color=GRAY,   transform=ax.get_xaxis_transform(), fontweight='bold')
    legend = ax.get_legend()
    if legend:
        legend.remove()


# ══════════════════════════════════════════════════════════════════════════════
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

# ── Pre-load propensity table for Love plots ───────────────────────────────────
prop_raw = con.execute("""
    SELECT DSYSRTKY, tx_group, sex, race, subsite, census_region,
           age_at_dx, van_walraven_score, dx_year,
           coag, chf, cpd, rf, has_nodal_dx,
           psm_matched_A, psm_matched_B, psm_matched_C
    FROM opscc_propensity
""").df()

prop_raw['male']     = (prop_raw['sex']  == 'Male').astype(int)
prop_raw['white']    = (prop_raw['race'] == 'White').astype(int)
prop_raw['black']    = (prop_raw['race'] == 'Black').astype(int)
prop_raw['hispanic'] = (prop_raw['race'] == 'Hispanic').astype(int)
prop_raw['asian_pi'] = (prop_raw['race'] == 'Asian/PI').astype(int)
prop_raw['van_walraven_score'] = prop_raw['van_walraven_score'].fillna(0)
for _f in ['coag', 'chf', 'cpd', 'rf', 'has_nodal_dx']:
    prop_raw[_f] = prop_raw[_f].fillna(0).astype(int)
prop_raw['dx_year']       = prop_raw['dx_year'].fillna(prop_raw['dx_year'].median()).astype(int)
prop_raw['subsite_c01']   = (prop_raw['subsite'] == 'C01').astype(int)
prop_raw['subsite_c09']   = (prop_raw['subsite'] == 'C09').astype(int)
prop_raw['subsite_c10']   = (prop_raw['subsite'] == 'C10').astype(int)
prop_raw['region_south']  = (prop_raw['census_region'] == 'South').astype(int)
prop_raw['region_midwest']= (prop_raw['census_region'] == 'Midwest').astype(int)
prop_raw['region_west']   = (prop_raw['census_region'] == 'West').astype(int)


# Hold per-comp outcome dataframes for the cross-comparison cumulative-incidence figure
out_by_comp = {}

for comp, match_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*60}\n  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}\n{'#'*60}")

    # ── 1. Survival data ────────────────────────────────────────────────────
    surv = con.execute(f"""
        SELECT s.DSYSRTKY, s.tx_group, s.first_tx_date,
               s.age_at_dx, s.van_walraven_score,
               s.event, s.t_days,
               DATEDIFF('day', s.first_tx_date, co.first_chemo_date) AS days_tx_to_chemo
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        LEFT JOIN opscc_cohort co ON co.DSYSRTKY = s.DSYSRTKY
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
          AND s.t_days >= 0
    """).df()

    if len(surv) == 0:
        print(f"  No matched patients for Comparison {comp}. Skipping figures.")
        out_by_comp[comp] = None
        continue

    # ── Comp C only: re-anchor survival on first_chemo_date ──────────────────
    # For TORS+CRT patients, days_tx_to_chemo > 0 (surgery first, then chemo).
    # For CRT patients, days_tx_to_chemo ~ 0 (chemo at first_tx_date).
    # Re-anchored t_days = days from chemo start. Patients who died before
    # chemo started (very rare) are dropped.
    if comp == 'C':
        surv['days_tx_to_chemo'] = surv['days_tx_to_chemo'].fillna(0)
        surv['t_days'] = surv['t_days'] - surv['days_tx_to_chemo']
        surv = surv[surv['t_days'] >= 0].copy()

    surv['t_years'] = surv['t_days'] / 365.25
    surv['elix'] = (surv['van_walraven_score'] > 0).map({False: 'Low', True: 'High'})
    tors_s = surv[surv['tx_group'] == tors_label]
    ctrl_s = surv[surv['tx_group'] == ctrl_label]
    print(f"  Survival cohort: {len(surv):,}  "
          f"{tors_label}={len(tors_s):,}  {ctrl_label}={len(ctrl_s):,}")

    # ── 2. Outcomes data ────────────────────────────────────────────────────
    out_df = con.execute(f"""
        SELECT
            s.DSYSRTKY, s.tx_group, s.first_tx_date,
            s.age_at_dx, s.van_walraven_score,
            f.ffs_censor_date,
            DATEDIFF('day', s.first_tx_date, f.ffs_censor_date)      AS follow_up_days,
            o.has_dysphagia,
            DATEDIFF('day', s.first_tx_date, o.first_dysphagia_date) AS days_dys,
            DATEDIFF('day', s.first_tx_date, co.first_chemo_date)    AS days_tx_to_chemo,
            g.first_post_placement_day AS g_first_post_placement_day,
            slp.first_post_day_from_tx AS slp_first_post_day_from_tx,
            g.placement_by_14d   AS g_placement_by_14d,
            g.placement_by_30d   AS g_placement_by_30d,
            g.placement_by_90d   AS g_placement_by_90d,
            g.placement_by_180d  AS g_placement_by_180d,
            g.placement_by_365d  AS g_placement_by_365d,
            g.placement_by_1095d AS g_placement_by_1095d,
            slp.event_by_14d   AS slp_event_by_14d,
            slp.event_by_30d   AS slp_event_by_30d,
            slp.event_by_90d   AS slp_event_by_90d,
            slp.event_by_180d  AS slp_event_by_180d,
            slp.event_by_365d  AS slp_event_by_365d,
            slp.event_by_1095d AS slp_event_by_1095d,
            g.days_tx_to_completion       AS days_tx_to_completion,
            g.delayed_placement_by_180d   AS g_delayed_by_180d,
            g.delayed_placement_by_365d   AS g_delayed_by_365d,
            g.delayed_placement_by_730d   AS g_delayed_by_730d,
            g.delayed_placement_by_1095d  AS g_delayed_by_1095d
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        JOIN opscc_ffs_dates  f USING (DSYSRTKY)
        JOIN opscc_outcomes   o USING (DSYSRTKY)
        LEFT JOIN opscc_cohort co  ON co.DSYSRTKY = s.DSYSRTKY
        LEFT JOIN opscc_slp   slp USING (DSYSRTKY)
        LEFT JOIN opscc_gtube_dependence g USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
    """).df()

    out_df['tors'] = (out_df['tx_group'] == tors_label).astype(int)
    out_df['elix_grp'] = (out_df['van_walraven_score'] > 0).map({False: 'Low', True: 'High'})

    # Follow-up from treatment completion (delayed-toxicity at-risk sets).
    # Computed from the original first_tx-anchored follow_up_days BEFORE the
    # Comp C chemo re-anchoring below mutates follow_up_days.
    out_df['foll_after_comp'] = out_df['follow_up_days'] - out_df['days_tx_to_completion']

    # ── Comp C only: re-anchor flags + days_dys on first_chemo_date ──────────
    # Events that occurred BEFORE chemo started are excluded from these chemo-
    # anchored flags (they live in the pre-CRT surgical era and are not part
    # of the head-to-head comparison from chemoradiation start).
    if comp == 'C':
        dchemo = out_df['days_tx_to_chemo'].fillna(0)

        # Re-anchor follow-up window on chemo start
        out_df['follow_up_days'] = out_df['follow_up_days'] - dchemo

        # Re-anchor dysphagia: days_dys becomes days from chemo. Events that
        # occurred before chemo (negative) are treated as "no event" in the
        # chemo-anchored timeline so they don't inflate post-chemo OR.
        out_df['days_dys'] = out_df['days_dys'] - dchemo
        pre_chemo_dys = out_df['days_dys'] < 0
        out_df.loc[pre_chemo_dys, 'has_dysphagia'] = False
        out_df.loc[pre_chemo_dys, 'days_dys'] = np.nan

        # Re-build G-tube placement_by_T flags from chemo start
        gd = out_df['g_first_post_placement_day'] - dchemo
        for T, lbl in [(14, '14d'), (30, '30d'), (90, '90d'),
                       (180, '180d'), (365, '365d'), (1095, '1095d')]:
            out_df[f'g_placement_by_{lbl}'] = (gd.notna()) & (gd >= 0) & (gd <= T)

        # Re-build SLP event_by_T flags from chemo start
        sd = out_df['slp_first_post_day_from_tx'] - dchemo
        for T, lbl in [(14, '14d'), (30, '30d'), (90, '90d'),
                       (180, '180d'), (365, '365d'), (1095, '1095d')]:
            out_df[f'slp_event_by_{lbl}'] = (sd.notna()) & (sd >= 0) & (sd <= T)

    out_by_comp[comp] = (out_df, tors_label, ctrl_label)
    print(f"  Outcomes cohort: {len(out_df):,}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Overall KM
    # ════════════════════════════════════════════════════════════════════════
    out_km = rf"{OUT_DIR}\fig_km_{comp}.png"
    km_xlabel = 'Years from chemoradiation start' if comp == 'C' else 'Years from treatment'
    km_subtitle = ' (anchored on chemoradiation start)' if comp == 'C' else ''
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('white'); fig.subplots_adjust(bottom=0.22)
    plot_km(ax, tors_s, ctrl_s, tors_label, ctrl_label,
            f'Overall Survival — {tors_label} vs {ctrl_label}{km_subtitle}',
            max_yr=5, x_label=km_xlabel)
    legend_elements = [
        Line2D([0],[0], color=SCARLET, lw=2,
               label=f'{tors_label} (n={len(tors_s):,})'),
        Line2D([0],[0], color=GRAY, lw=2, linestyle='--',
               label=f'{ctrl_label} (n={len(ctrl_s):,})'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9,
              loc='lower right', bbox_to_anchor=(0.99, 0.05))
    ax.text(0.5, -0.30, 'Numbers at risk', ha='center', va='top',
            fontsize=7.5, color='#555555',
            transform=ax.get_xaxis_transform(), style='italic')
    plt.savefig(out_km, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {out_km}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Subgroup KM (2×2)
    # ════════════════════════════════════════════════════════════════════════
    out_sub = rf"{OUT_DIR}\fig_km_sub_{comp}.png"
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.32)
    subgroups = [
        (surv[surv['age_at_dx'] < 75],        'Age < 75'),
        (surv[surv['age_at_dx'] >= 75],       'Age ≥75'),
        (surv[surv['elix'] == 'Low'],         'Low Comorbidity\n(VW ≤0)'),
        (surv[surv['elix'] == 'High'],        'High Comorbidity\n(VW >0)'),
    ]
    for idx, (sub, title) in enumerate(subgroups):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        t_sub = sub[sub['tx_group'] == tors_label]
        c_sub = sub[sub['tx_group'] == ctrl_label]
        if len(t_sub) < 10 or len(c_sub) < 10:
            ax.text(0.5, 0.5, 'Insufficient n', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#888888')
            ax.set_title(title, fontsize=12, fontweight='bold', color=DARK40, pad=6)
            ax.spines[['top','right']].set_visible(False)
            continue
        plot_km(ax, t_sub, c_sub, tors_label, ctrl_label, title,
                show_table=True, max_yr=5, x_label=km_xlabel)
    legend_elements = [
        Line2D([0],[0], color=SCARLET, lw=2,          label=tors_label),
        Line2D([0],[0], color=GRAY,    lw=2, ls='--', label=ctrl_label),
    ]
    fig.legend(handles=legend_elements, fontsize=12, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.suptitle(f'Overall Survival by Subgroup — {tors_label} vs {ctrl_label}{km_subtitle}',
                 fontsize=20, fontweight='bold', color=DARK40, y=1.01)
    plt.savefig(out_sub, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {out_sub}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Forest plot: 3 outcomes × 6 timepoints, OR over time
    # All-matched cohort. Subgroup ORs available in subgroup_analysis.py.
    # ════════════════════════════════════════════════════════════════════════
    out_frst = rf"{OUT_DIR}\fig_forest_{comp}.png"

    TIME_ROWS = [
        ('14d',   14),
        ('30d',   30),
        ('90d',   90),
        ('180d',  180),
        ('1-yr',  365),
        ('3-yr',  1095),
    ]
    TP_LABELS = [t for t, _ in TIME_ROWS]
    N_TP = len(TIME_ROWS)

    # (column label, kind, args)
    #   'legacy': dysphagia (uses has_dysphagia + days_dys + cutoff)
    #   'flag':   G-tube/SLP (uses pre-built event_by_T / placement_by_T flag)
    forest_outcomes = [
        ('Dysphagia',         'legacy',     ('has_dysphagia', 'days_dys')),
        ('G-tube placement',  'flag_gtube', None),
        ('SLP visit',         'flag_slp',   None),
    ]
    N_OUT = len(forest_outcomes)

    fig, axes = plt.subplots(1, N_OUT, figsize=(4.2 * N_OUT, 0.7 * N_TP + 2.2), sharey=True)
    fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.12)
    y = np.arange(N_TP)[::-1]  # top-to-bottom (14d at top, 3-yr at bottom)

    for ax, (label, kind, args) in zip(axes, forest_outcomes):
        for i, yi in enumerate(y):
            ax.axhspan(yi - 0.5, yi + 0.5, color='#FDF5F6' if i % 2 == 0 else '#FFFFFF', zorder=0)

        for tp_i, (tp_lbl, cutoff_d) in enumerate(TIME_ROWS):
            if kind == 'legacy':
                has_col, days_col = args
                or_v, lo, hi, sig = compute_or(out_df, has_col, days_col, cutoff_d)
            elif kind == 'flag_gtube':
                or_v, lo, hi, sig = compute_or_flag(out_df, f'g_placement_by_{cutoff_d}d')
            else:  # flag_slp
                or_v, lo, hi, sig = compute_or_flag(out_df, f'slp_event_by_{cutoff_d}d')

            yi = y[tp_i]
            if np.isnan(or_v):
                ax.text(1.0, yi, 'N/A', va='center', ha='center', fontsize=8, color='#999999')
                continue
            ax.plot([lo, hi], [yi, yi], color=SCARLET, lw=1.6, zorder=2, alpha=0.85)
            fc = SCARLET if sig else 'white'
            ax.scatter(or_v, yi, color=SCARLET, s=75, zorder=3,
                       facecolors=fc, edgecolors=SCARLET, linewidths=1.4)

        ax.axvline(1.0, color='#888888', lw=1.0, linestyle='--', zorder=1)
        ax.set_xscale('log'); ax.set_xlim(0.04, 20)
        ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks(y); ax.set_yticklabels(TP_LABELS, fontsize=10, color='#444444')
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlabel('Odds Ratio (log scale)', fontsize=10, labelpad=4)
        ax.set_title(label, fontsize=11, fontweight='bold', color=DARK60, pad=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.27, -0.22, f'← Favors {tors_label}', transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#555555', style='italic')
        ax.text(0.79, -0.22, f'Favors {ctrl_label} →', transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#555555', style='italic')

    # Y-axis label on leftmost panel
    y_axis_label = 'Time since chemoradiation start' if comp == 'C' else 'Time since first treatment'
    axes[0].set_ylabel(y_axis_label,
                       fontsize=11, fontweight='bold', color=DARK60, labelpad=8)

    legend_elements = [
        Line2D([0],[0], marker='o', color=SCARLET, markerfacecolor='white',
               markersize=8, lw=0, label='p ≥ 0.05'),
        Line2D([0],[0], marker='o', color=SCARLET, markerfacecolor=SCARLET,
               markersize=8, lw=0, label='p < 0.05'),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    forest_subtitle = ' (from chemoradiation start)' if comp == 'C' else ''
    fig.suptitle(f'Functional Outcomes Over Time — {tors_label} vs {ctrl_label}{forest_subtitle}',
                 fontsize=16, fontweight='bold', color=DARK60, y=1.01)
    plt.tight_layout(rect=[0.04, 0.14, 1, 1])
    plt.savefig(out_frst, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {out_frst}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Love plot
    # ════════════════════════════════════════════════════════════════════════
    if comp == 'A':
        df_pre = prop_raw[
            (prop_raw['tx_group'].isin([tors_label, ctrl_label])) &
            (prop_raw['has_nodal_dx'] == 0)
        ].copy()
    else:
        df_pre = prop_raw[prop_raw['tx_group'].isin([tors_label, ctrl_label])].copy()
    df_pre['treatment'] = (df_pre['tx_group'] == tors_label).astype(int)

    df_post = prop_raw[
        (prop_raw[match_col] == True) &
        (prop_raw['tx_group'].isin([tors_label, ctrl_label]))
    ].copy()
    df_post['treatment'] = (df_post['tx_group'] == tors_label).astype(int)
    bal_vars = BAL_VARS_NODAL if comp == 'B' else BAL_VARS
    plot_love(df_pre, df_post, tors_label, ctrl_label, comp, bal_vars)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 5 + 6 — Separate cumulative-incidence panels for G-tube and SLP
# Each: 1 row × 3 cols (Comp A, B, C)
# ══════════════════════════════════════════════════════════════════════════════

# X axis = day boundaries (log-spaced visually)
x_days = [14, 30, 90, 180, 365, 1095]
x_labels = ['14d', '30d', '90d', '180d', '1-yr', '3-yr']


def render_cuminc_panel(out_path, suptitle, y_label, prefix, intervals, ylim_top=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor('white')

    for col_idx, (comp, match_col, tors_label, ctrl_label) in enumerate(COMPARISONS):
        ax = axes[col_idx]
        bundle = out_by_comp.get(comp)
        if bundle is None:
            ax.text(0.5, 0.5, 'No matched patients', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='#888888')
            ax.set_title(f'Comp {comp}: {tors_label} vs {ctrl_label}',
                         fontsize=11, fontweight='bold', color=DARK40)
            continue
        df, tors_label, ctrl_label = bundle

        for arm_val, arm_label, color, ls in [
            (1, tors_label, SCARLET, '-'),
            (0, ctrl_label, GRAY,    '--'),
        ]:
            s = df[df['tors'] == arm_val]
            n = len(s)
            ps, los, his = [], [], []
            for _, _, flag in intervals:
                k = int(s[prefix + flag].fillna(False).sum())
                p, lo, hi = wilson_band(k, n)
                ps.append(100*p); los.append(100*lo); his.append(100*hi)
            ax.fill_between(x_days, los, his, color=color, alpha=CI_ALPHA, step='post')
            ax.step(x_days, ps, where='post', color=color, lw=2.2,
                    linestyle=ls, label=f'{arm_label} (n={n:,})')
            ax.scatter(x_days, ps, color=color, s=35, zorder=3)

        ax.set_xscale('log'); ax.set_xlim(10, 1500)
        ax.set_xticks(x_days); ax.set_xticklabels(x_labels, fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=11)
        x_lbl = 'Days since chemoradiation start' if comp == 'C' else 'Days since first treatment'
        ax.set_xlabel(x_lbl, fontsize=10)
        anchor_suffix = '\n(anchored on chemoradiation start)' if comp == 'C' else ''
        ax.set_title(f'Comp {comp}: {tors_label} vs {ctrl_label}{anchor_suffix}',
                     fontsize=11, fontweight='bold', color=DARK40, pad=4)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        if ylim_top is not None:
            ax.set_ylim(0, ylim_top)
        else:
            ax.set_ylim(bottom=0)

    fig.suptitle(suptitle, fontsize=16, fontweight='bold', color=DARK60, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


print("\n  Generating G-tube cumulative-incidence panel...")
render_cuminc_panel(
    out_path=rf"{OUT_DIR}\fig_cuminc_gtube.png",
    suptitle='Cumulative incidence of G-tube placement',
    y_label='% with ≥1 placement',
    prefix='g_',
    intervals=GTUBE_INTERVALS,
    ylim_top=40,
)

print("\n  Generating SLP cumulative-incidence panel...")
render_cuminc_panel(
    out_path=rf"{OUT_DIR}\fig_cuminc_slp.png",
    suptitle='Cumulative incidence of SLP visits',
    y_label='% with ≥1 SLP visit',
    prefix='slp_',
    intervals=SLP_INTERVALS,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Delayed G-tube placement after treatment completion (≥90d washout)
# Anchored on last RT/chemo date. Marker for delayed swallowing failure /
# G-tube dependence. All-comers cumulative incidence with at-risk denominators.
# ══════════════════════════════════════════════════════════════════════════════

DELAYED_X = [(180, '6-mo'), (365, '1-yr'), (730, '2-yr'), (1095, '3-yr')]


def render_delayed_gtube(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor('white')
    xs = [d for d, _ in DELAYED_X]

    for col_idx, (comp, match_col, tors_label, ctrl_label) in enumerate(COMPARISONS):
        ax = axes[col_idx]
        bundle = out_by_comp.get(comp)
        if bundle is None:
            ax.text(0.5, 0.5, 'No matched patients', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='#888888')
            ax.set_title(f'Comp {comp}: {tors_label} vs {ctrl_label}',
                         fontsize=11, fontweight='bold', color=DARK40)
            continue
        df, tors_label, ctrl_label = bundle

        for arm_val, arm_label, color, ls in [
            (1, tors_label, SCARLET, '-'),
            (0, ctrl_label, GRAY,    '--'),
        ]:
            s = df[df['tors'] == arm_val]
            ps, los, his, n_last = [], [], [], 0
            for L, _ in DELAYED_X:
                flag = s[f'g_delayed_by_{L}d'].fillna(False).astype(bool)
                elig = (s['foll_after_comp'] >= L) | flag
                e = s[elig]
                k = int(e[f'g_delayed_by_{L}d'].fillna(False).sum())
                n = len(e)
                n_last = n
                p, lo, hi = wilson_band(k, n)
                ps.append(100*p); los.append(100*lo); his.append(100*hi)
            ax.fill_between(xs, los, his, color=color, alpha=CI_ALPHA, step='post')
            ax.step(xs, ps, where='post', color=color, lw=2.2,
                    linestyle=ls, label=f'{arm_label} (n={len(s):,})')
            ax.scatter(xs, ps, color=color, s=35, zorder=3)

        ax.set_xticks(xs); ax.set_xticklabels([lbl for _, lbl in DELAYED_X], fontsize=9)
        ax.set_xlim(150, 1150)
        if col_idx == 0:
            ax.set_ylabel('% with delayed placement (>90d post-completion)', fontsize=11)
        ax.set_xlabel('Time after treatment completion', fontsize=10)
        ax.set_title(f'Comp {comp}: {tors_label} vs {ctrl_label}',
                     fontsize=11, fontweight='bold', color=DARK40, pad=4)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.set_ylim(bottom=0)

    fig.suptitle('Delayed G-tube placement after treatment completion (≥90d washout)',
                 fontsize=16, fontweight='bold', color=DARK60, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


print("\n  Generating delayed G-tube placement panel...")
render_delayed_gtube(rf"{OUT_DIR}\fig_delayed_gtube.png")

con.close()
print("\nAll figures saved.")
