"""
make_figures.py
Generates six figures for the two-comparison OPSCC study:

  Comparison A — TORS alone vs RT alone
  Comparison B — TORS + RT  vs CT/CRT

Per comparison:
  1. Overall survival Kaplan-Meier curve   (fig_km_A.png      / fig_km_B.png)
  2. Subgroup KM curves (age / Elixhauser) (fig_km_sub_A.png  / fig_km_sub_B.png)
  3. Forest plot — functional outcome ORs  (fig_forest_A.png  / fig_forest_B.png)

All cohort assembly, C77 exclusion, and FFS censoring are handled by SQL
pipeline steps 10-12. This script only reads pre-built tables.
"""
import sys
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

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_DIR  = r"F:\CMS\projects\opscc"

# ── OSU colour scheme ──────────────────────────────────────────────────────────
SCARLET   = '#BA0C2F'
GRAY      = '#A7B1B7'
DARK40    = '#70071C'
DARK60    = '#4A0513'
CI_ALPHA  = 0.15

COMPARISONS = [
    ('A', 'psm_matched_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'TORS + RT',  'CT/CRT'),
]

OUTCOMES = [
    ('Dysphagia',    'has_dysphagia',    'days_dys'),
    ('G-tube',       'has_gtube',        'days_gt'),
    ('Tracheostomy', 'has_tracheostomy', 'days_tr'),
]

TIMEPOINTS = [
    ('6-mo',  182),
    ('1-yr',  365),
    ('3-yr', 1095),
    ('5-yr', 1825),
    ('Any',   None),
]

# ── OR helper (mirrors outcomes_analysis.py) ──────────────────────────────────
def compute_or(sub, has_col, days_col, cutoff):
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

    t_ = elig[elig['tors'] == 1]
    c_ = elig[elig['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return float('nan'), float('nan'), float('nan'), False
    or_v   = (a * d) / (b * c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return or_v, lo, hi, (p < 0.05)


# ── KM panel helper ────────────────────────────────────────────────────────────
def plot_km(ax, t_df, c_df, tors_label, ctrl_label, title,
            show_table=True, max_yr=5):
    kmf_t = KaplanMeierFitter(label=tors_label)
    kmf_c = KaplanMeierFitter(label=ctrl_label)
    kmf_t.fit(t_df['t_years'], t_df['event'])
    kmf_c.fit(c_df['t_years'], c_df['event'])

    tl = kmf_t.confidence_interval_survival_function_
    cl = kmf_c.confidence_interval_survival_function_

    ax.fill_between(tl.index, tl.iloc[:, 0], tl.iloc[:, 1],
                    alpha=CI_ALPHA, color=SCARLET, step='post')
    ax.fill_between(cl.index, cl.iloc[:, 0], cl.iloc[:, 1],
                    alpha=CI_ALPHA, color=GRAY,    step='post')

    kmf_t.plot_survival_function(ax=ax, color=SCARLET, lw=2.0, ci_show=False)
    kmf_c.plot_survival_function(ax=ax, color=GRAY,    lw=2.0, ci_show=False,
                                 linestyle='--')

    lr    = logrank_test(t_df['t_years'], c_df['t_years'],
                         t_df['event'],  c_df['event'])
    p_str = 'p < 0.001' if lr.p_value < 0.001 else f'p = {lr.p_value:.3f}'

    ax.set_xlim(0, max_yr)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Years from treatment', fontsize=9)
    ax.set_ylabel('Overall Survival', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', color=DARK40, pad=6)
    ax.text(0.97, 0.97, f'Log-rank {p_str}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    if show_table:
        checkpoints = list(range(0, max_yr + 1))
        t_risk = [int((t_df['t_years'] >= cp).sum()) for cp in checkpoints]
        c_risk = [int((c_df['t_years'] >= cp).sum()) for cp in checkpoints]
        for cp, tr, cr in zip(checkpoints, t_risk, c_risk):
            ax.text(cp, -0.13, str(tr), ha='center', va='top', fontsize=7,
                    color=SCARLET, transform=ax.get_xaxis_transform())
            ax.text(cp, -0.20, str(cr), ha='center', va='top', fontsize=7,
                    color=GRAY,   transform=ax.get_xaxis_transform())
        # shorten labels for at-risk table to avoid clipping
        t_short = tors_label.replace(' alone', '').replace(' + RT', '+RT')
        c_short = ctrl_label.replace(' alone', '')
        ax.text(-0.5, -0.13, t_short, ha='right', va='top', fontsize=7,
                color=SCARLET, transform=ax.get_xaxis_transform(),
                fontweight='bold')
        ax.text(-0.5, -0.20, c_short, ha='right', va='top', fontsize=7,
                color=GRAY,   transform=ax.get_xaxis_transform(),
                fontweight='bold')

    legend = ax.get_legend()
    if legend:
        legend.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
con = duckdb.connect(DB_PATH, read_only=True)
con.execute(
    "SET memory_limit='24GB'; SET threads=12; "
    "SET temp_directory='F:\\CMS\\duckdb_temp';"
)

for comp, match_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*60}")
    print(f"  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}")
    print(f"{'#'*60}")

    # ── 1. Survival data — from opscc_survival (C77 excluded, Dec-31 censor) ──
    print("  Loading survival data...")
    surv = con.execute(f"""
        SELECT s.DSYSRTKY, s.tx_group, s.first_tx_date,
               s.age_at_dx, s.van_walraven_score,
               s.event, s.t_days
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
          AND s.t_days >= 0
    """).df()

    surv['t_years'] = surv['t_days'] / 365.25
    t1_s = surv['van_walraven_score'].quantile(1/3)
    t2_s = surv['van_walraven_score'].quantile(2/3)
    surv['elix'] = pd.cut(surv['van_walraven_score'],
                          bins=[-np.inf, t1_s, t2_s, np.inf],
                          labels=['Low', 'Mid', 'High'])

    tors_s = surv[surv['tx_group'] == tors_label]
    ctrl_s = surv[surv['tx_group'] == ctrl_label]
    print(f"  Survival cohort (C77 excl): {len(surv):,}  "
          f"{tors_label}={len(tors_s):,}  {ctrl_label}={len(ctrl_s):,}")

    # ── 2. Outcomes data — from opscc_ffs_dates (C77 excl, FFS censor) ────────
    print("  Loading outcomes data...")
    out_df = con.execute(f"""
        SELECT
            s.DSYSRTKY, s.tx_group, s.first_tx_date,
            s.age_at_dx, s.van_walraven_score,
            f.ffs_censor_date,
            DATEDIFF('day', s.first_tx_date, f.ffs_censor_date)      AS follow_up_days,
            o.has_dysphagia,
            DATEDIFF('day', s.first_tx_date, o.first_dysphagia_date) AS days_dys,
            o.has_gtube,
            DATEDIFF('day', s.first_tx_date, o.first_gtube_date)     AS days_gt,
            o.has_tracheostomy,
            DATEDIFF('day', s.first_tx_date, o.first_trach_date)     AS days_tr
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        JOIN opscc_ffs_dates  f USING (DSYSRTKY)
        JOIN opscc_outcomes   o USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
    """).df()

    out_df['tors'] = (out_df['tx_group'] == tors_label).astype(int)
    q1 = out_df['van_walraven_score'].quantile(1/3)
    q2 = out_df['van_walraven_score'].quantile(2/3)
    out_df['elix_grp'] = pd.cut(out_df['van_walraven_score'],
                                 bins=[-np.inf, q1, q2, np.inf],
                                 labels=['Low', 'Mid', 'High'])
    print(f"  Outcomes cohort (C77 excl):  {len(out_df):,}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Overall KM (C77 excluded)
    # ════════════════════════════════════════════════════════════════════════
    out_km = rf"{OUT_DIR}\fig_km_{comp}.png"
    print(f"  Generating Overall KM...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(bottom=0.22)

    plot_km(ax, tors_s, ctrl_s, tors_label, ctrl_label,
            f'Overall Survival — {tors_label} vs {ctrl_label}', max_yr=5)

    legend_elements = [
        Line2D([0],[0], color=SCARLET, lw=2,
               label=f'{tors_label} (n={len(tors_s):,})'),
        Line2D([0],[0], color=GRAY, lw=2, linestyle='--',
               label=f'{ctrl_label} (n={len(ctrl_s):,})'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9,
              loc='upper right', bbox_to_anchor=(0.99, 0.88))
    ax.text(0.5, -0.30, 'Numbers at risk',
            ha='center', va='top', fontsize=7.5, color='#555555',
            transform=ax.get_xaxis_transform(), style='italic')

    plt.savefig(out_km, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out_km}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Subgroup KM (2 × 3 grid, C77 excluded)
    # ════════════════════════════════════════════════════════════════════════
    out_sub = rf"{OUT_DIR}\fig_km_sub_{comp}.png"
    print(f"  Generating Subgroup KM...")
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.32)

    subgroups = [
        (surv,                                'All Matched (C77 excl)'),
        (surv[surv['age_at_dx'] < 75],        'Age < 75'),
        (surv[surv['age_at_dx'] >= 75],       'Age \u226575'),
        (surv[surv['elix'] == 'Low'],
         f'Low Comorbidity\n(VW \u2264{t1_s:.0f})'),
        (surv[surv['elix'] == 'Mid'],
         f'Mid Comorbidity\n(VW {t1_s:.0f}\u2013{t2_s:.0f})'),
        (surv[surv['elix'] == 'High'],
         f'High Comorbidity\n(VW >{t2_s:.0f})'),
    ]

    for idx, (sub, title) in enumerate(subgroups):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        t_sub = sub[sub['tx_group'] == tors_label]
        c_sub = sub[sub['tx_group'] == ctrl_label]
        if len(t_sub) < 10 or len(c_sub) < 10:
            ax.text(0.5, 0.5, 'Insufficient n', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#888888')
            ax.set_title(title, fontsize=10, fontweight='bold',
                         color=DARK40, pad=6)
            ax.spines[['top','right']].set_visible(False)
            continue
        plot_km(ax, t_sub, c_sub, tors_label, ctrl_label,
                title, show_table=True, max_yr=5)

    legend_elements = [
        Line2D([0],[0], color=SCARLET, lw=2,          label=tors_label),
        Line2D([0],[0], color=GRAY,    lw=2, ls='--', label=ctrl_label),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)
    fig.suptitle(f'Overall Survival by Subgroup — {tors_label} vs {ctrl_label}',
                 fontsize=13, fontweight='bold', color=DARK40, y=1.01)

    plt.savefig(out_sub, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out_sub}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Forest plot (subgroup × timepoint, all 3 outcomes)
    # ════════════════════════════════════════════════════════════════════════
    out_frst = rf"{OUT_DIR}\fig_forest_{comp}.png"
    print(f"  Computing ORs for forest plot...")

    strata_defs = [
        ('All matched',                         out_df),
        ('Age < 75',                            out_df[out_df['age_at_dx'] < 75]),
        ('Age \u226575',                        out_df[out_df['age_at_dx'] >= 75]),
        (f'Low comorbidity\n(VW \u2264{q1:.0f})', out_df[out_df['elix_grp'] == 'Low']),
        (f'High comorbidity\n(VW >{q2:.0f})',   out_df[out_df['elix_grp'] == 'High']),
    ]
    SG_LABELS = [s for s, _ in strata_defs]
    N_SG = len(SG_LABELS)

    # Build [outcome][sg][tp] data arrays
    TP_LABELS  = [tp for tp, _ in TIMEPOINTS]
    TP_COLORS  = ['#74C2E1', '#2E86AB', '#1A5276', '#0D3B66', '#C0392B']
    TP_MARKERS = ['o', 's', '^', 'D', 'P']
    N_TP = len(TIMEPOINTS)

    outcomes_plot = []
    for out_lbl, has_col, days_col in OUTCOMES:
        sg_rows = []
        for _, sub in strata_defs:
            tp_row = []
            for _, cutoff in TIMEPOINTS:
                or_v, lo, hi, sig = compute_or(sub, has_col, days_col, cutoff)
                tp_row.append(None if np.isnan(or_v) else (or_v, lo, hi, sig))
            sg_rows.append(tp_row)
        outcomes_plot.append((out_lbl, sg_rows))

    ROW_H = 0.65
    GAP   = 1.2

    def row_y(sg, tp):
        base = (N_SG - 1 - sg) * (N_TP * ROW_H + GAP)
        return base + (N_TP - 1 - tp) * ROW_H

    y_min = row_y(N_SG - 1, N_TP - 1) - 0.7
    y_max = row_y(0, 0) + 0.7
    y_sg_center = {sg: (row_y(sg, 0) + row_y(sg, N_TP - 1)) / 2
                   for sg in range(N_SG)}

    fig_h = (y_max - y_min) * 0.42 + 1.5
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_h), sharey=True)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.08)

    for ax, (outcome, sg_data) in zip(axes, outcomes_plot):
        # Alternating row shading
        for sg in range(N_SG):
            y_lo = row_y(sg, N_TP - 1) - 0.5
            y_hi = row_y(sg, 0) + 0.5
            shade = '#FDF5F6' if sg % 2 == 0 else '#FFFFFF'
            ax.axhspan(y_lo, y_hi, color=shade, zorder=0)

        # Subgroup labels (left panel only)
        for sg in range(N_SG):
            if ax == axes[0]:
                ax.text(-0.28, y_sg_center[sg], SG_LABELS[sg],
                        ha='right', va='center', fontsize=8.5,
                        fontweight='bold', color=DARK60,
                        transform=ax.get_yaxis_transform(), linespacing=1.3)

        # Data points
        for sg, sg_row in enumerate(sg_data):
            for tp, entry in enumerate(sg_row):
                y = row_y(sg, tp)
                if entry is None:
                    ax.text(1.0, y, 'N/A', va='center', ha='center',
                            fontsize=6.5, color='#999999')
                    continue
                or_v, lo, hi, sig = entry
                col = TP_COLORS[tp]
                ax.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, alpha=0.85)
                fc = col if sig else 'white'
                ax.scatter(or_v, y, color=col, s=55, zorder=3,
                           marker=TP_MARKERS[tp], facecolors=fc,
                           edgecolors=col, linewidths=1.2)

        ax.axvline(1.0, color='#888888', lw=1.0, linestyle='--', zorder=1)
        ax.set_xscale('log')
        ax.set_xlim(0.04, 20)
        ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([row_y(sg, tp)
                       for sg in range(N_SG) for tp in range(N_TP)])
        ax.set_yticklabels(
            [TP_LABELS[tp] for sg in range(N_SG) for tp in range(N_TP)],
            fontsize=7, color='#444444')
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel('Odds Ratio (log scale)', fontsize=9)
        ax.set_title(outcome, fontsize=11, fontweight='bold', color=DARK60, pad=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.38, -0.04, f'\u2190 Favors {tors_label}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, color='#555555', style='italic')
        ax.text(0.72, -0.04, f'Favors {ctrl_label} \u2192',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, color='#555555', style='italic')

    legend_elements = [
        matplotlib.lines.Line2D([0],[0], marker=TP_MARKERS[i], color=TP_COLORS[i],
                                 markerfacecolor=TP_COLORS[i], markersize=7,
                                 label=TP_LABELS[i], lw=1.2)
        for i in range(N_TP)
    ]
    legend_elements += [
        matplotlib.lines.Line2D([0],[0], marker='o', color='#666666',
                                 markerfacecolor='white', markersize=7, lw=0,
                                 label='p \u2265 0.05 (open)'),
        matplotlib.lines.Line2D([0],[0], marker='o', color='#666666',
                                 markerfacecolor='#666666', markersize=7, lw=0,
                                 label='p < 0.05 (filled)'),
    ]
    fig.legend(handles=legend_elements, fontsize=8.5, loc='lower center',
               ncol=7, bbox_to_anchor=(0.5, -0.03), framealpha=0.9,
               title='Time from treatment', title_fontsize=8.5)
    fig.suptitle(
        f'Odds Ratios for Functional Outcomes — {tors_label} vs {ctrl_label}'
        f'  (PSM cohort, C77 excl.)',
        fontsize=12, fontweight='bold', color=DARK60, y=1.01)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig(out_frst, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out_frst}")

con.close()
print("\nAll figures saved.")
