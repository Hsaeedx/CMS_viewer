"""
make_figures.py
Generates three figures for OPSCC TORS vs CT/CRT study:
  1. Overall survival Kaplan-Meier curve
  2. Subgroup survival KM curves (age <75/>=75, Elixhauser tertile)
  3. Forest plot — functional outcome odds ratios (anytime, by subgroup)
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
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

DB_PATH   = r"F:\CMS\cms_data.duckdb"
OUT_KM    = r"F:\CMS\projects\opscc\fig_km_overall.png"
OUT_SUB   = r"F:\CMS\projects\opscc\fig_km_subgroup.png"
OUT_FORST = r"F:\CMS\projects\opscc\fig_forest.png"

TORS_COL  = '#1F6B3A'
CTCRT_COL = '#2E5090'
CI_ALPHA  = 0.15

# ── 1. Pull matched cohort survival data ──────────────────────────────────────
print("Loading survival data...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

df = con.execute("""
    WITH c77 AS (
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN inp_claimsk_all i ON i.DSYSRTKY = o.DSYSRTKY,
        UNNEST([i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
                i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,
                i.ICD_DGNS_CD4,  i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,
                i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,  i.ICD_DGNS_CD9,
                i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
                i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN out_claimsk_all oc ON oc.DSYSRTKY = o.DSYSRTKY,
        UNNEST([oc.PRNCPAL_DGNS_CD, oc.ICD_DGNS_CD1, oc.ICD_DGNS_CD2,
                oc.ICD_DGNS_CD3, oc.ICD_DGNS_CD4,  oc.ICD_DGNS_CD5,
                oc.ICD_DGNS_CD6, oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
                oc.ICD_DGNS_CD9, oc.ICD_DGNS_CD10]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN car_linek_all cl ON cl.DSYSRTKY = o.DSYSRTKY
        WHERE cl.LINE_ICD_DGNS_CD LIKE 'C77%'
          AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
    ),
    matched AS (
        SELECT DSYSRTKY, tx_group, first_tx_date, psm_match_id,
               van_walraven_score, age_at_dx
        FROM opscc_propensity
        WHERE psm_matched = TRUE
          AND tx_group IN ('TORS only', 'CT/CRT only')
          AND DSYSRTKY NOT IN (SELECT DSYSRTKY FROM c77)
    ),
    mbsf_sum AS (
        SELECT m.DSYSRTKY,
            MAX(CAST(m.RFRNC_YR AS INTEGER))                        AS last_yr,
            MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                     THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END)  AS death_date
        FROM mbsf_all m
        JOIN matched p ON m.DSYSRTKY = p.DSYSRTKY
        GROUP BY m.DSYSRTKY
    )
    SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
           p.age_at_dx, p.van_walraven_score,
           s.death_date,
           make_date(s.last_yr, 12, 31) AS censor_date
    FROM matched p
    JOIN mbsf_sum s ON p.DSYSRTKY = s.DSYSRTKY
""").df()
con.close()

df['first_tx_date'] = pd.to_datetime(df['first_tx_date'])
df['death_date']    = pd.to_datetime(df['death_date'])
df['censor_date']   = pd.to_datetime(df['censor_date'])
df['event_date']    = df['death_date'].combine_first(df['censor_date'])
df['event']         = df['death_date'].notna().astype(int)
df['t_days']        = (df['event_date'] - df['first_tx_date']).dt.days
df['t_years']       = df['t_days'] / 365.25
df = df[df['first_tx_date'].notna() & (df['t_days'] >= 0)].copy()

# Elixhauser tertiles
t1 = df['van_walraven_score'].quantile(1/3)
t2 = df['van_walraven_score'].quantile(2/3)
df['elix'] = pd.cut(df['van_walraven_score'],
                    bins=[-np.inf, t1, t2, np.inf],
                    labels=['Low', 'Mid', 'High'])

tors  = df[df['tx_group'] == 'TORS only']
ctcrt = df[df['tx_group'] == 'CT/CRT only']
print(f"  Matched cohort: {len(df):,}  TORS={len(tors):,}  CT/CRT={len(ctcrt):,}")


# ── KM helper ─────────────────────────────────────────────────────────────────
def plot_km(ax, t_df, c_df, title, show_table=True, max_yr=8):
    kmf_t = KaplanMeierFitter(label='TORS')
    kmf_c = KaplanMeierFitter(label='CT/CRT')
    kmf_t.fit(t_df['t_years'], t_df['event'])
    kmf_c.fit(c_df['t_years'], c_df['event'])

    tl = kmf_t.confidence_interval_survival_function_
    cl = kmf_c.confidence_interval_survival_function_

    timeline = np.linspace(0, max_yr, 500)

    # CI bands
    ax.fill_between(tl.index, tl.iloc[:,0], tl.iloc[:,1],
                    alpha=CI_ALPHA, color=TORS_COL, step='post')
    ax.fill_between(cl.index, cl.iloc[:,0], cl.iloc[:,1],
                    alpha=CI_ALPHA, color=CTCRT_COL, step='post')

    # KM step curves
    kmf_t.plot_survival_function(ax=ax, color=TORS_COL,  lw=2.0, ci_show=False)
    kmf_c.plot_survival_function(ax=ax, color=CTCRT_COL, lw=2.0, ci_show=False,
                                 linestyle='--')

    lr = logrank_test(t_df['t_years'], c_df['t_years'], t_df['event'], c_df['event'])
    p_str = 'p < 0.001' if lr.p_value < 0.001 else f'p = {lr.p_value:.3f}'

    ax.set_xlim(0, max_yr)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Years from treatment', fontsize=9)
    ax.set_ylabel('Overall Survival', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', color='#1F4E79', pad=6)
    ax.text(0.97, 0.97, f'Log-rank {p_str}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    # At-risk table
    if show_table:
        checkpoints = list(range(0, max_yr + 1))
        t_risk = [int((t_df['t_years'] >= cp).sum()) for cp in checkpoints]
        c_risk = [int((c_df['t_years'] >= cp).sum()) for cp in checkpoints]
        y_t = -0.13
        y_c = -0.20
        for xi, (cp, tr, cr) in enumerate(zip(checkpoints, t_risk, c_risk)):
            ax.text(cp, y_t, str(tr), ha='center', va='top', fontsize=7,
                    color=TORS_COL, transform=ax.get_xaxis_transform())
            ax.text(cp, y_c, str(cr), ha='center', va='top', fontsize=7,
                    color=CTCRT_COL, transform=ax.get_xaxis_transform())
        ax.text(-0.5, y_t, 'TORS',   ha='right', va='top', fontsize=7,
                color=TORS_COL,  transform=ax.get_xaxis_transform(), fontweight='bold')
        ax.text(-0.5, y_c, 'CT/CRT', ha='right', va='top', fontsize=7,
                color=CTCRT_COL, transform=ax.get_xaxis_transform(), fontweight='bold')

    legend = ax.get_legend()
    if legend:
        legend.remove()


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Overall KM
# ════════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Overall KM...")
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor('white')
fig.subplots_adjust(bottom=0.22)

plot_km(ax, tors, ctcrt, 'Overall Survival — TORS vs CT/CRT', max_yr=5)

legend_elements = [
    Line2D([0],[0], color=TORS_COL,  lw=2, label=f'TORS (n={len(tors):,})'),
    Line2D([0],[0], color=CTCRT_COL, lw=2, linestyle='--',
           label=f'CT/CRT (n={len(ctcrt):,})'),
]
ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9,
          loc='upper right', bbox_to_anchor=(0.99, 0.88))

ax.text(0.5, -0.30,
        'Numbers at risk',
        ha='center', va='top', fontsize=7.5, color='#555555',
        transform=ax.get_xaxis_transform(), style='italic')

plt.savefig(OUT_KM, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_KM}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Subgroup KM (2 × 3 grid)
# ════════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Subgroup KM...")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.32)

subgroups = [
    (df,                                  'All Matched'),
    (df[df['age_at_dx'] < 75],            'Age < 75'),
    (df[df['age_at_dx'] >= 75],           'Age \u226575'),
    (df[df['elix'] == 'Low'],             f'Low Comorbidity\n(van Walraven \u22640)'),
    (df[df['elix'] == 'Mid'],             f'Mid Comorbidity\n(van Walraven 1\u20134)'),
    (df[df['elix'] == 'High'],            f'High Comorbidity\n(van Walraven >4)'),
]

for idx, (sub, title) in enumerate(subgroups):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    t_sub = sub[sub['tx_group'] == 'TORS only']
    c_sub = sub[sub['tx_group'] == 'CT/CRT only']
    plot_km(ax, t_sub, c_sub, title, show_table=True, max_yr=5)

# Shared legend
legend_elements = [
    Line2D([0],[0], color=TORS_COL,  lw=2,           label='TORS only'),
    Line2D([0],[0], color=CTCRT_COL, lw=2, ls='--',  label='CT/CRT only'),
]
fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
           ncol=2, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

fig.suptitle('Overall Survival by Subgroup — TORS vs CT/CRT',
             fontsize=13, fontweight='bold', color='#1F4E79', y=1.01)

plt.savefig(OUT_SUB, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_SUB}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Forest plot (subgroup × timepoint, all 3 outcomes)
# ════════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Forest plot (subgroup × timepoint)...")

# ── Timepoint styling ─────────────────────────────────────────────────────────
TP_LABELS  = ['6-mo', '1-yr', '3-yr', '5-yr', 'Any']
TP_COLORS  = ['#74C2E1', '#2E86AB', '#1A5276', '#0D3B66', '#C0392B']
TP_MARKERS = ['o', 's', '^', 'D', 'P']
N_TP = 5

# ── Subgroup definitions ──────────────────────────────────────────────────────
SG_LABELS = [
    'All matched',
    'Age < 75',
    'Age \u226575',
    'Low comorbidity\n(VW \u22640)',
    'High comorbidity\n(VW >3)',
]
N_SG = len(SG_LABELS)

# ── Data: [sg][tp] = (OR, lo, hi, sig) or None if N/A ────────────────────────
# Rows: All, Age<75, Age>=75, Elix Low, Elix High
# Cols: 6-mo, 1-yr, 3-yr, 5-yr, Anytime
dysphagia_data = [
    [(0.83,0.62,1.11,False),(0.72,0.54,0.98,True), (0.50,0.35,0.73,True), (0.42,0.25,0.69,True), (0.64,0.48,0.84,True) ],
    [(0.71,0.51,0.99,True), (0.60,0.43,0.84,True), (0.45,0.30,0.69,True), (0.38,0.22,0.67,True), (0.55,0.40,0.75,True) ],
    [(1.59,0.82,3.09,False),(1.52,0.78,2.96,False),(0.76,0.34,1.71,False),(0.59,0.19,1.84,False),(1.18,0.62,2.22,False)],
    [(0.69,0.47,1.01,False),(0.59,0.40,0.86,True), (0.39,0.25,0.64,True), (0.32,0.17,0.62,True), (0.47,0.33,0.68,True) ],
    [(1.06,0.63,1.78,False),(0.88,0.52,1.49,False),(0.80,0.44,1.48,False),(0.81,0.34,1.91,False),(0.91,0.55,1.50,False)],
]
gtube_data = [
    [(0.30,0.22,0.41,True),(0.25,0.18,0.34,True),(0.29,0.21,0.40,True),(0.33,0.22,0.48,True),(0.29,0.22,0.38,True)],
    [(0.28,0.20,0.41,True),(0.24,0.17,0.34,True),(0.29,0.20,0.41,True),(0.32,0.21,0.49,True),(0.26,0.19,0.36,True)],
    [(0.36,0.18,0.72,True),(0.31,0.16,0.60,True),(0.30,0.15,0.60,True),(0.34,0.14,0.82,True),(0.43,0.24,0.77,True)],
    [(0.31,0.20,0.47,True),(0.25,0.16,0.38,True),(0.30,0.20,0.45,True),(0.32,0.20,0.51,True),(0.27,0.19,0.38,True)],
    [(0.29,0.17,0.50,True),(0.25,0.15,0.42,True),(0.27,0.15,0.47,True),(0.32,0.15,0.65,True),(0.29,0.18,0.46,True)],
]
trach_data = [
    [(0.45,0.24,0.84,True),(0.32,0.19,0.54,True),(0.31,0.19,0.51,True),(0.33,0.19,0.55,True),(0.31,0.20,0.50,True)],
    [(0.46,0.23,0.93,True),(0.35,0.19,0.62,True),(0.32,0.18,0.55,True),(0.33,0.19,0.58,True),(0.31,0.18,0.52,True)],
    [(0.41,0.10,1.70,False),(0.23,0.06,0.87,True),(0.29,0.08,0.97,True),(0.31,0.09,1.02,False),(0.36,0.12,1.06,False)],
    [(0.28,0.11,0.71,True),(0.21,0.09,0.48,True),(0.22,0.11,0.45,True),(0.23,0.11,0.48,True),(0.23,0.12,0.45,True)],
    [(0.69,0.25,1.96,False),(0.40,0.19,0.87,True),(0.44,0.20,0.95,True),(0.47,0.20,1.08,False),(0.41,0.20,0.86,True)],
]

outcomes_data = [
    ('Dysphagia',       dysphagia_data),
    ('Gastrostomy Tube', gtube_data),
    ('Tracheostomy',    trach_data),
]

# ── Y-layout: sg=0 at top, tp=0 (6-mo) at top within each sg ─────────────────
ROW_H = 0.65   # vertical space between timepoints within a group
GAP   = 1.2    # extra vertical space between subgroups

def row_y(sg, tp):
    base = (N_SG - 1 - sg) * (N_TP * ROW_H + GAP)
    return base + (N_TP - 1 - tp) * ROW_H

y_min = row_y(N_SG - 1, N_TP - 1) - 0.7
y_max = row_y(0, 0) + 0.7
y_sg_center = {sg: (row_y(sg, 0) + row_y(sg, N_TP - 1)) / 2 for sg in range(N_SG)}

# ── Figure ────────────────────────────────────────────────────────────────────
fig_h = (y_max - y_min) * 0.42 + 1.5
fig, axes = plt.subplots(1, 3, figsize=(18, fig_h), sharey=True)
fig.patch.set_facecolor('white')
fig.subplots_adjust(wspace=0.08)

for ax, (outcome, sg_data) in zip(axes, outcomes_data):

    # Alternating group shading
    for sg in range(N_SG):
        y_lo = row_y(sg, N_TP - 1) - 0.5
        y_hi = row_y(sg, 0) + 0.5
        shade = '#F5F9FF' if sg % 2 == 0 else '#FFFFFF'
        ax.axhspan(y_lo, y_hi, color=shade, zorder=0)

    # Group separator lines & subgroup labels (left axis only)
    for sg in range(N_SG):
        if ax == axes[0]:
            ax.text(-0.28, y_sg_center[sg], SG_LABELS[sg],
                    ha='right', va='center', fontsize=8.5, fontweight='bold',
                    color='#1F4E79', transform=ax.get_yaxis_transform(),
                    linespacing=1.3)

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
            # CI line
            ax.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, alpha=0.85)
            # Marker: filled if sig, open if not
            fc = col if sig else 'white'
            ax.scatter(or_v, y, color=col, s=55, zorder=3,
                       marker=TP_MARKERS[tp], facecolors=fc,
                       edgecolors=col, linewidths=1.2)

    # Reference line
    ax.axvline(1.0, color='#888888', lw=1.0, linestyle='--', zorder=1)

    ax.set_xscale('log')
    ax.set_xlim(0.04, 20)
    ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([row_y(sg, tp) for sg in range(N_SG) for tp in range(N_TP)])
    ax.set_yticklabels(
        [TP_LABELS[tp] for sg in range(N_SG) for tp in range(N_TP)],
        fontsize=7, color='#444444')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Odds Ratio (log scale)', fontsize=9)
    ax.set_title(outcome, fontsize=11, fontweight='bold', color='#1F4E79', pad=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Favors arrows
    ax.text(0.38, -0.04, '\u2190 Favors TORS',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#555555', style='italic')
    ax.text(0.72, -0.04, 'Favors CT/CRT \u2192',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#555555', style='italic')

# ── Shared timepoint legend ───────────────────────────────────────────────────
legend_elements = [
    matplotlib.lines.Line2D([0], [0], marker=TP_MARKERS[i], color=TP_COLORS[i],
                             markerfacecolor=TP_COLORS[i], markersize=7,
                             label=TP_LABELS[i], lw=1.2)
    for i in range(N_TP)
]
legend_elements += [
    matplotlib.lines.Line2D([0], [0], marker='o', color='#666666',
                             markerfacecolor='white', markersize=7, lw=0,
                             label='p \u2265 0.05 (open)'),
    matplotlib.lines.Line2D([0], [0], marker='o', color='#666666',
                             markerfacecolor='#666666', markersize=7, lw=0,
                             label='p < 0.05 (filled)'),
]
fig.legend(handles=legend_elements, fontsize=8.5, loc='lower center',
           ncol=7, bbox_to_anchor=(0.5, -0.03), framealpha=0.9,
           title='Time from treatment', title_fontsize=8.5)

fig.suptitle(
    'Odds Ratios for Functional Outcomes — TORS vs CT/CRT  (PSM cohort, n=1,193)',
    fontsize=12, fontweight='bold', color='#1F4E79', y=1.01)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig(OUT_FORST, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_FORST}")

print("\nAll figures saved.")
