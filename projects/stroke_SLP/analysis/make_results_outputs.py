"""
make_results_outputs.py

Generates:
  1. Figure2.png  — primary outcomes (Aspiration PNA, PEG/G-tube, Mortality), 600 DPI
  2. Table3.xlsx  — all outcomes, single PSM comparison (Early 8-35d vs Late 36-90d)

Uses PSM-matched cohort with time-varying Cox PH (CoxTimeVaryingFitter).
Reference group: Late (36-90d).
Person-time split at days_to_slp_outpt; immortal time handled correctly.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from stroke_slp_common import OUT_DIR, add_aspiration_related_pna, connect, run_tv_cox

FOREST_PATH  = OUT_DIR / "Figure2.png"
RESULTS_PATH = OUT_DIR / "Table3.xlsx"

COMPARISONS = [
    ('A', 'Early', 'Late', 'psm_matched_A'),
]

# (label, event_col, competing_col, exclude_peg)
OUTCOMES = [
    ('Aspiration-related PNA', 'days_to_asp_related', 'days_to_death', False),
    ('PEG/G-tube',     'days_to_gtube',      'days_to_death', True),
    ('Mortality',      'days_to_death',       None,            False),
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
con = connect(read_only=True)
df_all = con.execute("""
    SELECT
        p.DSYSRTKY, p.slp_timing_group, p.days_to_slp_outpt,
        p.age_at_adm, p.index_los, p.van_walraven_score,
        p.peg_placed, p.psm_matched_A,
        o.days_to_death,
        o.days_to_gtube, o.pre_stroke_tube,
        o.days_to_pneumonia, o.first_pneumonia_code
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()
df_all = add_aspiration_related_pna(df_all)
print(f"  Loaded {len(df_all):,} rows")


# ── Time-varying Cox helpers ───────────────────────────────────────────────────
# ── Run models ─────────────────────────────────────────────────────────────────
results = []
for comp_label, treat_grp, ctrl_grp, match_col in COMPARISONS:
    df_comp = df_all[df_all[match_col] == True].copy()
    print(f"\nComparison {comp_label}: {treat_grp} vs {ctrl_grp} ref  (n={len(df_comp):,})")
    for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
        sub = df_comp.copy()
        if excl_peg:
            sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
        res, n, n_ev = run_tv_cox(sub, ev_col, comp_col, treat_grp)
        entry = {
            'Comparison': f"{treat_grp} vs {ctrl_grp} (ref)",
            'comp_label': comp_label,
            'treat_grp':  treat_grp,
            'ctrl_grp':   ctrl_grp,
            'Outcome':    out_label,
            'N':          n,
            'Events':     n_ev,
            'Event_pct':  round(100 * n_ev / n, 1) if n > 0 else 0,
        }
        if res:
            hr, lo, hi, p = res
            entry.update({'HR': round(hr, 2), 'CI_low': round(lo, 2), 'CI_high': round(hi, 2),
                          'p_raw': p, 'p_str': '<0.0001' if p < 0.0001 else f'{p:.4f}'})
        else:
            entry.update({'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan,
                          'p_raw': np.nan, 'p_str': 'N/A'})
        results.append(entry)
        tag = (f"HR={hr:.2f} [{lo:.2f}\u2013{hi:.2f}] p={entry['p_str']}"
               if res else "failed")
        print(f"  {out_label}: n={n:,} ev={n_ev:,}  {tag}")

df_res = pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# FOREST PLOT — primary outcomes: Aspiration PNA, PEG/G-tube, Mortality
# Single comparison: Early (8-35d) vs Late (36-90d reference)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding forest plot...")

C_SCARLET = '#ba0c2f'
C_DARK    = '#70071c'

# Layout: 3 outcome bands, one data row each
ROW_Y = {
    'asp_hdr':                8.6,
    'Aspiration-related PNA': 7.6,
    'gtube_hdr':     6.1,
    'PEG/G-tube':    5.1,
    'mort_hdr':      3.6,
    'Mortality':     2.6,
}
YLIM = (1.8, 9.5)

fig = plt.figure(figsize=(14, 6), facecolor='white')
gs  = gridspec.GridSpec(1, 3, width_ratios=[3.2, 5.0, 3.5],
                        wspace=0.0, left=0.01, right=0.99,
                        top=0.88, bottom=0.12)

ax_L = fig.add_subplot(gs[0])
ax_M = fig.add_subplot(gs[1])
ax_R = fig.add_subplot(gs[2])

for ax in [ax_L, ax_M, ax_R]:
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    ax.set_facecolor('white')
for ax in [ax_L, ax_R]:
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# Alternating band shading
for ax in [ax_L, ax_M, ax_R]:
    ax.axhspan(6.8, 9.1, color='#fdf5f6', alpha=0.8, zorder=0)
    ax.axhspan(4.3, 5.8, color='#f9eaeb', alpha=0.8, zorder=0)
    ax.axhspan(1.9, 3.3, color='#fdf5f6', alpha=0.8, zorder=0)

HDR_KW = dict(fontsize=9, fontweight='bold', color='#555555', va='bottom')
ax_L.text(0.97, YLIM[1] + 0.02, 'Outcome',
          ha='right', transform=ax_L.get_yaxis_transform(), **HDR_KW)
ax_M.axvline(1.0, color='#888888', lw=1.2, linestyle='--', zorder=1)
ax_R.text(0.03, YLIM[1] + 0.02, 'HR [95% CI]              p-value',
          ha='left', transform=ax_R.get_yaxis_transform(), **HDR_KW)


def sec_header(y, text):
    for ax in [ax_L, ax_M, ax_R]:
        ax.axhline(y - 0.3, color='#CCCCCC', lw=0.8, zorder=0)
    ax_L.text(0.97, y, text, ha='right', va='center', fontsize=10.5,
              fontweight='bold', color='#70071c',
              transform=ax_L.get_yaxis_transform())


sec_header(ROW_Y['asp_hdr'],   'Aspiration Pneumonia')
sec_header(ROW_Y['gtube_hdr'], 'PEG / G-tube Placement')
sec_header(ROW_Y['mort_hdr'],  'All-Cause Mortality')

for _, row in df_res.iterrows():
    out = row['Outcome']
    if out not in ROW_Y or pd.isna(row['HR']):
        continue
    ypos = ROW_Y[out]
    hr, lo, hi = row['HR'], row['CI_low'], row['CI_high']

    ax_M.plot([lo, hi], [ypos, ypos], color=C_SCARLET, lw=2.0,
              solid_capstyle='round', zorder=3)
    ax_M.plot(hr, ypos, marker='o', color=C_SCARLET, markersize=10,
              markeredgecolor='white', markeredgewidth=1.0, zorder=4)
    p_fmt = row['p_str'] if row['p_str'].startswith('<') else f"= {row['p_str']}"
    ax_R.text(0.03, ypos,
              f"  {hr:.2f}  [{lo:.2f}\u2013{hi:.2f}]     p {p_fmt}",
              ha='left', va='center', fontsize=9, color='#70071c',
              transform=ax_R.get_yaxis_transform())

ax_M.set_xscale('log')
ax_M.set_xlim(0.45, 2.8)
ax_M.set_xticks([0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5])
ax_M.set_xticklabels(['0.50', '0.70', '1.00', '1.25', '1.50', '2.00', '2.50'], fontsize=8.5)
ax_M.set_xlabel('Hazard Ratio (log scale)\nReference group: Late SLP (Week 5+)',
                fontsize=9.5, color='#333333')
for sp in ['top', 'left', 'right']:
    ax_M.spines[sp].set_visible(False)
ax_M.spines['bottom'].set_color('#AAAAAA')
ax_M.tick_params(axis='x', length=3, color='#AAAAAA')

n_pts = int(df_all['psm_matched_A'].sum())
fig.suptitle(
    'Time-Varying Cox HR: SLP Timing and Primary Post-Stroke Outcomes\n'
    f'PSM-matched cohort (Early Wks 1\u20134 vs Late Wk 5+)  \u2022  n\u2248{n_pts//2:,} pairs  '
    '\u2022  Max follow-up 365 days',
    fontsize=11, fontweight='bold', color='#70071c', y=0.99, va='top'
)

plt.savefig(str(FOREST_PATH), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {FOREST_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding results table Excel...")

HEADER_FILL  = PatternFill('solid', fgColor='70071c')
HEADER_FONT  = Font(bold=True, color='FFFFFF', size=10)
SECTION_FILL = PatternFill('solid', fgColor='e0c4ca')
SECTION_FONT = Font(bold=True, size=10, color='70071c')
COMP_FILL    = PatternFill('solid', fgColor='ba0c2f')
COMP_FONT    = Font(bold=True, color='FFFFFF', size=10)
ALT_FILL     = PatternFill('solid', fgColor='fdf5f6')
TITLE_FONT   = Font(bold=True, size=12, color='70071c')
SIG_FONT     = Font(bold=True, size=10, color='ba0c2f')
THIN         = Side(style='thin', color='CCCCCC')
THIN_BORDER  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

OUTCOME_NOTES = {
    'Aspiration-related PNA': 'ICD-10 J18 (unspecified) + J69 (aspiration); death treated as competing event',
    'PEG/G-tube':     'Excludes index PEG and pre-stroke tube; death as competing event',
    'Mortality':      'All-cause death from MBSF; max follow-up 365 days',
}

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'TV_Cox_Results'

ws.append(['Table: Time-Varying Cox PH \u2014 SLP Timing and Post-Stroke Outcomes'])
ws['A1'].font = TITLE_FONT
ws.append(['PSM-matched Medicare stroke patients  \u2022  Reference: Late SLP (Week 5+)  '
           '\u2022  Person-time split at days_to_slp_outpt'])
ws['A2'].font = Font(italic=True, size=10, color='555555')
ws.append([])

cols = ['Comparison', 'Outcome', 'N', 'Events', 'Event %',
        'HR (95% CI)', 'p-value', 'Note']

hdr_row = ws.max_row + 1
for ci, col in enumerate(cols, 1):
    cell = ws.cell(row=hdr_row, column=ci, value=col)
    cell.font      = HEADER_FONT
    cell.fill      = HEADER_FILL
    cell.alignment = Alignment(horizontal='center', wrap_text=True)
ws.row_dimensions[hdr_row].height = 28

prev_comp = None
alt = 0
for _, row in df_res.iterrows():
    comp = row['Comparison']
    if comp != prev_comp:
        ri = ws.max_row + 1
        for ci in range(1, len(cols) + 1):
            cell = ws.cell(row=ri, column=ci, value=comp if ci == 1 else '')
            cell.font      = COMP_FONT
            cell.fill      = COMP_FILL
            cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                       vertical='center')
        prev_comp = comp
        alt = 0

    alt += 1
    ri = ws.max_row + 1
    ws.row_dimensions[ri].height = 16
    hr_str = (f"{row['HR']:.2f}  [{row['CI_low']:.2f}\u2013{row['CI_high']:.2f}]"
              if pd.notna(row['HR']) else 'N/A')
    row_vals = [
        '', row['Outcome'],
        f"{row['N']:,}", f"{row['Events']:,}", f"{row['Event_pct']:.1f}%",
        hr_str, row['p_str'],
        OUTCOME_NOTES.get(row['Outcome'], ''),
    ]
    for ci, val in enumerate(row_vals, 1):
        cell = ws.cell(row=ri, column=ci, value=val)
        cell.fill      = ALT_FILL if alt % 2 == 0 else PatternFill()
        cell.alignment = Alignment(horizontal='left', vertical='center')
        cell.border    = THIN_BORDER
        if ci == 6 and pd.notna(row['p_raw']) and row['p_raw'] < 0.05:
            cell.font = SIG_FONT

ws.column_dimensions['A'].width = 30
ws.column_dimensions['B'].width = 22
ws.column_dimensions['C'].width = 10
ws.column_dimensions['D'].width = 10
ws.column_dimensions['E'].width = 10
ws.column_dimensions['F'].width = 24
ws.column_dimensions['G'].width = 12
ws.column_dimensions['H'].width = 45

wb.save(str(RESULTS_PATH))
print(f"Saved: {RESULTS_PATH}")
