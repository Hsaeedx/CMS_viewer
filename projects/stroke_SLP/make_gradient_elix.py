"""
make_gradient_elix.py

1. Supp_Figure1.png — HR comparison: Early (8-35d) vs Late (36-90d ref) across 3 outcomes.
   Single PSM comparison, all three primary outcomes.

2. Supp_Table1.xlsx — TV Cox HRs stratified by van Walraven quartile.
   Within each quartile, run TV Cox on the PSM-matched comparison A cohort.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

_out_dir      = Path(os.getenv("project_paths", ".")) / "stroke_SLP" / "output_files"
_out_dir.mkdir(parents=True, exist_ok=True)
DB_PATH       = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
GRADIENT_PATH = _out_dir / "Supp_Figure1.png"
ELIX_PATH     = _out_dir / "Supp_Table1.xlsx"

MAX_FOLLOW = 365
TV_COVARIATES = ['age_at_adm', 'van_walraven_score', 'index_los']

COMPARISONS = [
    ('A', 'Early', 'Late', 'psm_matched_A'),
]

# (label, event_col, competing_col, exclude_peg)
OUTCOMES = [
    ('Aspiration-related PNA', 'days_to_asp_related', 'days_to_death', False),
    ('PEG/G-tube',             'days_to_gtube',        'days_to_death', True),
    ('Mortality',              'days_to_death',         None,            False),
]

# ── Load & filter ─────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_all = con.execute("""
    SELECT p.DSYSRTKY, p.slp_timing_group, p.days_to_slp_outpt,
           p.age_at_adm, p.index_los, p.van_walraven_score,
           p.peg_placed, p.psm_matched_A,
           o.days_to_death, o.days_to_pneumonia, o.first_pneumonia_code,
           o.days_to_gtube, o.pre_stroke_tube
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()
print(f"  Loaded {len(df_all):,} rows")

# J18 + J69 composite → aspiration-related PNA
import numpy as _np
df_all['days_to_asp_related'] = _np.where(
    df_all['first_pneumonia_code'].isin(['J18', 'J69']),
    df_all['days_to_pneumonia'], _np.nan
)


# ── TV Cox helpers ─────────────────────────────────────────────────────────────
def build_tv_df(df, event_col, competing_col, treat_grp):
    records = []
    for _, row in df.iterrows():
        slp_day  = float(row['days_to_slp_outpt'])
        ev_day   = row[event_col]
        comp_day = (row[competing_col]
                    if competing_col and pd.notna(row[competing_col]) else np.nan)

        candidates = [float(MAX_FOLLOW)]
        if pd.notna(ev_day):   candidates.append(float(ev_day))
        if pd.notna(comp_day): candidates.append(float(comp_day))
        end_time = min(candidates)

        final_event = int(
            pd.notna(ev_day)
            and float(ev_day) <= MAX_FOLLOW
            and float(ev_day) == end_time
        )
        group_flag = 1 if row['slp_timing_group'] == treat_grp else 0
        base = {col: float(row[col]) if pd.notna(row[col]) else 0.0
                for col in TV_COVARIATES}

        if end_time <= slp_day:
            records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                            'stop': max(end_time, 0.5),
                            'trt': 0, 'event': final_event, **base})
        else:
            records.append({'id': row['DSYSRTKY'], 'start': 0.0,
                            'stop': slp_day, 'trt': 0, 'event': 0, **base})
            records.append({'id': row['DSYSRTKY'], 'start': slp_day,
                            'stop': max(end_time, slp_day + 0.5),
                            'trt': group_flag, 'event': final_event, **base})
    return pd.DataFrame(records)


def run_tv_cox(df, event_col, competing_col, treat_grp):
    tv = build_tv_df(df, event_col, competing_col, treat_grp)
    tv = tv.dropna(subset=TV_COVARIATES)
    # Standardize continuous covariates to prevent exp overflow
    for col in ['age_at_adm', 'van_walraven_score', 'index_los']:
        if col in tv.columns:
            sd = tv[col].std()
            if sd > 0:
                tv[col] = (tv[col] - tv[col].mean()) / sd
    n_pts  = tv['id'].nunique()
    n_evts = int(tv['event'].sum())
    if n_evts < 10:
        return None, n_pts, n_evts
    try:
        ctv = CoxTimeVaryingFitter()
        ctv.fit(tv, id_col='id', start_col='start', stop_col='stop',
                event_col='event', show_progress=False)
        r    = ctv.summary.loc['trt']
        hr   = np.exp(r['coef'])
        lo95 = np.exp(r['coef lower 95%'])
        hi95 = np.exp(r['coef upper 95%'])
        p    = r['p']
        return (hr, lo95, hi95, p), n_pts, n_evts
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, n_pts, n_evts


# ── Run all comparisons × outcomes ────────────────────────────────────────────
print("\nRunning TV Cox models...")
main_results = {}   # (comp_label, out_label) -> (hr, lo, hi, p) or None

for comp_label, treat_grp, ctrl_grp, match_col in COMPARISONS:
    df_comp = df_all[df_all[match_col] == True].copy()
    print(f"\n  Comparison {comp_label}: {treat_grp} vs {ctrl_grp} (n={len(df_comp):,})")
    for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
        sub = df_comp.copy()
        if excl_peg:
            sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
        res, n, n_ev = run_tv_cox(sub, ev_col, comp_col, treat_grp)
        main_results[(comp_label, out_label)] = res
        tag = (f"HR={res[0]:.2f} [{res[1]:.2f}\u2013{res[2]:.2f}]"
               if res else "failed")
        print(f"    {out_label}: n={n:,} ev={n_ev:,}  {tag}")


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY RESULTS FIGURE (Supp_Figure1) — single comparison, 3 outcomes
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding primary results figure (Supp_Figure1)...")

OUTCOME_COLORS = {
    'Aspiration-related PNA': '#ba0c2f',
    'PEG/G-tube':             '#70071c',
    'Mortality':              '#a7b1b7',
}
OUTCOME_MARKERS = {
    'Aspiration-related PNA': 'o',
    'PEG/G-tube':             's',
    'Mortality':              '^',
}

# Median days per group in matched cohort
df_a = df_all[df_all['psm_matched_A'] == True].copy()
med_days = df_a.groupby('slp_timing_group')['days_to_slp_outpt'].median()
X_DAYS = {
    'Early': float(med_days.get('Early', 21)),
    'Late':  float(med_days.get('Late',  52)),
}
print(f"  Median days: {X_DAYS}")

fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
ax.set_facecolor('white')
ax.axhline(1.0, color='#888888', lw=1.2, linestyle='--', zorder=1)
ax.axhspan(0.95, 1.05, color='#EEEEEE', alpha=0.5, zorder=0)

x_treat = X_DAYS['Early']
x_ref   = X_DAYS['Late']

for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
    col = OUTCOME_COLORS[out_label]
    mkr = OUTCOME_MARKERS[out_label]
    res = main_results.get(('A', out_label))

    # connecting line
    hr_ref = 1.0
    hr_trt = res[0] if res else np.nan
    if not np.isnan(hr_trt):
        ax.plot([x_treat, x_ref], [hr_trt, hr_ref], color=col, lw=2.0, alpha=0.85, zorder=3)
        # CI whiskers
        lo, hi = res[1], res[2]
        ax.plot([x_treat, x_treat], [lo, hi], color=col, lw=1.4, zorder=3)
        ax.plot([x_treat - 0.8, x_treat + 0.8], [lo, lo], color=col, lw=1.0, zorder=3)
        ax.plot([x_treat - 0.8, x_treat + 0.8], [hi, hi], color=col, lw=1.0, zorder=3)
        ax.plot(x_treat, hr_trt, marker=mkr, color=col, markersize=10,
                markeredgecolor='white', markeredgewidth=0.8, zorder=4)
    # Reference: open marker
    ax.plot(x_ref, hr_ref, marker=mkr, color='white', markersize=10,
            markeredgecolor=col, markeredgewidth=2.0, zorder=4)
    ax.text(x_ref + 1.5, hr_ref, out_label,
            va='center', ha='left', fontsize=9, color=col, fontweight='bold')

ax.set_xlim(0, 75)
ax.set_xticks([x_treat, x_ref])
ax.set_xticklabels([
    f"Early SLP\n(Weeks 1\u20134)",
    f"Late SLP\n(Week 5+)\nREFERENCE",
], fontsize=10)
ax.set_xlabel('Timing of First Outpatient SLP Visit (weeks post-discharge)', fontsize=11, labelpad=8)

ax.set_yscale('log')
ax.set_ylim(0.50, 2.5)
ax.set_yticks([0.5, 0.7, 1.0, 1.25, 1.5, 2.0])
ax.set_yticklabels(['0.50', '0.70', '1.00', '1.25', '1.50', '2.00'], fontsize=10)
ax.set_ylabel('Adjusted Hazard Ratio (log scale)\nReference: Late SLP, Week 5+ (PSM)', fontsize=10.5)

ax.text(x_ref, 1.03, 'HR = 1.00', ha='center', va='bottom',
        fontsize=8.5, color='#555555', fontstyle='italic')

n_pairs = int(df_all['psm_matched_A'].sum()) // 2
ax.set_title(
    'Early SLP (Weeks 1\u20134) vs Late SLP (Week 5+): Time-Varying Cox HRs by Outcome\n'
    f'PSM-matched cohort  \u2022  {n_pairs:,} pairs  \u2022  Max follow-up 365 days',
    fontsize=11, fontweight='bold', color='#70071c', pad=10)

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.spines['left'].set_color('#AAAAAA')
ax.spines['bottom'].set_color('#AAAAAA')
ax.tick_params(axis='both', color='#AAAAAA')

plt.tight_layout()
plt.savefig(str(GRADIENT_PATH), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {GRADIENT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# ELIXHAUSER STRATIFIED RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Elixhauser stratified results...")

# Quartile boundaries from the full matched population (use comp A)
df_a = df_all[df_all['psm_matched_A'] == True].copy()
vw = df_a['van_walraven_score']
_, bins = pd.qcut(vw, q=4, duplicates='drop', retbins=True, labels=False)
n_bins = len(bins) - 1
ql_labels = []
for i in range(n_bins):
    lo_b, hi_b = bins[i], bins[i + 1]
    if i == 0:
        ql_labels.append(f'Q{i+1}  (VW <= {hi_b:.0f})  -- Lowest comorbidity')
    elif i == n_bins - 1:
        ql_labels.append(f'Q{i+1}  (VW > {lo_b:.0f})  -- Highest comorbidity')
    else:
        ql_labels.append(f'Q{i+1}  (VW {lo_b:.0f}-{hi_b:.0f})')
q1, q2, q3 = vw.quantile([0.25, 0.50, 0.75])

# Assign quartile labels using same bin boundaries on both matched cohorts
def assign_quartile(series):
    return pd.cut(series, bins=bins, labels=False, include_lowest=True).map(
        {i: lab for i, lab in enumerate(ql_labels)})

elix_rows = []
for comp_label, treat_grp, ctrl_grp, match_col in COMPARISONS:
    df_comp = df_all[df_all[match_col] == True].copy()
    df_comp['vw_q'] = assign_quartile(df_comp['van_walraven_score'])

    for qlab in ql_labels:
        q_sub = df_comp[df_comp['vw_q'] == qlab]
        n_q   = len(q_sub)
        n_grp = {g: int((q_sub['slp_timing_group'] == g).sum())
                 for g in ['Early', 'Late']}
        print(f"  {comp_label} {qlab}: N={n_q:,}  {n_grp}")

        for out_label, ev_col, comp_col, excl_peg in OUTCOMES:
            sub = q_sub.copy()
            if excl_peg:
                sub = sub[(sub['peg_placed'] == 0) & (sub['pre_stroke_tube'].fillna(0) == 0)]
            res, n, n_ev = run_tv_cox(sub, ev_col, comp_col, treat_grp)

            base = {
                'Comparison':        f"{treat_grp} vs {ctrl_grp} (ref)",
                'Elixhauser Quartile': qlab,
                'N in quartile':     f"{n_q:,}",
                'N (treat)':         f"{n_grp.get(treat_grp, 0):,}",
                'N (ref)':           f"{n_grp.get(ctrl_grp, 0):,}",
                'Outcome':           out_label,
                'N in model':        f"{n:,}",
                'Events':            f"{n_ev:,}",
            }
            if res is None:
                elix_rows.append({**base, 'HR (95% CI)': 'Too few events', 'p-value': ''})
            else:
                hr, lo, hi, p = res
                p_str = '<0.0001' if p < 0.0001 else f'{p:.4f}'
                elix_rows.append({**base,
                    'HR (95% CI)': f"{hr:.2f}  [{lo:.2f}\u2013{hi:.2f}]",
                    'p-value':     p_str,
                })

df_elix = pd.DataFrame(elix_rows)

# ── Write Excel ────────────────────────────────────────────────────────────────
HEADER_FILL = PatternFill('solid', fgColor='70071c')
HEADER_FONT = Font(bold=True, color='FFFFFF', size=10)
COMP_FILL   = PatternFill('solid', fgColor='ba0c2f')
COMP_FONT   = Font(bold=True, color='FFFFFF', size=10)
QHDR_FILL   = PatternFill('solid', fgColor='e0c4ca')
QHDR_FONT   = Font(bold=True, size=10, color='70071c')
OUT_FILL    = PatternFill('solid', fgColor='fdf5f6')
OUT_FONT    = Font(bold=True, size=10, color='ba0c2f')
ALT_FILL    = PatternFill('solid', fgColor='f5f5f5')
TITLE_FONT  = Font(bold=True, size=12, color='70071c')
SIG_FONT    = Font(bold=True, size=10, color='ba0c2f')

DISPLAY_COLS = ['Comparison', 'Elixhauser Quartile', 'Outcome',
                'N in quartile', 'N (treat)', 'N (ref)',
                'N in model', 'Events', 'HR (95% CI)', 'p-value']

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Elix_Stratified'

ws.append(['Table: TV Cox Results Stratified by Elixhauser Comorbidity (van Walraven Score)'])
ws['A1'].font = TITLE_FONT
ws.append([f'Reference: SLP 31\u201390d  \u2022  '
           f'Quartile cutpoints: Q1\u2264{q1:.0f}, Q2\u2264{q2:.0f}, Q3\u2264{q3:.0f}  \u2022  '
           f'PSM-matched cohort  \u2022  Max follow-up 365 days'])
ws['A2'].font = Font(italic=True, size=10, color='555555')
ws.append([])

hdr_row = ws.max_row + 1
for ci, col in enumerate(DISPLAY_COLS, 1):
    cell = ws.cell(row=hdr_row, column=ci, value=col)
    cell.font      = HEADER_FONT
    cell.fill      = HEADER_FILL
    cell.alignment = Alignment(horizontal='center', wrap_text=True)
ws.row_dimensions[hdr_row].height = 30

prev_comp = None
prev_q    = None
prev_out  = None
alt = 0

for _, row in df_elix.iterrows():
    comp = row['Comparison']
    q    = row['Elixhauser Quartile']
    out  = row['Outcome']

    if comp != prev_comp:
        ri = ws.max_row + 1
        ws.row_dimensions[ri].height = 20
        for ci, col in enumerate(DISPLAY_COLS, 1):
            cell = ws.cell(row=ri, column=ci, value=comp if ci == 1 else '')
            cell.font      = COMP_FONT
            cell.fill      = COMP_FILL
            cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                       vertical='center')
        prev_comp = comp
        prev_q    = None
        prev_out  = None
        alt = 0

    if q != prev_q:
        ri = ws.max_row + 1
        ws.row_dimensions[ri].height = 18
        for ci, col in enumerate(DISPLAY_COLS, 1):
            val = q if ci == 2 else (row.get(col, '') if col in
                  ('N in quartile', 'N (treat)', 'N (ref)') else '')
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font      = QHDR_FONT
            cell.fill      = QHDR_FILL
            cell.alignment = Alignment(horizontal='left' if ci <= 2 else 'center',
                                       vertical='center')
        prev_q   = q
        prev_out = None
        alt = 0

    if out != prev_out:
        ri = ws.max_row + 1
        ws.row_dimensions[ri].height = 16
        for ci, col in enumerate(DISPLAY_COLS, 1):
            val = out if ci == 3 else ''
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font      = OUT_FONT
            cell.fill      = OUT_FILL
            cell.alignment = Alignment(horizontal='left', vertical='center')
        prev_out = out
        alt = 0

    alt += 1
    ri = ws.max_row + 1
    ws.row_dimensions[ri].height = 16
    for ci, col in enumerate(DISPLAY_COLS, 1):
        skip_cols = ('Comparison', 'Elixhauser Quartile',
                     'N in quartile', 'N (treat)', 'N (ref)', 'Outcome')
        val = '' if col in skip_cols else row.get(col, '')
        cell = ws.cell(row=ri, column=ci, value=val)
        cell.fill      = ALT_FILL if alt % 2 == 0 else PatternFill()
        cell.alignment = Alignment(horizontal='left', vertical='center')
        if ci == 9 and str(val) not in ('Too few events', ''):
            try:
                p_val = float(row['p-value'].replace('<', '')) if '<' in str(row['p-value']) else float(row['p-value'])
                if (p_val < 0.05 and '<' in str(row['p-value'])) or (str(row['p-value']) != '' and p_val < 0.05):
                    cell.font = SIG_FONT
            except Exception:
                pass

for ci, width in zip(range(1, len(DISPLAY_COLS) + 1),
                     [32, 32, 18, 13, 10, 10, 12, 10, 24, 12]):
    ws.column_dimensions[ws.cell(1, ci).column_letter].width = width

wb.save(str(ELIX_PATH))
print(f"Saved: {ELIX_PATH}")
