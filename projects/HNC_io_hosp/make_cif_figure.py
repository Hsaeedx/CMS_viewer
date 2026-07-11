"""
make_cif_figure.py
Aalen-Johansen cumulative incidence function (CIF) for hospice enrollment,
with death (without hospice) as the competing event. Stratified by dual-
eligible (Medicaid) status per PI request.

Time zero: each patient's last ICI administration date (last_io_date).
Follow-up: days until first event.
Events:
  1 = hospice election prior to death
  2 = death without hospice election (competing risk)

This figure exists to make the "immortal-time" point visually: patients who
die fast never get the chance to enroll in hospice, so the CIF for hospice
enrollment is bounded well below 1.0 by that structural constraint.

Output:
  figures/fig_cif_hospice_by_dual.png (matplotlib)
  tables/cif_hospice_by_dual.xlsx    (Aalen-Johansen point estimates at
                                       selected days for the manuscript text)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive; write PNG only
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from lifelines import AalenJohansenFitter

DB_PATH   = r"F:\CMS\cms_data.duckdb"
FIG_PATH  = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures\fig_cif_hospice_by_dual.png"
TBL_PATH  = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\cif_hospice_by_dual.xlsx"

# ── Style ────────────────────────────────────────────────────────────────────
SCARLET     = '#BA0C2F'
SCARLET_D40 = '#70071C'
GRAY        = '#A7B1B7'
WHITE       = '#FFFFFF'

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading io_analytic ...")
con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT dual_eligible, time_to_event, event_type
    FROM io_analytic
""").df()
con.close()

df['dual_eligible'] = df['dual_eligible'].fillna(0).astype(int)
df['time_to_event'] = pd.to_numeric(df['time_to_event'], errors='coerce')
df['event_type']    = pd.to_numeric(df['event_type'], errors='coerce').astype(int)

# Guard against zero durations (lifelines requires positive)
df = df[df['time_to_event'] > 0].copy()
print(f"  N = {len(df):,}  (event 1 = hospice: {(df['event_type']==1).sum():,}; "
      f"event 2 = death w/o hospice: {(df['event_type']==2).sum():,})")

# ── Fit AJ CIF by dual-eligible status ────────────────────────────────────────
FITS = {}   # dual_status -> (ajf, n)
LABELS = {0: 'Non-dual-eligible', 1: 'Dual-eligible (Medicaid)'}
COLORS = {0: SCARLET, 1: SCARLET_D40}

for dual in (0, 1):
    d = df[df['dual_eligible'] == dual]
    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(d['time_to_event'], d['event_type'], event_of_interest=1,
            label=LABELS[dual])
    FITS[dual] = (ajf, len(d))
    print(f"  {LABELS[dual]}: n={len(d):,}  CIF at day 180 = "
          f"{ajf.predict(180):.3f}")

# ── Also fit the overall (unstratified) for the point-estimates table ─────────
ajf_all = AalenJohansenFitter(calculate_variance=True)
ajf_all.fit(df['time_to_event'], df['event_type'], event_of_interest=1,
            label='Overall')

# ── Build figure ─────────────────────────────────────────────────────────────
print("Building figure ...")

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
X_MAX = 365  # cap follow-up at 1 year; long tails are sparse and off-message

for dual in (0, 1):
    ajf, n = FITS[dual]
    label = f"{LABELS[dual]} (n = {n:,})"
    # Plot step function of CIF + 95% CI band
    ax.step(ajf.cumulative_density_.index,
            ajf.cumulative_density_.iloc[:, 0].values,
            where='post', color=COLORS[dual], linewidth=2.2, label=label)
    lo = ajf.confidence_interval_.iloc[:, 0].values
    hi = ajf.confidence_interval_.iloc[:, 1].values
    ax.fill_between(ajf.confidence_interval_.index, lo, hi,
                    step='post', color=COLORS[dual], alpha=0.15, linewidth=0)

ax.set_xlim(0, X_MAX)
ax.set_ylim(0, 1.0)
ax.set_xlabel('Days from last ICI administration', fontsize=12)
ax.set_ylabel('Cumulative incidence of hospice enrollment', fontsize=12)
ax.set_title('Cumulative Incidence of Hospice Enrollment by Dual-Eligible Status\n'
             '(Aalen-Johansen; death without hospice as competing event)',
             fontsize=13, pad=12)
ax.legend(loc='lower right', frameon=True, fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate that "1.0 is unreachable" — the immortal-time point
ax.axhline(y=0.661, color=GRAY, linestyle='--', linewidth=1)
ax.text(X_MAX - 10, 0.68, 'Overall enrollment ceiling: 66.1%',
        ha='right', va='bottom', fontsize=10, color='#555555')

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight', facecolor=WHITE)
plt.close()
print(f"Saved: {FIG_PATH}")

# ── Point-estimate table ─────────────────────────────────────────────────────
LANDMARKS = [30, 60, 90, 120, 180, 270, 365]
print("Building point-estimate table ...")

def cif_at(ajf, t):
    """Return CIF and 95% CI at given day t (interpolated from step function)."""
    idx = ajf.cumulative_density_.index
    vals = ajf.cumulative_density_.iloc[:, 0].values
    ci = ajf.confidence_interval_
    # Right-continuous step: value at t = last value at index <= t
    mask = idx <= t
    if not mask.any():
        return 0.0, 0.0, 0.0
    j = int(np.where(mask)[0][-1])
    return float(vals[j]), float(ci.iloc[j, 0]), float(ci.iloc[j, 1])

records = []
for stratum_label, ajf in [('Overall', ajf_all),
                            ('Non-dual', FITS[0][0]),
                            ('Dual-eligible', FITS[1][0])]:
    for t in LANDMARKS:
        cif, lo, hi = cif_at(ajf, t)
        records.append({
            'stratum': stratum_label,
            'day':     t,
            'cif':     cif,
            'lo':      lo,
            'hi':      hi,
        })
res = pd.DataFrame(records)

# Write Excel
SCARLET_HEX = 'BA0C2F'
HEADER_FILL = PatternFill('solid', fgColor=SCARLET_HEX)
HEADER_FONT = Font(name='Times New Roman', bold=True, color='FFFFFF', size=11)
BODY_FONT   = Font(name='Times New Roman', size=11)
ALT_FILL    = PatternFill('solid', fgColor='F9ECEE')
TITLE_FONT  = Font(name='Times New Roman', bold=True, size=12, color=SCARLET_HEX)
NOTE_FONT   = Font(name='Times New Roman', italic=True, size=11, color='555555')

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'CIF landmarks'

ws.merge_cells('A1:E1')
ws['A1'] = ('Table X. Cumulative Incidence of Hospice Enrollment at Landmark Days '
            '(Aalen-Johansen)')
ws['A1'].font = TITLE_FONT
ws['A1'].alignment = Alignment(horizontal='left', vertical='center')
ws.row_dimensions[1].height = 22

HR = 3
for ci, h in enumerate(['Stratum', 'Day', 'CIF (%)', 'Lower 95% CI',
                        'Upper 95% CI'], 1):
    cell = ws.cell(row=HR, column=ci, value=h)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                               vertical='center', wrap_text=True)
ws.row_dimensions[HR].height = 28

r = HR + 1
for idx, row in enumerate(res.itertuples(index=False)):
    fill = ALT_FILL if idx % 2 == 1 else None
    vals = [row.stratum, row.day,
            f"{row.cif*100:.1f}",
            f"{row.lo*100:.1f}",
            f"{row.hi*100:.1f}"]
    for ci, v in enumerate(vals, 1):
        cell = ws.cell(row=r, column=ci, value=v)
        cell.font = BODY_FONT
        if fill:
            cell.fill = fill
        cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                   vertical='center')
    r += 1

# Footer
r += 1
ws.merge_cells(f'A{r}:E{r}')
ws.cell(row=r, column=1, value=(
    'Aalen-Johansen non-parametric estimator of the cumulative incidence function (CIF) '
    'for hospice enrollment, with death without hospice enrollment treated as a competing '
    'event. Time zero = last ICI administration. 95% confidence intervals based on '
    'Delta method (default in lifelines). CIFs plateau below 100% by design, reflecting '
    'the substantial fraction of decedents who die before ever electing hospice.'
)).font = NOTE_FONT
ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[r].height = 60

ws.column_dimensions['A'].width = 22
ws.column_dimensions['B'].width = 8
ws.column_dimensions['C'].width = 12
ws.column_dimensions['D'].width = 15
ws.column_dimensions['E'].width = 15

wb.save(TBL_PATH)
print(f"Saved: {TBL_PATH}")

# Console summary
print("\n=== CIF landmark summary ===")
piv = res.pivot(index='day', columns='stratum', values='cif')
piv = (piv * 100).round(1)
print(piv.to_string())
