"""
make_table2.py
Hospice utilization and end-of-life care patterns
Columns: Outcome | Value
Denominators made explicit in section headers.
Output: C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/tables/table2.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\table2.xlsx"

print("Loading io_analytic...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        hospice_enrolled, hospice_los_days, hospice_short_stay,
        days_last_io_to_hospice, days_last_io_to_death,
        days_last_io_to_death_cat,
        in_hospital_death
    FROM io_analytic
""").df()
con.close()

df['hospice_enrolled']  = df['hospice_enrolled'].fillna(0).astype(int)
df['in_hospital_death'] = df['in_hospital_death'].fillna(0).astype(int)
df['hospice_short_stay']= df['hospice_short_stay'].fillna(0).astype(int)

hosp   = df[df['hospice_enrolled'] == 1]
nohosp = df[df['hospice_enrolled'] == 0]
N      = len(df)
Nh     = len(hosp)
Nnh    = len(nohosp)

print(f"  Total={N:,}  Hospice={Nh:,} ({100*Nh/N:.1f}%)  No hospice={Nnh:,}")

# ── Format helpers ─────────────────────────────────────────────────────────────
def n_pct(series, val=1):
    n = (series == val).sum()
    p = 100.0 * n / len(series)
    return f"{n:,} ({p:.1f}%)"

def med_iqr(series):
    s = series.dropna()
    return f"{s.median():.0f} ({s.quantile(0.25):.0f}–{s.quantile(0.75):.0f})"

# ── Build rows ────────────────────────────────────────────────────────────────
rows = []

def section(title):
    rows.append({'label': title, 'is_section': True, 'val': ''})

def add(label, val, indent=False):
    rows.append({
        'label':      ('    ' + label) if indent else label,
        'is_section': False,
        'val':        val,
    })

# ── PRIMARY OUTCOME ────────────────────────────────────────────────────────────
section(f'PRIMARY OUTCOME  (N = {N:,})')
add('Hospice enrolled', n_pct(df['hospice_enrolled']))

# ── HOSPICE UTILIZATION ────────────────────────────────────────────────────────
section(f'HOSPICE UTILIZATION  (among enrolled, n = {Nh:,})')
add('Hospice LOS, median days (IQR)', med_iqr(hosp['hospice_los_days']))
add('Short stay ≤7 days', n_pct(hosp['hospice_short_stay']))
add('Days from last ICI dose to hospice enrollment, median (IQR)',
    med_iqr(hosp['days_last_io_to_hospice']))

# ── ICI TIMING ─────────────────────────────────────────────────────────────────
section(f'ICI TIMING TO DEATH  (N = {N:,})')
add('Days from last ICI dose to death, median (IQR)',
    med_iqr(df['days_last_io_to_death']))
for cat, label in [
    ('<=3 days',   '≤3 days'),
    ('4-14 days',  '4–14 days'),
    ('15-30 days', '15–30 days'),
    ('31-90 days', '31–90 days'),
    ('>90 days',   '>90 days'),
]:
    n   = (df['days_last_io_to_death_cat'] == cat).sum()
    p   = 100.0 * n / N
    add(label, f"{n:,} ({p:.1f}%)", indent=True)

# ── SECONDARY OUTCOMES ────────────────────────────────────────────────────────
section(f'SECONDARY OUTCOMES  (N = {N:,})')
add('In-hospital death', n_pct(df['in_hospital_death']))
add('In-hospital death among hospice enrollees',
    f"{n_pct(hosp['in_hospital_death'])}  (of n = {Nh:,})")
add('In-hospital death among non-enrollees',
    f"{n_pct(nohosp['in_hospital_death'])}  (of n = {Nnh:,})")

# ── Write Excel ───────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")

HEADER_FILL  = PatternFill('solid', fgColor='BA0C2F')
HEADER_FONT  = Font(name='Times New Roman', bold=True, color='FFFFFF', size=11)
BODY_FONT    = Font(name='Times New Roman', size=11)
SECTION_FILL = PatternFill('solid', fgColor='F5D0D6')
SECTION_FONT = Font(name='Times New Roman', bold=True, size=11, color='7A0820')
ALT_FILL     = PatternFill('solid', fgColor='F9ECEE')
TITLE_FONT   = Font(name='Times New Roman', bold=True, size=12, color='BA0C2F')

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Table 2'

ws.append(['Table 2. Hospice Utilization and End-of-Life Care Patterns'])
ws['A1'].font = TITLE_FONT
ws.append([])

header_row = ws.max_row + 1
for ci, cn in enumerate(['Outcome', 'Value'], 1):
    cell = ws.cell(row=header_row, column=ci, value=cn)
    cell.font      = HEADER_FONT
    cell.fill      = HEADER_FILL
    cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                wrap_text=True, vertical='center')
ws.row_dimensions[header_row].height = 28

alt = 0
for ri, row_data in enumerate(pd.DataFrame(rows).itertuples(index=False), header_row + 1):
    is_sec = row_data.is_section
    if not is_sec:
        alt += 1
    for ci, val in enumerate([row_data.label, row_data.val], 1):
        cell = ws.cell(row=ri, column=ci, value=val)
        if is_sec:
            cell.font = SECTION_FONT
            cell.fill = SECTION_FILL
        else:
            cell.font = BODY_FONT
            if alt % 2 == 0:
                cell.fill = ALT_FILL
        cell.alignment = Alignment(
            horizontal='left' if ci == 1 else 'center',
            vertical='center', wrap_text=True)

ws.column_dimensions['A'].width = 58
ws.column_dimensions['B'].width = 26
ws.freeze_panes = f'B{header_row + 1}'

footer_row = ws.max_row + 2
ws.cell(row=footer_row, column=1,
        value='Hospice LOS and days from last ICI dose to hospice enrollment are reported among hospice enrollees only. '
              'ICI timing and in-hospital death are reported for the full cohort unless otherwise noted. '
              'Short hospice stay defined as LOS ≤7 days. '
              'LOS = length of stay.')
ws.cell(row=footer_row, column=1).font = Font(name='Times New Roman', italic=True, size=11, color='555555')

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
