"""
make_table1.py
Patient characteristics: Overall + stratified by hospice enrollment
Columns: Overall | Hospice | No Hospice | SMD
Output: C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/table1.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from scipy.stats import norm
from math import floor, sqrt

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\table1.xlsx"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading io_analytic...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        hospice_enrolled,
        age_at_death, age_cat,
        sex, race,
        dual_eligible,
        census_region, urban_rural,
        subsite_category,
        io_agent, io_regimen,
        last_episode_doses,
        primary_curative_type,
        had_surgery, had_radiation,
        van_walraven_score,
        days_last_io_to_death,
        in_hospital_death,
        death_year
    FROM io_analytic
""").df()
con.close()

df['hospice_enrolled'] = df['hospice_enrolled'].fillna(0).astype(int)
df['in_hospital_death'] = df['in_hospital_death'].fillna(0).astype(int)
df['dual_eligible'] = df['dual_eligible'].fillna(0).astype(int)
df['had_surgery'] = df['had_surgery'].fillna(0).astype(int)
df['had_radiation'] = df['had_radiation'].fillna(0).astype(int)
df['io_regimen'] = df['io_regimen'].replace({'IO monotherapy': 'ICI monotherapy', 'chemo-IO': 'chemo-ICI'})

hosp   = df[df['hospice_enrolled'] == 1]
nohosp = df[df['hospice_enrolled'] == 0]

print(f"  Total: {len(df):,}  Hospice: {len(hosp):,}  No hospice: {len(nohosp):,}")

# ── SMD helpers ───────────────────────────────────────────────────────────────
def smd_bin(col, val=None):
    """SMD for binary: |p1 - p2| / sqrt((p1(1-p1) + p2(1-p2)) / 2)"""
    p1 = (hosp[col] == val).mean() if val is not None else hosp[col].mean()
    p2 = (nohosp[col] == val).mean() if val is not None else nohosp[col].mean()
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return float(abs(p1 - p2) / denom) if denom > 0 else 0.0

def smd_cont(col):
    """SMD for continuous: |mean1 - mean2| / sqrt((var1 + var2) / 2)"""
    a = hosp[col].dropna()
    b = nohosp[col].dropna()
    denom = np.sqrt((a.var() + b.var()) / 2)
    return float(abs(a.mean() - b.mean()) / denom) if denom > 0 else 0.0

def smd_cat(col):
    """Categorical SMD: max binary SMD across all levels"""
    cats = df[col].dropna().unique()
    return max((smd_bin(col, c) for c in cats), default=0.0)

def fmt_smd(s):
    return f"{s:.2f}"

# ── Effect-size + 95% CI helpers (hospice vs no-hospice; JAMA-style) ──────────
# For continuous variables: Hodges-Lehmann median difference + rank-based 95% CI
#   (the effect-size partner of the Mann-Whitney U test).
# For binary variables: risk difference in percentage points + Wald 95% CI.
# For multi-level categoricals: blank (use SMD column for imbalance summary).

def hl_diff_ci(col, alpha=0.05):
    """Hodges-Lehmann median difference (hospice - no-hospice) and 95% CI."""
    a = hosp[col].dropna().values.astype(float)
    b = nohosp[col].dropna().values.astype(float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None, None
    diffs = (a[:, None] - b[None, :]).ravel()
    diffs.sort()
    N = len(diffs)
    hl = float(np.median(diffs))
    z = float(norm.ppf(1 - alpha / 2))
    se = sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    k = int(floor(N / 2.0 - z * se))
    if k < 1:
        return hl, float(diffs[0]), float(diffs[-1])
    return hl, float(diffs[k - 1]), float(diffs[N - k])

def rd_ci(col, val=None, alpha=0.05):
    """Risk difference (hospice - no-hospice, percentage points) with Wald 95% CI."""
    if val is not None:
        x1 = int((hosp[col]   == val).sum())
        x2 = int((nohosp[col] == val).sum())
    else:
        x1 = int(hosp[col].sum())
        x2 = int(nohosp[col].sum())
    n1, n2 = len(hosp), len(nohosp)
    p1, p2 = x1 / n1, x2 / n2
    rd = (p1 - p2) * 100.0   # percentage points
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = float(norm.ppf(1 - alpha / 2))
    return rd, (p1 - p2 - z * se) * 100.0, (p1 - p2 + z * se) * 100.0

def fmt_effect(eff, lo, hi, *, decimals=1):
    if eff is None:
        return ''
    return f"{eff:.{decimals}f} ({lo:.{decimals}f} to {hi:.{decimals}f})"

# ── Format helpers ─────────────────────────────────────────────────────────────
def n_pct(series, val=None):
    if val is not None:
        n = (series == val).sum()
        p = 100.0 * n / len(series)
    else:
        n = series.sum()
        p = 100.0 * n / len(series)
    return f"{n:,} ({p:.1f}%)"

def med_iqr(series):
    s = series.dropna()
    return f"{s.median():.0f} ({s.quantile(0.25):.0f}–{s.quantile(0.75):.0f})"

def mean_sd(series):
    s = series.dropna()
    return f"{s.mean():.1f} ({s.std():.1f})"

# ── Build rows ────────────────────────────────────────────────────────────────
rows = []

def section(title):
    rows.append({'label': title, 'is_section': True,
                 'overall': '', 'hospice': '', 'no_hospice': '', 'smd': '', 'effect': ''})

def add_cont(label, col, fmt='median', indent=False, show_effect=True):
    """Continuous variable. Effect = Hodges-Lehmann median diff + rank-based 95% CI."""
    fn = med_iqr if fmt == 'median' else mean_sd
    try:
        smd = fmt_smd(smd_cont(col))
    except Exception:
        smd = ''
    if show_effect:
        eff, lo, hi = hl_diff_ci(col)
        effect_str = fmt_effect(eff, lo, hi, decimals=1)
    else:
        effect_str = ''
    rows.append({
        'label':      ('    ' + label) if indent else label,
        'is_section': False,
        'overall':    fn(df[col]),
        'hospice':    fn(hosp[col]),
        'no_hospice': fn(nohosp[col]),
        'smd':        smd,
        'effect':     effect_str,
    })

def add_bin(label, col, val=None, show_smd=True, indent=False, show_effect=True):
    """Binary variable. Effect = risk difference (percentage points) + Wald 95% CI."""
    try:
        smd = fmt_smd(smd_bin(col, val)) if show_smd else ''
    except Exception:
        smd = ''
    if show_effect:
        eff, lo, hi = rd_ci(col, val)
        effect_str = fmt_effect(eff, lo, hi, decimals=1)
    else:
        effect_str = ''
    rows.append({
        'label':      ('    ' + label) if indent else label,
        'is_section': False,
        'overall':    n_pct(df[col], val),
        'hospice':    n_pct(hosp[col], val),
        'no_hospice': n_pct(nohosp[col], val),
        'smd':        smd,
        'effect':     effect_str,
    })

def add_cat(label, col, indent=False):
    """Multi-level categorical. Effect column left blank (use SMD for imbalance summary)."""
    try:
        smd = fmt_smd(smd_cat(col))
    except Exception:
        smd = ''
    rows.append({
        'label':      ('    ' + label) if indent else label,
        'is_section': False,
        'overall':    '', 'hospice': '', 'no_hospice': '', 'smd': smd, 'effect': '',
    })
    for val in sorted(df[col].dropna().unique()):
        rows.append({
            'label': '        ' + str(val),
            'is_section': False,
            'overall':    n_pct(df[col], val),
            'hospice':    n_pct(hosp[col], val),
            'no_hospice': n_pct(nohosp[col], val),
            'smd': '', 'effect': '',
        })

# ── Demographics ───────────────────────────────────────────────────────────────
section('DEMOGRAPHICS')
add_cont('Age at death, median (IQR)', 'age_at_death')
add_bin('Male sex', 'sex', 'Male')
add_cat('Race/ethnicity', 'race')
add_bin('Dual eligible (Medicaid)', 'dual_eligible', 1)
add_cat('Urban/rural', 'urban_rural')

section('CLINICAL CHARACTERISTICS')
add_cat('HNC subsite', 'subsite_category')
add_cat('ICI agent', 'io_agent')
add_cat('ICI regimen', 'io_regimen')
add_cont('Last episode ICI doses, median (IQR)', 'last_episode_doses')

section('PRIOR CURATIVE THERAPY')
add_cat('Primary curative type', 'primary_curative_type')

section('COMORBIDITY')
add_cont('van Walraven score, median (IQR)', 'van_walraven_score')

section('OUTCOMES')
add_bin('Hospice enrolled', 'hospice_enrolled', 1, show_smd=False, show_effect=False)
add_bin('In-hospital death', 'in_hospital_death', 1)
add_cont('Days from last ICI to death, median (IQR)', 'days_last_io_to_death')

# ── Build DataFrame ────────────────────────────────────────────────────────────
df_table = pd.DataFrame(rows)

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")

HEADER_FILL  = PatternFill('solid', fgColor='BA0C2F')   # OSU Scarlet
HEADER_FONT  = Font(name='Times New Roman', bold=True, color='FFFFFF', size=11)
BODY_FONT    = Font(name='Times New Roman', size=11)
SECTION_FILL = PatternFill('solid', fgColor='F5D0D6')   # Light scarlet tint
SECTION_FONT = Font(name='Times New Roman', bold=True, size=11, color='7A0820')
ALT_FILL     = PatternFill('solid', fgColor='F9ECEE')   # Very light tint
TITLE_FONT   = Font(name='Times New Roman', bold=True, size=12, color='BA0C2F')
SMD_HIGH     = PatternFill('solid', fgColor='FFE699')   # Highlight SMD >=0.20

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Table 1'

ws.append(['Table 1. Patient Characteristics'])
ws['A1'].font = TITLE_FONT
ws.append([])

n_hosp   = len(hosp)
n_nohosp = len(nohosp)

# Headers
header_row = ws.max_row + 1
cols = [
    'Characteristic',
    'Overall',
    'Hospice Enrolled',
    'No Hospice',
    'SMD',
    'Difference (95% CI)',
]
for ci, cn in enumerate(cols, 1):
    cell = ws.cell(row=header_row, column=ci, value=cn)
    cell.font      = HEADER_FONT
    cell.fill      = HEADER_FILL
    cell.alignment = Alignment(horizontal='center' if ci > 1 else 'left',
                                wrap_text=True, vertical='center')
ws.row_dimensions[header_row].height = 36

SMD_COL = 5
EFFECT_COL = 6

alt = 0
for ri, row_data in enumerate(df_table.itertuples(index=False), header_row + 1):
    is_sec = row_data.is_section
    if not is_sec:
        alt += 1
    vals = [row_data.label, row_data.overall, row_data.hospice,
            row_data.no_hospice, row_data.smd, row_data.effect]
    for ci, val in enumerate(vals, 1):
        cell = ws.cell(row=ri, column=ci, value=val)
        if is_sec:
            cell.font = SECTION_FONT
            cell.fill = SECTION_FILL
        else:
            cell.font = BODY_FONT
            if alt % 2 == 0:
                cell.fill = ALT_FILL
        # Highlight SMD >= 0.20 (potentially important imbalance)
        if ci == SMD_COL and not is_sec and val not in ('', None):
            try:
                if float(val) >= 0.20:
                    cell.fill = SMD_HIGH
            except (ValueError, TypeError):
                pass
        # Bold effect cells whose 95% CI excludes zero (i.e., "significant" by CI)
        if ci == EFFECT_COL and not is_sec and isinstance(val, str) and val:
            try:
                lo_str = val.split('(', 1)[1].split(' to ')[0]
                hi_str = val.split(' to ')[1].rstrip(')')
                lo_val = float(lo_str)
                hi_val = float(hi_str)
                if (lo_val > 0 and hi_val > 0) or (lo_val < 0 and hi_val < 0):
                    cell.font = Font(name='Times New Roman', size=11, bold=True)
            except (IndexError, ValueError):
                pass
        cell.alignment = Alignment(
            horizontal='left' if ci == 1 else 'center',
            vertical='center', wrap_text=(ci == 1))

ws.column_dimensions['A'].width = 44
for letter in ['B', 'C', 'D', 'E']:
    ws.column_dimensions[letter].width = 18
ws.column_dimensions['F'].width = 26

ws.freeze_panes = f'B{header_row + 1}'

footer_row = ws.max_row + 2
ws.cell(row=footer_row, column=1,
        value='Continuous variables: median (IQR). Binary/categorical: n (%). '
              'SMD = standardized mean difference; '
              '<0.10 negligible, 0.10–<0.20 modest, ≥0.20 potentially important (highlighted). '
              'Difference (95% CI) compares hospice-enrolled vs non-enrolled (hospice − no hospice): '
              'Hodges-Lehmann median difference (with rank-based CI, the effect-size partner of the '
              'Mann-Whitney U test) for continuous variables, in original units; '
              'absolute risk difference (with Wald CI) for binary variables, in percentage points. '
              'Bolded entries indicate 95% CIs that exclude zero.')
ws.cell(row=footer_row, column=1).font = Font(name='Times New Roman', italic=True, size=11, color='555555')

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
