"""
make_table1.py

Single-sheet Table 1: cohort characteristics before and after propensity matching.

Layout (one table, two grouped column sets):
  Characteristic | Before Matching [Early Wks 1-4 | Late Wk 5+ | SMD] | After Matching [Early Wks 1-4 | Late Wk 5+ | SMD]

Comparison: Early SLP (Weeks 1-4, days 8-35) vs Late SLP (Week 5+, days 36-90).
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

_out_dir = Path(os.getenv("project_paths", ".")) / "stroke_SLP" / "output_files"
_out_dir.mkdir(parents=True, exist_ok=True)
DB_PATH  = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
OUT_PATH = _out_dir / "Table1.xlsx"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        p.DSYSRTKY,
        p.slp_timing_group,
        p.psm_matched_A,
        p.dschg_group,
        p.age_at_adm,
        p.sex,
        p.race,
        p.stroke_type,
        p.index_los,
        p.van_walraven_score,
        p.adm_year,
        p.mech_vent,
        p.prior_stroke,
        p.dementia,
        p.afib,
        p.hypertension,
        p.dyslipid,
        p.smoking,
        p.rucc_group,
        p.dual_eligible,
        o.days_to_death,
        o.days_to_aspiration,
        o.days_to_gtube
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()

for col in ['days_to_death', 'days_to_aspiration', 'days_to_gtube']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Cohort subsets ─────────────────────────────────────────────────────────────
early_pre  = df[df['slp_timing_group'] == 'Early']
late_pre   = df[df['slp_timing_group'] == 'Late']
early_post = df[(df['psm_matched_A'] == True) & (df['slp_timing_group'] == 'Early')]
late_post  = df[(df['psm_matched_A'] == True) & (df['slp_timing_group'] == 'Late')]

print(f"  Pre-match:  Early={len(early_pre):,}  Late={len(late_pre):,}")
print(f"  Post-match: Early={len(early_post):,}  Late={len(late_post):,}")


# ── SMD helpers ────────────────────────────────────────────────────────────────
def _smd_raw(v1, v0):
    v1 = np.array(v1, dtype=float); v1 = v1[~np.isnan(v1)]
    v0 = np.array(v0, dtype=float); v0 = v0[~np.isnan(v0)]
    if not len(v1) or not len(v0): return np.nan
    pooled = np.sqrt((v1.std()**2 + v0.std()**2) / 2)
    return abs(v1.mean() - v0.mean()) / pooled if pooled else 0.0

def smd_cont(col):
    return lambda e, l: _smd_raw(e[col].values, l[col].values)

def smd_bin(col):
    def _fn(e, l):
        p1 = e[col].fillna(0).mean(); p0 = l[col].fillna(0).mean()
        pooled = np.sqrt((p1*(1-p1) + p0*(1-p0)) / 2)
        return abs(p1 - p0) / pooled if pooled else 0.0
    return _fn

def smd_cat(col, val):
    def _fn(e, l):
        p1 = (e[col] == val).mean(); p0 = (l[col] == val).mean()
        pooled = np.sqrt((p1*(1-p1) + p0*(1-p0)) / 2)
        return abs(p1 - p0) / pooled if pooled else 0.0
    return _fn

def fmt_smd(v):
    try: return f'{float(v):.3f}' if not np.isnan(float(v)) else ''
    except: return ''


# ── Display formatters ─────────────────────────────────────────────────────────
def fmt_mean_sd(col):
    return lambda d: f"{d[col].mean():.1f} ({d[col].std():.1f})"

def fmt_pct(col, val=None):
    def _fn(d):
        if val is not None:
            n = int((d[col] == val).sum()); p = 100.0 * n / max(len(d), 1)
        else:
            n = int(d[col].fillna(0).sum()); p = 100.0 * n / max(len(d), 1)
        return f"{n:,} ({p:.1f}%)"
    return _fn

def fmt_event(col, days=365):
    def _fn(d):
        n = int((d[col] <= days).sum()); p = 100.0 * n / max(len(d), 1)
        return f"{n:,} ({p:.1f}%)"
    return _fn


# ── Row specification ──────────────────────────────────────────────────────────
# (label, fmt_fn, smd_fn, indent)   smd_fn=None for section headers & outcome rows
ROWS = [
    ('DEMOGRAPHICS',                             None, None, False),
    ('Age, mean (SD)',                           fmt_mean_sd('age_at_adm'),           smd_cont('age_at_adm'),        True),
    ('Female, n (%)',                            fmt_pct('sex', 'Female'),             smd_cat('sex', 'Female'),      True),
    ('Race: White, n (%)',                       fmt_pct('race', 'White'),             smd_cat('race', 'White'),      True),
    ('Race: Black, n (%)',                       fmt_pct('race', 'Black'),             smd_cat('race', 'Black'),      True),
    ('Race: Hispanic, n (%)',                    fmt_pct('race', 'Hispanic'),          smd_cat('race', 'Hispanic'),   True),

    ('STROKE CHARACTERISTICS',                   None, None, False),
    ('Ischemic, n (%)',                          fmt_pct('stroke_type', 'Ischemic'),   smd_cat('stroke_type','Ischemic'), True),
    ('Intracerebral hemorrhage, n (%)',          fmt_pct('stroke_type', 'ICH'),        smd_cat('stroke_type','ICH'),      True),
    ('Subarachnoid hemorrhage, n (%)',           fmt_pct('stroke_type', 'SAH'),        smd_cat('stroke_type','SAH'),      True),

    ('HOSPITAL COURSE',                          None, None, False),
    ('Index LOS, mean (SD) days',               fmt_mean_sd('index_los'),             smd_cont('index_los'),         True),
    ('Mechanical ventilation, n (%)',            fmt_pct('mech_vent'),                 smd_bin('mech_vent'),          True),
    ('Discharge to home (no HHA), n (%)',        fmt_pct('dschg_group', 'Home'),       smd_cat('dschg_group','Home'),     True),
    ('Discharge with home health agency, n (%)', fmt_pct('dschg_group', 'Home+HHA'),  smd_cat('dschg_group','Home+HHA'), True),
    ('Admission year, mean (SD)',                fmt_mean_sd('adm_year'),              smd_cont('adm_year'),          True),

    ('COMORBIDITIES',                            None, None, False),
    ('van Walraven score, mean (SD)',            fmt_mean_sd('van_walraven_score'),    smd_cont('van_walraven_score'),True),
    ('Atrial fibrillation, n (%)',               fmt_pct('afib'),                      smd_bin('afib'),               True),
    ('Hypertension, n (%)',                      fmt_pct('hypertension'),              smd_bin('hypertension'),       True),
    ('Dyslipidemia, n (%)',                      fmt_pct('dyslipid'),                  smd_bin('dyslipid'),           True),
    ('Smoking, n (%)',                           fmt_pct('smoking'),                   smd_bin('smoking'),            True),
    ('Prior stroke, n (%)',                      fmt_pct('prior_stroke'),              smd_bin('prior_stroke'),       True),
    ('Dementia, n (%)',                          fmt_pct('dementia'),                  smd_bin('dementia'),           True),

    ('GEOGRAPHY & SOCIOECONOMIC STATUS',         None, None, False),
    ('Metro county, n (%)',                      fmt_pct('rucc_group', 'Metro'),       smd_cat('rucc_group','Metro'),    True),
    ('Nonmetro county, n (%)',                   fmt_pct('rucc_group', 'Nonmetro'),    smd_cat('rucc_group','Nonmetro'), True),
    ('Rural county, n (%)',                      fmt_pct('rucc_group', 'Rural'),       smd_cat('rucc_group','Rural'),    True),
    ('Dual eligible (Medicare+Medicaid), n (%)', fmt_pct('dual_eligible'),             smd_bin('dual_eligible'),         True),

]

SECTION_LABELS = {r[0] for r in ROWS if r[1] is None}

# ── Compute all cell values ────────────────────────────────────────────────────
# Columns (left to right):
#   A: Characteristic
#   B: Early Wks 1-4 (pre)    C: Late Wk 5+ (pre)    D: SMD (pre)
#   E: Early Wks 1-4 (post)   F: Late Wk 5+ (post)   G: SMD (post)

table_rows = []
for label, fmt_fn, smd_fn, indent in ROWS:
    lbl = ('   ' if indent else '') + label
    if fmt_fn is None:
        table_rows.append((lbl, '', '', '', '', '', ''))
    else:
        def _get(fn, d):
            try: return fn(d)
            except: return '\u2014'
        ep = _get(fmt_fn, early_pre)
        lp = _get(fmt_fn, late_pre)
        s1 = fmt_smd(smd_fn(early_pre, late_pre)) if smd_fn else ''
        em = _get(fmt_fn, early_post)
        lm = _get(fmt_fn, late_post)
        s2 = fmt_smd(smd_fn(early_post, late_post)) if smd_fn else ''
        table_rows.append((lbl, ep, lp, s1, em, lm, s2))

# ── Style constants ────────────────────────────────────────────────────────────
C_SCARLET    = 'BA0C2F'
C_DARK       = '70071C'
C_SECTION_BG = 'F0D8DC'
C_ALT_BG     = 'FDF5F6'
C_GRAY       = 'A7B1B7'
C_WHITE      = 'FFFFFF'

def fill(hex_col): return PatternFill('solid', fgColor=hex_col)
def font(bold=False, color='000000', size=10, italic=False):
    return Font(bold=bold, color=color, size=size, italic=italic)

THIN  = Side(style='thin',   color='CCCCCC')
MED   = Side(style='medium', color=C_DARK)
THICK = Side(style='medium', color=C_SCARLET)

def border(bottom=None, top=None, left=None, right=None):
    return Border(bottom=bottom or Side(style=None),
                  top=top       or Side(style=None),
                  left=left     or Side(style=None),
                  right=right   or Side(style=None))

# ── Write workbook ─────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")
wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Table1'

ws.append([])   # blank row

# Row 1: Title
ws.append(['Table 1. Cohort Characteristics Before and After Propensity Score Matching'])
ws['A2'].font = font(bold=True, size=13, color=C_SCARLET)
ws.merge_cells('A2:G2')
ws['A2'].alignment = Alignment(horizontal='left')
ws.row_dimensions[2].height = 20


# Row 4: Group header (merged spans)
# Cols: A=Characteristic, B-D=Before Matching, E-G=After Matching
GRP_ROW = 4
ws.cell(GRP_ROW, 1, '').font  = font(bold=True, size=10, color=C_WHITE)
ws.cell(GRP_ROW, 1).fill                    = fill(C_SCARLET)
ws.cell(GRP_ROW, 1).alignment               = Alignment(horizontal='center', vertical='center')

ws.cell(GRP_ROW, 2, 'Before Matching').font  = font(bold=True, size=10, color=C_WHITE)
ws.cell(GRP_ROW, 2).fill                     = fill(C_SCARLET)
ws.cell(GRP_ROW, 2).alignment                = Alignment(horizontal='center', vertical='center')
ws.merge_cells(f'B{GRP_ROW}:D{GRP_ROW}')

ws.cell(GRP_ROW, 5, 'After Matching').font   = font(bold=True, size=10, color=C_WHITE)
ws.cell(GRP_ROW, 5).fill                     = fill(C_SCARLET)
ws.cell(GRP_ROW, 5).alignment                = Alignment(horizontal='center', vertical='center')
ws.merge_cells(f'E{GRP_ROW}:G{GRP_ROW}')
ws.row_dimensions[GRP_ROW].height = 18

# Row 5: Column name headers
COL_HDRS = ['', 'Early SLP', 'Late SLP', 'SMD',
                 'Early SLP', 'Late SLP', 'SMD']
HDR_ROW = 5
for ci, h in enumerate(COL_HDRS, 1):
    cell = ws.cell(HDR_ROW, ci, h)
    cell.font      = font(bold=True, size=9, color=C_WHITE)
    cell.fill      = fill(C_SCARLET) if ci in (1, 5, 6, 7) else fill(C_SCARLET)
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
ws.row_dimensions[HDR_ROW].height = 30

# Row 6: Sample sizes
N_ROW = 6
N_VALS = ['', f'n = {len(early_pre):,}', f'n = {len(late_pre):,}', '',
               f'n = {len(early_post):,}', f'n = {len(late_post):,}', '']
for ci, v in enumerate(N_VALS, 1):
    cell = ws.cell(N_ROW, ci, v)
    cell.font      = font(bold=False, size=9, color=C_WHITE, italic=True)
    cell.fill      = fill(C_SCARLET) if ci in (1, 5, 6, 7) else fill(C_SCARLET)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border    = border(bottom=MED)
ws.row_dimensions[N_ROW].height = 16

# ── Data rows ──────────────────────────────────────────────────────────────────
alt = 0
for row_vals in table_rows:
    ri  = ws.max_row + 1
    lbl = str(row_vals[0]).strip()
    is_sec = lbl in SECTION_LABELS

    if not is_sec:
        alt += 1

    for ci, val in enumerate(row_vals, 1):
        cell = ws.cell(ri, ci, val)

        if is_sec:
            cell.font = font(bold=True, size=10, color=C_DARK)
            cell.fill = fill(C_SECTION_BG)
            cell.border = border(top=Side(style='thin', color=C_DARK),
                                 bottom=Side(style='thin', color=C_DARK))
        else:
            if alt % 2 == 0:
                cell.fill = fill(C_ALT_BG)

            # SMD columns: bold black = good balance (<0.10); muted gray = imbalanced (>=0.10)
            if ci in (4, 7) and val:
                try:
                    v = float(val)
                    cell.font = font(bold=True, size=10, color=C_SCARLET) if v < 0.10 \
                                else font(size=10, italic=True, color='888888')
                except (ValueError, TypeError):
                    pass
            else:
                cell.font = font(size=10)

        cell.alignment = Alignment(
            horizontal='left'   if ci == 1 else 'center',
            vertical='center',
            wrap_text=(ci == 1)
        )

    ws.row_dimensions[ri].height = 15

# ── Column widths ──────────────────────────────────────────────────────────────
ws.column_dimensions['A'].width = 40
ws.column_dimensions['B'].width = 18
ws.column_dimensions['C'].width = 18
ws.column_dimensions['D'].width = 7
ws.column_dimensions['E'].width = 18
ws.column_dimensions['F'].width = 18
ws.column_dimensions['G'].width = 7

ws.freeze_panes = f'B{N_ROW + 1}'

# ── Footnote ───────────────────────────────────────────────────────────────────
fn_row = ws.max_row + 2
ws.cell(fn_row, 1,
    'Comparison: Early SLP (Weeks 1\u20134) vs Late SLP (Week 5+).'
    'SMD = standardized mean difference; values \u22650.10 (bold red) indicate imbalance.'
    'PSM: 1:1 greedy nearest-neighbor matching,'
    'caliper = 0.2 \u00d7 SD(logit propensity score). '
).font = font(italic=True, size=8, color='666666')
ws.merge_cells(f'A{fn_row}:G{fn_row}')

wb.save(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
