"""
make_table1.py
Cohort demographics: pre- and post-propensity score matching
Columns: Pre-match TORS | Pre-match CT/CRT | SMD | Post-match TORS | Post-match CT/CRT | SMD
Output: F:\CMS\projects\opscc\table1_psm.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb, numpy as np, pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"F:\CMS\projects\opscc\table1_psm.xlsx"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        tx_group, psm_matched,
        age_at_dx, age_group, sex, race,
        van_walraven_score,
        chf, carit, valv, pcd, pvd, hypunc, hypc, para, ond, cpd,
        diabunc, diabc, hypothy, rf, ld, pud, aids, lymph, metacanc,
        solidtum, rheumd, coag, obes, wloss, fed, blane, dane,
        alcohol, drug, psycho, depre
    FROM opscc_propensity
    WHERE tx_group IN ('TORS only', 'CT/CRT only')
""").df()
con.close()

# Boolean/int conversion for Elixhauser flags
elix_flags = [
    'chf','carit','valv','pcd','pvd','hypunc','hypc','para','ond','cpd',
    'diabunc','diabc','hypothy','rf','ld','pud','aids','lymph','metacanc',
    'solidtum','rheumd','coag','obes','wloss','fed','blane','dane',
    'alcohol','drug','psycho','depre'
]
df[elix_flags] = df[elix_flags].fillna(0).astype(int)
df['van_walraven_score'] = df['van_walraven_score'].fillna(0)
df['tors'] = (df['tx_group'] == 'TORS only').astype(int)

# Combined variables for concise table
df['hypertension']  = ((df['hypunc'] == 1) | (df['hypc'] == 1)).astype(int)
df['diabetes']      = ((df['diabunc'] == 1) | (df['diabc'] == 1)).astype(int)
df['race_other']    = df['race'].isin(['Other', 'Unknown', 'AIAN']).astype(int)

# ── Define four groups ─────────────────────────────────────────────────────────
pre_tors   = df[df['tx_group'] == 'TORS only']
pre_ctcrt  = df[df['tx_group'] == 'CT/CRT only']
post_tors  = df[(df['tx_group'] == 'TORS only')  & (df['psm_matched'] == True)]
post_ctcrt = df[(df['tx_group'] == 'CT/CRT only') & (df['psm_matched'] == True)]

print(f"  Pre-match:  TORS={len(pre_tors):,}  CT/CRT={len(pre_ctcrt):,}")
print(f"  Post-match: TORS={len(post_tors):,}  CT/CRT={len(post_ctcrt):,}")

# ── SMD helpers ────────────────────────────────────────────────────────────────
def smd_cont(a, b):
    d = a.mean() - b.mean()
    p = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(d / p) if p > 0 else 0.0

def smd_bin(a, b):
    p1, p2 = a.mean(), b.mean()
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return abs((p1 - p2) / denom) if denom > 0 else 0.0

# ── Format helpers ─────────────────────────────────────────────────────────────
def mean_sd(s):
    return f"{s.mean():.1f} ({s.std():.1f})"

def pct(s, val=None):
    p = 100.0*(s == val).mean() if val is not None else 100.0*s.mean()
    n = int((s == val).sum()) if val is not None else int(s.sum())
    return f"{n:,} ({p:.1f}%)"

def fmt_smd(v):
    return f"{v:.3f}"

# ── Build table rows ───────────────────────────────────────────────────────────
rows = []

def section(title):
    rows.append({'label': title, 'is_section': True,
                 'pre_tors': '', 'pre_ctcrt': '', 'pre_smd': '',
                 'post_tors': '', 'post_ctcrt': '', 'post_smd': ''})

def add(label, pre_t, pre_c, post_t, post_c, smd_fn, indent=False):
    rows.append({
        'label':      ('    ' + label) if indent else label,
        'is_section': False,
        'pre_tors':   pre_t,
        'pre_ctcrt':  pre_c,
        'pre_smd':    smd_fn(pre_tors, pre_ctcrt),
        'post_tors':  post_t,
        'post_ctcrt': post_c,
        'post_smd':   smd_fn(post_tors, post_ctcrt),
    })

def cont(label, col, indent=False):
    add(label,
        mean_sd(pre_tors[col]),   mean_sd(pre_ctcrt[col]),
        mean_sd(post_tors[col]),  mean_sd(post_ctcrt[col]),
        lambda t, c, col=col: fmt_smd(smd_cont(t[col], c[col])),
        indent=indent)

def binary(label, col, val=None, indent=False):
    if val is not None:
        fn_pct = lambda s: pct(s[col], val)
        fn_smd = lambda t, c, col=col, val=val: fmt_smd(
            smd_bin((t[col]==val).astype(float), (c[col]==val).astype(float)))
    else:
        fn_pct = lambda s: pct(s[col])
        fn_smd = lambda t, c, col=col: fmt_smd(smd_bin(t[col].astype(float), c[col].astype(float)))
    add(label,
        fn_pct(pre_tors), fn_pct(pre_ctcrt),
        fn_pct(post_tors), fn_pct(post_ctcrt),
        fn_smd, indent=indent)

# ── Demographics ───────────────────────────────────────────────────────────────
section('DEMOGRAPHICS')
cont('Age at diagnosis, mean (SD)', 'age_at_dx')
binary('Male sex', 'sex', 'Male')

section('RACE / ETHNICITY')
binary('White',      'race', 'White',    indent=True)
binary('Black',      'race', 'Black',    indent=True)
binary('Hispanic',   'race', 'Hispanic', indent=True)
binary('Asian / PI', 'race', 'Asian/PI', indent=True)
binary('Other / Unknown', 'race_other',  indent=True)

section('COMORBIDITIES')
cont('van Walraven score, mean (SD)', 'van_walraven_score')
binary('Hypertension',             'hypertension',  indent=True)
binary('Chronic pulmonary disease','cpd',           indent=True)
binary('Diabetes mellitus',        'diabetes',      indent=True)
binary('Cardiac arrhythmia',       'carit',         indent=True)
binary('Congestive heart failure', 'chf',           indent=True)
binary('Renal failure',            'rf',            indent=True)
binary('Alcohol abuse',            'alcohol',       indent=True)
binary('Depression',               'depre',         indent=True)
binary('Obesity',                  'obes',          indent=True)
binary('Other neurological',       'ond',           indent=True)

# ── Build DataFrame ────────────────────────────────────────────────────────────
df_table = pd.DataFrame(rows)

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")

HEADER_FILL  = PatternFill('solid', fgColor='1F4E79')
HEADER_FONT  = Font(bold=True, color='FFFFFF', size=10)
SECTION_FILL = PatternFill('solid', fgColor='BDD7EE')
SECTION_FONT = Font(bold=True, size=10, color='1F4E79')
ALT_FILL     = PatternFill('solid', fgColor='EBF3FB')
SMD_WARN     = PatternFill('solid', fgColor='FFE699')   # SMD >= 0.10 highlight
TITLE_FONT   = Font(bold=True, size=13, color='1F4E79')

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Table 1'

ws.append(['Table 1. Cohort Characteristics — Pre- and Post-Propensity Score Matching'])
ws['A1'].font = TITLE_FONT
ws.append([])

# Column headers (merged groups)
n_pre_tors   = len(pre_tors)
n_pre_ctcrt  = len(pre_ctcrt)
n_post_tors  = len(post_tors)
n_post_ctcrt = len(post_ctcrt)

header_row = ws.max_row + 1

# Group header row
ws.cell(row=header_row, column=1, value='')
ws.cell(row=header_row, column=2, value='Before Matching').font = Font(bold=True, size=10, color='1F4E79')
ws.cell(row=header_row, column=5, value='After Matching').font  = Font(bold=True, size=10, color='1F4E79')
ws.merge_cells(start_row=header_row, start_column=2, end_row=header_row, end_column=4)
ws.merge_cells(start_row=header_row, start_column=5, end_row=header_row, end_column=7)
for col in [2, 5]:
    c = ws.cell(row=header_row, column=col)
    c.fill      = PatternFill('solid', fgColor='D6E4F0')
    c.font      = Font(bold=True, size=10, color='1F4E79')
    c.alignment = Alignment(horizontal='center')

# Column sub-headers
subheader_row = header_row + 1
cols = [
    'Characteristic',
    f'TORS Only\n(n={n_pre_tors:,})',
    f'CT/CRT Only\n(n={n_pre_ctcrt:,})',
    'SMD',
    f'TORS Only\n(n={n_post_tors:,})',
    f'CT/CRT Only\n(n={n_post_ctcrt:,})',
    'SMD',
]
for ci, cn in enumerate(cols, 1):
    cell = ws.cell(row=subheader_row, column=ci, value=cn)
    cell.font      = HEADER_FONT
    cell.fill      = HEADER_FILL
    cell.alignment = Alignment(horizontal='center' if ci > 1 else 'left',
                                wrap_text=True, vertical='center')

ws.row_dimensions[subheader_row].height = 32

# Data rows
alt = 0
for ri, row_data in enumerate(df_table.itertuples(index=False), subheader_row + 1):
    is_sec = row_data.is_section
    if not is_sec:
        alt += 1
    vals = [row_data.label, row_data.pre_tors, row_data.pre_ctcrt, row_data.pre_smd,
            row_data.post_tors, row_data.post_ctcrt, row_data.post_smd]
    for ci, val in enumerate(vals, 1):
        cell = ws.cell(row=ri, column=ci, value=val)
        if is_sec:
            cell.font = SECTION_FONT
            cell.fill = SECTION_FILL
        elif alt % 2 == 0:
            cell.fill = ALT_FILL

        # Highlight imbalanced SMD (>=0.10) in yellow
        if ci in (4, 7) and not is_sec and val != '':
            try:
                if float(val) >= 0.10:
                    cell.fill = SMD_WARN
            except (ValueError, TypeError):
                pass

        cell.alignment = Alignment(
            horizontal='left' if ci == 1 else 'center',
            vertical='center', wrap_text=(ci == 1))

# Column widths
ws.column_dimensions['A'].width = 40
for letter in ['B','C','D','E','F','G']:
    ws.column_dimensions[letter].width = 18

ws.freeze_panes = f'B{subheader_row + 1}'

# Footer note
footer_row = ws.max_row + 2
ws.cell(row=footer_row, column=1,
        value='SMD = standardized mean difference. Values ≥0.10 (highlighted) indicate potential imbalance. '
              'Continuous variables: n (mean ± SD). Categorical: n (%). '
              'Elixhauser comorbidities measured from claims in the 12 months prior to diagnosis.')
ws.cell(row=footer_row, column=1).font = Font(italic=True, size=8, color='555555')

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
