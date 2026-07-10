"""
make_supp_codes.py
Supplementary Table 1: ICD-10 codes used to define eligible head and neck cancer (HNC).

GROUND TRUTH: the code list and subsite mapping are sourced directly from the
pipeline SQL — specifically the CASE / WHERE clauses in 04_io_subsite.sql, which
is what the cohort assembly actually executes.

Output: tables/supp_table_codes.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

OUT_PATH = r"c:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\supp_table_codes.xlsx"

# Style — matches other tables
SCARLET, WHITE = 'BA0C2F', 'FFFFFF'
SECTION_PINK, ALT_PINK = 'F5D0D6', 'F9ECEE'

TITLE_FONT    = Font(name='Times New Roman', bold=True,   size=12, color=SCARLET)
SUBTITLE_FONT = Font(name='Times New Roman', italic=True, size=11, color='555555')
HEADER_FONT   = Font(name='Times New Roman', bold=True,   color=WHITE, size=11)
SECTION_FONT  = Font(name='Times New Roman', bold=True,   size=11, color='7A0820')
BODY_FONT     = Font(name='Times New Roman', size=11)
NOTE_FONT     = Font(name='Times New Roman', italic=True, size=11, color='555555')

HEADER_FILL  = PatternFill('solid', fgColor=SCARLET)
SECTION_FILL = PatternFill('solid', fgColor=SECTION_PINK)
ALT_FILL     = PatternFill('solid', fgColor=ALT_PINK)


# ── Pipeline-derived subsite mapping ──────────────────────────────────────────
# Ground truth: codes.json, which is read at runtime by build_step4_sql() in
# io_pipeline.py — the static 04_io_subsite.sql file is stale and not executed.
# This mapping matches both the live io_subsite table and the spec sheet.
SUBSITE_CODES = {
    'Oral Cavity': [
        ('C00.0–C00.9',          'Lip (external and inner aspects, all subsites)'),
        ('C02.0–C02.3, C02.8–C02.9', 'Tongue, other and unspecified parts (excludes C02.4, lingual tonsil)'),
        ('C03.0–C03.9',          'Gum (upper, lower, unspecified)'),
        ('C04.0–C04.9',          'Floor of mouth'),
        ('C05.0, C05.8–C05.9',   'Hard palate; overlapping and unspecified palate (excludes soft palate/uvula)'),
        ('C06.0–C06.9',          'Other and unspecified parts of mouth (cheek mucosa, vestibule, retromolar area)'),
    ],
    'Oropharynx': [
        ('C01',            'Base of tongue'),
        ('C02.4',          'Lingual tonsil'),
        ('C05.1–C05.2',    'Soft palate and uvula'),
        ('C09.0–C09.9',    'Tonsil (fossa, pillar, overlapping, unspecified)'),
        ('C10.0–C10.9',    'Oropharynx (vallecula, anterior epiglottis, lateral/posterior wall, branchial cleft)'),
        ('C14.0',          'Pharynx, unspecified'),
        ('C14.2',          'Waldeyer ring'),
        ('C14.8',          'Overlapping sites of lip, oral cavity, and pharynx'),
    ],
    'Hypopharynx': [
        ('C12',          'Pyriform sinus'),
        ('C13.0–C13.9',  'Hypopharynx (postcricoid, aryepiglottic fold hypopharyngeal aspect, posterior wall, overlapping, unspecified)'),
    ],
    'Larynx': [
        ('C32.0–C32.9',  'Larynx (glottis, supraglottis, subglottis, cartilage, overlapping, unspecified)'),
    ],
}

# Excluded subsites (for transparency — referenced in the methods text)
EXCLUDED_SUBSITES = [
    ('C11.x',           'Nasopharynx'),
    ('C07, C08.x',      'Major salivary glands (parotid, submandibular, sublingual)'),
    ('C30.0, C30.1',    'Nasal cavity / middle ear'),
    ('C31.x',           'Accessory (paranasal) sinuses'),
    ('C44.x',           'Cutaneous (skin) primaries'),
]

# ── Build workbook ────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")
wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'HNC ICD-10 Codes'

# Title
ws.merge_cells('A1:C1')
ws['A1'] = ('Supplementary Table 1. ICD-10 Diagnosis Codes Used to Identify '
            'Eligible Mucosal Head and Neck Cancer')
ws['A1'].font = TITLE_FONT
ws['A1'].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
ws.row_dimensions[1].height = 30

# Subtitle
ws.merge_cells('A2:C2')
ws['A2'] = ('Beneficiaries were required to have ≥2 claims on separate dates with at '
            'least one of the codes below during the 24 months before death.')
ws['A2'].font = SUBTITLE_FONT
ws['A2'].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
ws.row_dimensions[2].height = 28

# Column header
HEADER_ROW = 4
for ci, h in enumerate(['Subsite', 'ICD-10 Code', 'Description'], 1):
    cell = ws.cell(row=HEADER_ROW, column=ci, value=h)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = Alignment(horizontal='left' if ci != 2 else 'center',
                               vertical='center')
ws.row_dimensions[HEADER_ROW].height = 22

# Data rows — by subsite
SUBSITE_ORDER = ['Oral Cavity', 'Oropharynx', 'Hypopharynx', 'Larynx']
r = HEADER_ROW + 1
for subsite in SUBSITE_ORDER:
    codes = SUBSITE_CODES[subsite]
    # Section header row (subsite name)
    ws.merge_cells(f'A{r}:C{r}')
    cell = ws.cell(row=r, column=1, value=subsite.upper())
    cell.font = SECTION_FONT
    cell.fill = SECTION_FILL
    cell.alignment = Alignment(horizontal='left', vertical='center')
    ws.row_dimensions[r].height = 20
    r += 1
    # Code rows
    for idx, (code, desc) in enumerate(codes):
        fill = ALT_FILL if idx % 2 == 1 else PatternFill('solid', fgColor=WHITE)
        for ci, val in enumerate(['', code, desc], 1):
            cell = ws.cell(row=r, column=ci, value=val)
            cell.font = BODY_FONT
            cell.fill = fill
            cell.alignment = Alignment(horizontal='left' if ci != 2 else 'center',
                                       vertical='center', wrap_text=True)
        ws.row_dimensions[r].height = 20
        r += 1
    r += 1  # blank row between subsite groups

# Excluded subsites (transparency block)
ws.merge_cells(f'A{r}:C{r}')
cell = ws.cell(row=r, column=1, value='EXCLUDED SUBSITES (NOT ELIGIBLE)')
cell.font = SECTION_FONT
cell.fill = SECTION_FILL
cell.alignment = Alignment(horizontal='left', vertical='center')
ws.row_dimensions[r].height = 20
r += 1
for idx, (code_range, subsite) in enumerate(EXCLUDED_SUBSITES):
    fill = ALT_FILL if idx % 2 == 1 else PatternFill('solid', fgColor=WHITE)
    for ci, val in enumerate(['', code_range, subsite], 1):
        cell = ws.cell(row=r, column=ci, value=val)
        cell.font = BODY_FONT
        cell.fill = fill
        cell.alignment = Alignment(horizontal='left' if ci != 2 else 'center',
                                   vertical='center')
    r += 1

# Footnote
r += 1
ws.merge_cells(f'A{r}:C{r}')
note = ws.cell(row=r, column=1, value=(
    'Codes are formatted with the ICD-10-CM decimal convention (e.g., C32.0). '
    'Cutaneous primaries (C44.x) were excluded because cutaneous squamous cell carcinoma '
    'is managed differently from mucosal HNC.'
))
note.font = NOTE_FONT
note.alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[r].height = 70

# Column widths
ws.column_dimensions['A'].width = 16
ws.column_dimensions['B'].width = 26
ws.column_dimensions['C'].width = 70

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
