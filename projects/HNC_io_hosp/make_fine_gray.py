"""
make_fine_gray.py
Fine-Gray subdistribution hazard model as a sensitivity analysis to the
primary binary-logit regression (Table 3, timely enrollment).

Event of interest = 1 (hospice enrollment before death)
Competing event    = 2 (death without hospice enrollment)
Time zero          = last ICI administration

The Fine-Gray sHR is on the cumulative-incidence scale: it tells us how
each covariate shifts the CIF for hospice enrollment, accounting for the
competing risk of death before enrollment. This is the estimand the PI
requested as the "proper" way to disentangle real behavioral drivers of
hospice enrollment from the immortal-time-bias artifact.

Implementation:
  - Export analytic subset to CSV
  - Invoke Rscript on r_scripts/fine_gray.R (uses cmprsk::crr)
  - Parse CSV of sHRs + 95% CI + p
  - Format as Excel table matching manuscript style

Output: tables/table3_fine_gray.xlsx
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import openpyxl
import re
import subprocess
from pathlib import Path
from openpyxl.styles import Font, PatternFill, Alignment

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\table3_fine_gray.xlsx"
R_SCRIPT = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\r_scripts\fine_gray.R"
RSCRIPT_EXE = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"

TMP_DIR  = Path(r"C:\Users\hsaee\AppData\Local\Temp\claude\c--Users-hsaee-Desktop-CMS-viewer\22e7a4b6-e092-4b2a-ae6e-d7a1f2bb0bc3\scratchpad")
TMP_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV  = str(TMP_DIR / "fg_input.csv")
OUTPUT_CSV = str(TMP_DIR / "fg_output.csv")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading io_analytic ...")
con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT
        time_to_event, event_type,
        age_at_death, sex, race, dual_eligible, urban_rural, census_region,
        subsite_category, io_agent, io_regimen, last_episode_doses,
        van_walraven_score, primary_curative_type, death_year
    FROM io_analytic
""").df()
con.close()

# Data prep mirroring make_regression.py
df['dual_eligible']      = df['dual_eligible'].fillna(0).astype(int)
df['van_walraven_score'] = pd.to_numeric(df['van_walraven_score'], errors='coerce').fillna(0)
df['urban_rural']        = df['urban_rural'].fillna('Unknown')
df['census_region']      = df['census_region'].fillna('Unknown')
df['race_collapsed']     = df['race'].replace({
    'Asian/PI': 'Other/Unknown', 'Native American': 'Other/Unknown',
    'Other': 'Other/Unknown', 'Unknown': 'Other/Unknown',
})
df['io_regimen'] = df['io_regimen'].replace(
    {'IO monotherapy': 'ICI monotherapy', 'chemo-IO': 'chemo-ICI'})
df['age_at_death']       = pd.to_numeric(df['age_at_death'], errors='coerce')
df['last_episode_doses'] = pd.to_numeric(df['last_episode_doses'], errors='coerce')
df['death_year_c']       = df['death_year'].astype(float) - 2017
df['time_to_event']      = pd.to_numeric(df['time_to_event'], errors='coerce')
df['event_type']         = pd.to_numeric(df['event_type'], errors='coerce').astype(int)

# Set reference categories via factor ordering (R will pick first level as ref)
REF_ORDER = {
    'sex':                 ['Male', 'Female'],
    'race_collapsed':      ['White', 'Black', 'Hispanic', 'Other/Unknown'],
    'urban_rural':         ['Metro', 'Non-metro', 'Unknown'],
    'census_region':       ['South', 'Northeast', 'Midwest', 'West', 'Unknown'],
    'subsite_category':    ['Hypopharynx', 'Larynx', 'Oral Cavity', 'Oropharynx'],
    'io_agent':            ['pembrolizumab', 'nivolumab', 'both'],
    'io_regimen':          ['ICI monotherapy', 'chemo-ICI'],
    'primary_curative_type': ['radiation', 'surgery'],
}
for col, order in REF_ORDER.items():
    exist = [v for v in order if v in df[col].astype(str).unique()]
    df[col] = pd.Categorical(df[col], categories=exist)

# Drop rows with any missing predictor
COVARS = ['age_at_death', 'sex', 'race_collapsed', 'dual_eligible', 'urban_rural',
          'census_region', 'subsite_category', 'io_agent', 'io_regimen',
          'last_episode_doses', 'van_walraven_score',
          'primary_curative_type', 'death_year_c']

df_fit = df[COVARS + ['time_to_event', 'event_type']].copy()
df_fit = df_fit[df_fit['time_to_event'] > 0].dropna()

print(f"  N = {len(df_fit):,}  event=1 (hospice): {(df_fit['event_type']==1).sum():,}  "
      f"event=2 (death w/o hospice): {(df_fit['event_type']==2).sum():,}")

# Write CSV for R
print(f"Writing input CSV: {INPUT_CSV}")
df_fit.to_csv(INPUT_CSV, index=False)

# ── Invoke Rscript ───────────────────────────────────────────────────────────
covars_str = ",".join(COVARS)
print(f"Invoking Rscript ...")
result = subprocess.run(
    [RSCRIPT_EXE, R_SCRIPT, INPUT_CSV, covars_str, OUTPUT_CSV],
    capture_output=True, text=True,
)
print("--- R stdout ---")
print(result.stdout)
if result.stderr:
    print("--- R stderr ---")
    print(result.stderr)
if result.returncode != 0:
    sys.exit(f"Rscript failed with exit code {result.returncode}")

# ── Parse output ─────────────────────────────────────────────────────────────
print("Parsing sHRs ...")
res = pd.read_csv(OUTPUT_CSV)
print(res.to_string(index=False))

# Map R's model-matrix term names back to a readable label
# R's model.matrix produces column names like "sexFemale" (categorical) or "age_at_death"
def clean_r_term(t):
    for var, order in REF_ORDER.items():
        if t.startswith(var) and t != var:
            level = t[len(var):]
            return f"    {level} vs {order[0]}"
    label_map = {
        'age_at_death':       'Age at death (per year)',
        'dual_eligible':      'Dual eligible (Medicaid)',
        'last_episode_doses': 'ICI doses in final episode (per dose)',
        'van_walraven_score': 'van Walraven score (per unit)',
        'death_year_c':       'Calendar year (per year from 2017)',
    }
    return label_map.get(t, t)

# Group ordering to match Table 3 primary sheet
GROUP_ORDER = [
    ('Age at death (per year)',                  ['age_at_death']),
    ('Sex',                                       ['sex']),
    ('Race/ethnicity',                            ['race_collapsed']),
    ('Dual eligible (Medicaid)',                  ['dual_eligible']),
    ('Urban/rural',                               ['urban_rural']),
    ('Census region',                             ['census_region']),
    ('HNC subsite',                               ['subsite_category']),
    ('ICI agent',                                 ['io_agent']),
    ('ICI regimen',                               ['io_regimen']),
    ('ICI doses in final episode (per dose)',     ['last_episode_doses']),
    ('van Walraven score (per unit)',             ['van_walraven_score']),
    ('Prior curative therapy',                    ['primary_curative_type']),
    ('Calendar year (per year from 2017)',        ['death_year_c']),
]

# ── Write Excel ──────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")

SCARLET, WHITE = 'BA0C2F', 'FFFFFF'
SECTION_PINK, ALT_PINK = 'F5D0D6', 'F9ECEE'
SIG_YELLOW = 'FFE699'

TITLE_FONT   = Font(name='Times New Roman', bold=True, size=12, color=SCARLET)
HEADER_FONT  = Font(name='Times New Roman', bold=True, color=WHITE, size=11)
SECTION_FONT = Font(name='Times New Roman', bold=True, size=11, color='7A0820')
REF_FONT     = Font(name='Times New Roman', italic=True, size=11, color='888888')
BODY_FONT    = Font(name='Times New Roman', size=11)
NOTE_FONT    = Font(name='Times New Roman', italic=True, size=11, color='555555')

HEADER_FILL  = PatternFill('solid', fgColor=SCARLET)
SECTION_FILL = PatternFill('solid', fgColor=SECTION_PINK)
ALT_FILL     = PatternFill('solid', fgColor=ALT_PINK)
SIG_FILL     = PatternFill('solid', fgColor=SIG_YELLOW)
REF_FILL     = PatternFill('solid', fgColor='F0F0F0')

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Fine-Gray sensitivity'

n_ev1 = int((df_fit['event_type']==1).sum())
n_tot = int(len(df_fit))

ws.merge_cells('A1:C1')
ws['A1'] = (f'Table 3 (sensitivity). Fine-Gray Subdistribution Hazard Model — '
            f'Hospice Enrollment (competing event: death without hospice); '
            f'events = {n_ev1:,} / {n_tot:,}')
ws['A1'].font = TITLE_FONT
ws['A1'].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
ws.row_dimensions[1].height = 32

HR = 3
for ci, h in enumerate(['Variable', 'sHR (95% CI)', 'p-value'], 1):
    cell = ws.cell(row=HR, column=ci, value=h)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                               vertical='center', wrap_text=True)
ws.row_dimensions[HR].height = 24

# Build row list with section headers and refs
r = HR + 1

def fmt_shr(shr, lo, hi):
    return f"{shr:.2f} ({lo:.2f}-{hi:.2f})"

def fmt_p(p):
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

for section_label, var_list in GROUP_ORDER:
    var = var_list[0]
    # Find all R terms belonging to this variable
    # For continuous: exact match to var name
    # For categorical: startswith(var) and != var
    if var in REF_ORDER:
        levels = REF_ORDER[var]
        # Section header
        ws.cell(row=r, column=1, value=section_label).font = SECTION_FONT
        ws.cell(row=r, column=1).fill = SECTION_FILL
        for c in (2, 3):
            ws.cell(row=r, column=c).fill = SECTION_FILL
        r += 1
        # Ref row
        ws.cell(row=r, column=1, value=f'    {levels[0]} (ref)').font = REF_FONT
        ws.cell(row=r, column=1).fill = REF_FILL
        ws.cell(row=r, column=2, value='1.00  (—)').font = REF_FONT
        ws.cell(row=r, column=2).fill = REF_FILL
        ws.cell(row=r, column=2).alignment = Alignment(horizontal='center')
        ws.cell(row=r, column=3, value='ref').font = REF_FONT
        ws.cell(row=r, column=3).fill = REF_FILL
        ws.cell(row=r, column=3).alignment = Alignment(horizontal='center')
        r += 1
        # Non-ref levels
        for level in levels[1:]:
            r_term = f"{var}{level}"
            matching = res[res['term'] == r_term]
            if matching.empty:
                continue
            row = matching.iloc[0]
            is_sig = row['p'] < 0.05
            fill = SIG_FILL if is_sig else None
            ws.cell(row=r, column=1, value=f'    {level}').font = BODY_FONT
            ws.cell(row=r, column=2, value=fmt_shr(row['shr'], row['ci_lo'], row['ci_hi'])).font = BODY_FONT
            ws.cell(row=r, column=3, value=fmt_p(row['p'])).font = BODY_FONT
            if fill:
                for c in (1, 2, 3):
                    ws.cell(row=r, column=c).fill = fill
            ws.cell(row=r, column=2).alignment = Alignment(horizontal='center')
            ws.cell(row=r, column=3).alignment = Alignment(horizontal='center')
            r += 1
    else:
        # Continuous
        matching = res[res['term'] == var]
        if matching.empty:
            continue
        row = matching.iloc[0]
        is_sig = row['p'] < 0.05
        fill = SIG_FILL if is_sig else None
        ws.cell(row=r, column=1, value=section_label).font = BODY_FONT
        ws.cell(row=r, column=2, value=fmt_shr(row['shr'], row['ci_lo'], row['ci_hi'])).font = BODY_FONT
        ws.cell(row=r, column=3, value=fmt_p(row['p'])).font = BODY_FONT
        if fill:
            for c in (1, 2, 3):
                ws.cell(row=r, column=c).fill = fill
        ws.cell(row=r, column=2).alignment = Alignment(horizontal='center')
        ws.cell(row=r, column=3).alignment = Alignment(horizontal='center')
        r += 1

# Footer
r += 1
ws.merge_cells(f'A{r}:C{r}')
ws.cell(row=r, column=1, value=(
    'Fine-Gray subdistribution hazard model (cmprsk::crr, R 4.6.1). '
    'Event of interest = hospice enrollment; competing event = death without hospice enrollment. '
    'Time zero = last ICI administration. sHR = subdistribution hazard ratio; interpret as the '
    'multiplicative effect on the cumulative-incidence rate of hospice enrollment. '
    'Highlighted rows (yellow) = p<0.05. Reference categories shown in italics. '
    'This is the confirmatory competing-risks analysis to accompany the primary binary-outcome '
    'Table 3 (timely hospice enrollment).'
)).font = NOTE_FONT
ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[r].height = 90

ws.column_dimensions['A'].width = 46
ws.column_dimensions['B'].width = 24
ws.column_dimensions['C'].width = 12

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
