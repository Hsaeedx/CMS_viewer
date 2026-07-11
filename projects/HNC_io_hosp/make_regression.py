"""
make_regression.py
Table 3: Multivariable logistic regression — hospice enrollment (primary outcome)
Table 4: Multivariable logistic regression — in-hospital death (secondary outcome)
Output: C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/table3_regression.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\table3_regression.xlsx"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading io_analytic...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        hospice_enrolled, timely_enrollment, in_hospital_death,
        age_at_death, sex, race,
        dual_eligible, urban_rural, census_region,
        subsite_category, io_agent, io_regimen,
        last_episode_doses,
        van_walraven_score,
        primary_curative_type,
        death_year
    FROM io_analytic
""").df()
con.close()

# ── Data prep ──────────────────────────────────────────────────────────────────
df['hospice_enrolled']  = df['hospice_enrolled'].fillna(0).astype(int)
df['timely_enrollment'] = df['timely_enrollment'].fillna(0).astype(int)
df['in_hospital_death'] = df['in_hospital_death'].fillna(0).astype(int)
df['dual_eligible']     = df['dual_eligible'].fillna(0).astype(int)
df['van_walraven_score'] = pd.to_numeric(df['van_walraven_score'], errors='coerce').fillna(0)

# Fill unknown geography (non-US addresses, CT restructuring) rather than dropping
df['urban_rural']   = df['urban_rural'].fillna('Unknown')
df['census_region'] = df['census_region'].fillna('Unknown')

# Collapse sparse race categories
df['race_collapsed'] = df['race'].replace({
    'Asian/PI': 'Other/Unknown',
    'Native American': 'Other/Unknown',
    'Other': 'Other/Unknown',
    'Unknown': 'Other/Unknown',
})

df['io_regimen'] = df['io_regimen'].replace({'IO monotherapy': 'ICI monotherapy', 'chemo-IO': 'chemo-ICI'})

# Set reference categories using categorical dtype
cat_vars = {
    'sex':                ('Male', ['Male', 'Female']),
    'race_collapsed':     ('White', ['White', 'Black', 'Hispanic', 'Other/Unknown']),
    'urban_rural':        ('Metro', ['Metro', 'Non-metro', 'Unknown']),
    'census_region':      ('South', ['Northeast', 'Midwest', 'South', 'West', 'Unknown']),
    'subsite_category':   None,
    'io_agent':           ('pembrolizumab', None),
    'io_regimen':         ('ICI monotherapy', ['ICI monotherapy', 'chemo-ICI']),
    'primary_curative_type': ('radiation', None),
}

for col, spec in cat_vars.items():
    if spec is None:
        df[col] = pd.Categorical(df[col])
    else:
        ref, order = spec
        if order is None:
            df[col] = pd.Categorical(df[col])
        else:
            # Only keep levels that exist
            exist = [o for o in order if o in df[col].unique()]
            df[col] = pd.Categorical(df[col], categories=exist)
        # Reorder so ref is first
        if spec is not None and spec[0] in df[col].cat.categories:
            cats = [spec[0]] + [c for c in df[col].cat.categories if c != spec[0]]
            df[col] = df[col].cat.reorder_categories(cats)

df['age_at_death'] = pd.to_numeric(df['age_at_death'], errors='coerce')
df['last_episode_doses'] = pd.to_numeric(df['last_episode_doses'], errors='coerce')
df['death_year_c'] = df['death_year'].astype(float) - 2017  # centered at 2017

# Drop rows with any missing predictor
predictors = ['age_at_death', 'sex', 'race_collapsed', 'dual_eligible', 'urban_rural',
              'census_region', 'subsite_category', 'io_agent', 'io_regimen',
              'last_episode_doses', 'van_walraven_score',
              'primary_curative_type', 'death_year_c']

df_model = df[predictors + ['hospice_enrolled', 'timely_enrollment', 'in_hospital_death']].dropna()
print(f"  Model dataset: {len(df_model):,} rows (dropped {len(df)-len(df_model):,} with missing predictors)")

# ── Fit models ────────────────────────────────────────────────────────────────
# Primary outcome per PI: TIMELY enrollment (>3d before death), addresses "any
# enrollment" not capturing timing.
# Sensitivity: any hospice enrollment (original outcome).
# Also: in-hospital death.
formula = (
    'OUTCOME ~ age_at_death + C(sex) + C(race_collapsed) + dual_eligible '
    '+ C(urban_rural) + C(census_region) + C(subsite_category) '
    '+ C(io_agent) + C(io_regimen) '
    '+ last_episode_doses + van_walraven_score + C(primary_curative_type) '
    '+ death_year_c'
)

print("Fitting Table 3 (PRIMARY: timely hospice enrollment >3d before death)...")
model3 = smf.logit(formula.replace('OUTCOME', 'timely_enrollment'), data=df_model).fit(
    method='newton', maxiter=500, disp=False)

print("Fitting Table 3-sens (any hospice enrollment)...")
model3_sens = smf.logit(formula.replace('OUTCOME', 'hospice_enrolled'), data=df_model).fit(
    method='newton', maxiter=500, disp=False)

print("Fitting Table 4 (in-hospital death)...")
model4 = smf.logit(formula.replace('OUTCOME', 'in_hospital_death'), data=df_model).fit(
    method='newton', maxiter=500, disp=False)

# ── Extract results ────────────────────────────────────────────────────────────
def extract_results(model):
    """Return DataFrame with term, OR, 95% CI, p-value."""
    coef = model.params
    conf = model.conf_int()
    pval = model.pvalues
    results = pd.DataFrame({
        'term':  coef.index,
        'coef':  coef.values,
        'ci_lo': conf[0].values,
        'ci_hi': conf[1].values,
        'pval':  pval.values,
    })
    results['OR']    = np.exp(results['coef'])
    results['CI_lo'] = np.exp(results['ci_lo'])
    results['CI_hi'] = np.exp(results['ci_hi'])
    results = results[results['term'] != 'Intercept'].copy()
    return results

res3      = extract_results(model3)
res3_sens = extract_results(model3_sens)
res4      = extract_results(model4)

# ── Variable group ordering with reference categories ────────────────────────
# Each entry: (display_label, raw_prefix, reference_label)
GROUP_ORDER = [
    ('Age at death (per year)',                  'age_at_death',          None),
    ('Sex',                                      'C(sex)',                'Male'),
    ('Race/ethnicity',                           'C(race_collapsed)',     'White'),
    ('Dual eligible (Medicaid)',                 'dual_eligible',         None),
    ('Urban/rural',                              'C(urban_rural)',        'Metro'),
    ('Census region',                            'C(census_region)',      'South'),
    ('HNC subsite',                              'C(subsite_category)',   None),
    ('ICI agent',                                'C(io_agent)',           'pembrolizumab'),
    ('ICI regimen',                              'C(io_regimen)',         'ICI monotherapy'),
    ('ICI doses in final episode (per dose)',    'last_episode_doses',    None),
    ('van Walraven score (per unit)',             'van_walraven_score',    None),
    ('Prior curative therapy',                   'C(primary_curative_type)', 'radiation'),
    ('Calendar year (per year from 2017)',        'death_year_c',         None),
]

TERM_LABEL_MAP = {
    'age_at_death':       'Age at death (per year)',
    'dual_eligible':      'Dual eligible (Medicaid)',
    'last_episode_doses': 'ICI doses in final episode (per dose)',
    'van_walraven_score': 'van Walraven score (per unit)',
    'death_year_c':       'Calendar year (per year from 2017)',
}

def clean_term(t):
    """Convert statsmodels term name to readable indented label."""
    for raw, readable in TERM_LABEL_MAP.items():
        if t == raw:
            return readable
    # Categorical: C(var)[T.level]
    import re
    m = re.match(r'C\((\w+)\)\[T\.(.+)\]', t)
    if m:
        level = m.group(2)
        return f'  {level}'
    return t

def fmt_or(row):
    return f"{row['OR']:.2f} ({row['CI_lo']:.2f}–{row['CI_hi']:.2f})"

def fmt_p(p):
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

def build_display_rows(res):
    """Build ordered display rows with reference category placeholders."""
    rows = []
    used = set()

    for group_label, prefix, ref_label in GROUP_ORDER:
        # Find all terms for this variable
        if prefix.startswith('C('):
            mask = res['term'].str.startswith(prefix)
        else:
            mask = res['term'] == prefix

        group_terms = res[mask].copy()
        if group_terms.empty and prefix not in res['term'].values:
            continue

        # Section header row
        rows.append({'label': group_label, 'or_ci': '', 'p_fmt': '', 'is_section': True, 'significant': False})

        # Reference row (for categorical variables)
        if ref_label is not None:
            rows.append({'label': f'  {ref_label} (ref)', 'or_ci': '1.00  (—)', 'p_fmt': 'ref',
                         'is_section': False, 'significant': False, 'is_ref': True})

        # Non-ref terms
        for _, row in group_terms.iterrows():
            rows.append({
                'label':       clean_term(row['term']),
                'or_ci':       fmt_or(row),
                'p_fmt':       fmt_p(row['pval']),
                'is_section':  False,
                'significant': row['pval'] < 0.05,
                'is_ref':      False,
            })
            used.add(row['term'])

    df = pd.DataFrame(rows)
    for col in ['is_ref', 'significant']:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    return df

for res in [res3, res3_sens, res4]:
    res['label'] = res['term'].apply(clean_term)
    res['or_ci'] = res.apply(fmt_or, axis=1)
    res['p_fmt'] = res['pval'].apply(fmt_p)

df_display3      = build_display_rows(res3)
df_display3_sens = build_display_rows(res3_sens)
df_display4      = build_display_rows(res4)

# ── Write Excel ────────────────────────────────────────────────────────────────
print(f"Writing {OUT_PATH} ...")

HEADER_FILL  = PatternFill('solid', fgColor='BA0C2F')
HEADER_FONT  = Font(name='Times New Roman', bold=True, color='FFFFFF', size=11)
BODY_FONT    = Font(name='Times New Roman', size=11)
SECTION_FILL = PatternFill('solid', fgColor='F5D0D6')
SECTION_FONT = Font(name='Times New Roman', bold=True, size=11, color='7A0820')
ALT_FILL     = PatternFill('solid', fgColor='F9ECEE')
SIG_FILL     = PatternFill('solid', fgColor='FFE699')
REF_FILL     = PatternFill('solid', fgColor='F0F0F0')
TITLE_FONT   = Font(name='Times New Roman', bold=True, size=12, color='BA0C2F')
REF_FONT     = Font(name='Times New Roman', italic=True, size=11, color='888888')

wb = openpyxl.Workbook()

for sheet_name, table_num, df_disp, outcome_label, n_outcome in [
    ('Table 3 - Timely Hospice',  'Table 3',      df_display3,      'Timely Hospice Enrollment (>3d before death)', df_model['timely_enrollment'].sum()),
    ('Table 3 sens - Any Hospice','Table 3 sens', df_display3_sens, 'Any Hospice Enrollment (sensitivity)',          df_model['hospice_enrolled'].sum()),
    ('Table 4 - In-Hosp Death',   'Table 4',      df_display4,      'In-Hospital Death',                             df_model['in_hospital_death'].sum()),
]:
    ws = wb.create_sheet(title=sheet_name)

    ws.append([f'{table_num}. Multivariable Logistic Regression: {outcome_label} '
               f'(n events = {int(n_outcome):,} / {len(df_model):,})'])
    ws['A1'].font = TITLE_FONT
    ws.append([])

    hr = ws.max_row + 1
    for ci, cn in enumerate(['Variable', 'OR (95% CI)', 'p-value'], 1):
        cell = ws.cell(row=hr, column=ci, value=cn)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                   wrap_text=True, vertical='center')

    for ri, row in enumerate(df_disp.itertuples(index=False), hr + 1):
        is_sec = row.is_section
        is_ref = getattr(row, 'is_ref', False)
        is_sig = getattr(row, 'significant', False)

        vals = [row.label, row.or_ci, row.p_fmt]
        for ci, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            if is_sec:
                cell.font = SECTION_FONT
                cell.fill = SECTION_FILL
            elif is_ref:
                cell.font = REF_FONT
                cell.fill = REF_FILL
            else:
                cell.font = BODY_FONT
                if is_sig:
                    cell.fill = SIG_FILL
            cell.alignment = Alignment(
                horizontal='left' if ci == 1 else 'center', vertical='center')

    ws.column_dimensions['A'].width = 46
    ws.column_dimensions['B'].width = 24
    ws.column_dimensions['C'].width = 12
    ws.freeze_panes = f'B{hr + 1}'

    footer = ws.max_row + 2
    ws.cell(row=footer, column=1,
            value='OR = odds ratio; CI = 95% confidence interval. '
                  'Highlighted rows (yellow) = p<0.05. '
                  'Reference categories shown in italics. '
                  'Continuous variables interpreted as per-unit change.')
    ws.cell(row=footer, column=1).font = Font(name='Times New Roman', italic=True, size=11, color='555555')

# Remove default empty sheet
if 'Sheet' in wb.sheetnames:
    del wb['Sheet']

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"\nModel 3 (timely, PRIMARY):  pseudo-R2 = {model3.prsquared:.3f}, AIC = {model3.aic:.1f}")
print(f"Model 3 (any hospice, sens): pseudo-R2 = {model3_sens.prsquared:.3f}, AIC = {model3_sens.aic:.1f}")
print(f"Model 4 (in-hospital death): pseudo-R2 = {model4.prsquared:.3f}, AIC = {model4.aic:.1f}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
