"""
make_sensitivity.py
Supplementary analyses: two parallel comparisons.

A) Regimen-adjusted IO discontinuation timing (io_analytic — primary cohort)
   Sheet 1 — Timing comparison: fixed vs regimen-adjusted, by estimated regimen
   Sheet 2 — Outcomes by regimen: hospice + in-hospital death rates
   Sheet 3 — Reclassified patients: those outside 30d window but within regimen window

B) Broadened cohort analysis (io_analytic_itc)
   Sheet 4 — Primary vs broadened cohort: characteristics and outcomes with SMDs

Output: C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/sensitivity_analysis.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

import os
DB_PATH  = os.environ.get('CMS_DB', r"F:\CMS\cms_data.duckdb")
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\sensitivity_analysis.xlsx"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading io_analytic and io_analytic_itc...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT
        hospice_enrolled, in_hospital_death,
        days_last_io_to_death,
        io_within_14d_of_death, io_within_30d_of_death,
        median_interdose_days, estimated_regimen,
        discontinued_threshold_days, io_within_regimen_window,
        days_past_expected_dose_to_death,
        hospice_los_days, hospice_short_stay,
        age_cat, sex, subsite_category, io_agent
    FROM io_analytic
""").df()
df_itc = con.execute("""
    SELECT
        hospice_enrolled, in_hospital_death,
        days_last_io_to_death,
        io_within_14d_of_death, io_within_30d_of_death,
        io_within_regimen_window,
        days_past_expected_dose_to_death,
        had_prior_curative_therapy,
        hospice_los_days, hospice_short_stay,
        age_cat, sex, subsite_category, io_agent
    FROM io_analytic_itc
""").df()
con.close()

for col in ['hospice_enrolled','in_hospital_death','io_within_14d_of_death',
            'io_within_30d_of_death','io_within_regimen_window']:
    df[col] = df[col].fillna(0).astype(int)
df['estimated_regimen'] = df['estimated_regimen'].fillna('single-dose')

for col in ['hospice_enrolled','in_hospital_death','io_within_14d_of_death',
            'io_within_30d_of_death','io_within_regimen_window',
            'had_prior_curative_therapy','hospice_los_days','hospice_short_stay']:
    if col in df_itc.columns:
        df_itc[col] = df_itc[col].fillna(0)
for col in ['hospice_enrolled','in_hospital_death','io_within_14d_of_death',
            'io_within_30d_of_death','io_within_regimen_window',
            'had_prior_curative_therapy','hospice_short_stay']:
    df_itc[col] = df_itc[col].astype(int)

N    = len(df)
N_itc = len(df_itc)
print(f"  Primary cohort N = {N:,}")
print(f"  ITC cohort N     = {N_itc:,}")
print(f"  Regimen distribution:\n{df['estimated_regimen'].value_counts()}")

# ── Helpers ────────────────────────────────────────────────────────────────────
REGIMEN_ORDER = ['q2w', 'q3w', 'q4w', 'q6w', 'single-dose', 'other']
REGIMEN_LABELS = {
    'q2w':         'Every 2 weeks (q2w)',
    'q3w':         'Every 3 weeks (q3w)',
    'q4w':         'Every 4 weeks (q4w)',
    'q6w':         'Every 6 weeks (q6w)',
    'single-dose': 'Single dose only',
    'other':       'Other / irregular',
}

def med_iqr(series):
    s = series.dropna()
    if len(s) == 0:
        return '—'
    return f"{s.median():.0f} ({s.quantile(0.25):.0f}–{s.quantile(0.75):.0f})"

def n_pct(n, total):
    if total == 0:
        return '—'
    return f"{n:,} ({100.0 * n / total:.1f}%)"

# ── Sheet 1: Timing comparison ─────────────────────────────────────────────────
print("Building Sheet 1: Timing comparison...")

s1_rows = []

def s1_row(label, sub):
    n         = len(sub)
    mdn       = med_iqr(sub['days_last_io_to_death'])
    w14       = n_pct(sub['io_within_14d_of_death'].sum(), n)
    w30       = n_pct(sub['io_within_30d_of_death'].sum(), n)
    wreg      = n_pct(sub['io_within_regimen_window'].sum(), n)
    med_idose = med_iqr(sub['median_interdose_days'])
    is_single = sub['estimated_regimen'].eq('single-dose').all() if len(sub) > 0 else False
    mdn_past  = '—' if is_single else med_iqr(sub['days_past_expected_dose_to_death'])
    wreg_val  = '—' if is_single else wreg
    s1_rows.append({
        'Estimated Dosing Regimen':                                    label,
        'N':                                                           f"{n:,}",
        'Median time between ICI doses,\ndays (IQR)':                    med_idose,
        'Days from last ICI dose to death,\nmedian (IQR)':               mdn,
        'Days from next expected dose to death,\nmedian (IQR)†':        mdn_past,
        'Last ICI dose within 14 days of death,\nn (%)':                 w14,
        'Last ICI dose within 30 days of death,\nn (%)':                 w30,
        'Last ICI dose within regimen-adjusted window‡,\nn (%)':         wreg_val,
    })

s1_row('Overall', df)
for reg in REGIMEN_ORDER:
    sub = df[df['estimated_regimen'] == reg]
    if len(sub) > 0:
        s1_row(f'  {REGIMEN_LABELS[reg]}', sub)

df_s1 = pd.DataFrame(s1_rows)

# ── Sheet 2: Outcomes by regimen ───────────────────────────────────────────────
print("Building Sheet 2: Outcomes by regimen...")

s2_rows = []

def s2_row(label, sub):
    n    = len(sub)
    hosp = n_pct(sub['hospice_enrolled'].sum(), n)
    ihd  = n_pct(sub['in_hospital_death'].sum(), n)
    s2_rows.append({
        'Estimated Dosing Regimen':  label,
        'N':                         f"{n:,}",
        'Enrolled in hospice, n (%)': hosp,
        'Died in hospital, n (%)':   ihd,
    })

s2_row('Overall', df)
for reg in REGIMEN_ORDER:
    sub = df[df['estimated_regimen'] == reg]
    if len(sub) > 0:
        s2_row(f'  {REGIMEN_LABELS[reg]}', sub)

df_s2 = pd.DataFrame(s2_rows)

# ── Sheet 3: Reclassified patients ─────────────────────────────────────────────
print("Building Sheet 3: Reclassified patients...")

# Patients outside 30d window but inside their regimen window
reclassified = df[(df['io_within_30d_of_death'] == 0) & (df['io_within_regimen_window'] == 1)]
Nr = len(reclassified)
print(f"  Reclassified N = {Nr:,} ({100.0*Nr/N:.1f}% of cohort)")

COL_GROUP = 'Subgroup'
COL_HOSP  = 'Hospice enrolled,\nn (%)'
COL_IHD   = 'In-hospital death,\nn (%)'
COL_DAYS  = 'Last ICI to death,\nmedian days (IQR)'
COL_PAST  = 'Days past expected dose at death,\nmedian (IQR)†'
COL_NOTE  = 'Classification'

s3_rows = []
n_outside_30d  = (df['io_within_30d_of_death'] == 0).sum()
n_confirmed_dc = n_outside_30d - Nr
df_gt30        = df[df['io_within_30d_of_death'] == 0]
df_confirmed   = df[(df['io_within_30d_of_death'] == 0) & (df['io_within_regimen_window'] == 0)]

s3_rows.append({
    COL_GROUP: 'ICI >30 days before death',
    'N':       n_outside_30d,
    COL_HOSP:  n_pct(df_gt30['hospice_enrolled'].sum(), n_outside_30d),
    COL_IHD:   n_pct(df_gt30['in_hospital_death'].sum(), n_outside_30d),
    COL_DAYS:  med_iqr(df_gt30['days_last_io_to_death']),
    COL_PAST:  med_iqr(df_gt30['days_past_expected_dose_to_death']),
    COL_NOTE:  'ICI discontinued (primary definition)',
})
s3_rows.append({
    COL_GROUP: '  Died before next expected dose†',
    'N':       Nr,
    COL_HOSP:  n_pct(reclassified['hospice_enrolled'].sum(), Nr),
    COL_IHD:   n_pct(reclassified['in_hospital_death'].sum(), Nr),
    COL_DAYS:  med_iqr(reclassified['days_last_io_to_death']),
    COL_PAST:  med_iqr(reclassified['days_past_expected_dose_to_death']),
    COL_NOTE:  'Reclassified as on-treatment',
})
s3_rows.append({
    COL_GROUP: '  Died after next expected dose',
    'N':       n_confirmed_dc,
    COL_HOSP:  n_pct(df_confirmed['hospice_enrolled'].sum(), n_confirmed_dc),
    COL_IHD:   n_pct(df_confirmed['in_hospital_death'].sum(), n_confirmed_dc),
    COL_DAYS:  med_iqr(df_confirmed['days_last_io_to_death']),
    COL_PAST:  med_iqr(df_confirmed['days_past_expected_dose_to_death']),
    COL_NOTE:  'Confirmed discontinuation',
})
for reg in REGIMEN_ORDER:
    sub = reclassified[reclassified['estimated_regimen'] == reg]
    if len(sub) > 0:
        s3_rows.append({
            COL_GROUP: f'    {REGIMEN_LABELS[reg]}',
            'N':        len(sub),
            COL_HOSP:  n_pct(sub['hospice_enrolled'].sum(), len(sub)),
            COL_IHD:   n_pct(sub['in_hospital_death'].sum(), len(sub)),
            COL_DAYS:  med_iqr(sub['days_last_io_to_death']),
            COL_PAST:  med_iqr(sub['days_past_expected_dose_to_death']),
            COL_NOTE:  '',
        })

df_s3 = pd.DataFrame(s3_rows)

# ── Statistical helpers ───────────────────────────────────────────────────────
def fmt_smd(d):
    if pd.isna(d): return '—'
    return f'{d:.3f}'

def smd_binary(s1, s2, val=1):
    # (p1 - p2) / sqrt((p1(1-p1) + p2(1-p2)) / 2)
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0: return fmt_smd(float('nan'))
    p1 = (s1 == val).sum() / n1
    p2 = (s2 == val).sum() / n2
    pooled = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    if pooled == 0: return fmt_smd(0.0)
    return fmt_smd((p1 - p2) / (pooled ** 0.5))

def smd_continuous(s1, s2):
    a, b = s1.dropna(), s2.dropna()
    if len(a) == 0 or len(b) == 0: return fmt_smd(float('nan'))
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = (v1 + v2) / 2
    if pooled == 0: return fmt_smd(0.0)
    return fmt_smd((m1 - m2) / (pooled ** 0.5))

def smd_categorical(s1, s2):
    # Multivariate SMD across category proportions (Yang & Dalton, 2012)
    cats = sorted(set(s1.dropna()) | set(s2.dropna()))
    if len(cats) < 2: return fmt_smd(0.0)
    import numpy as np
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0: return fmt_smd(float('nan'))
    # Drop last category (linearly dependent)
    p1 = np.array([(s1 == c).sum() / n1 for c in cats[:-1]])
    p2 = np.array([(s2 == c).sum() / n2 for c in cats[:-1]])
    k = len(p1)
    # Pooled covariance matrix
    S1 = np.diag(p1) - np.outer(p1, p1)
    S2 = np.diag(p2) - np.outer(p2, p2)
    S  = (S1 + S2) / 2
    try:
        diff = p1 - p2
        d = float((diff @ np.linalg.pinv(S) @ diff) ** 0.5)
        return fmt_smd(d)
    except Exception:
        return fmt_smd(float('nan'))

# ── Sheet 4: Primary vs ITC comparison ────────────────────────────────────────
print("Building Sheet 4: Primary vs ITC comparison...")

P4_COL = 'Primary Cohort'
I4_COL = 'Broadened Cohort'

s4_rows = []

def s4_add(label, p_val, i_val, smd=''):
    s4_rows.append({'Characteristic': label, P4_COL: p_val, I4_COL: i_val, 'SMD': smd})

s4_add('Total patients, N', f'{N:,}', f'{N_itc:,}')
s4_add('OUTCOMES', '', '')
s4_add('  Enrolled in hospice, n (%)',
       n_pct(df['hospice_enrolled'].sum(), N),
       n_pct(df_itc['hospice_enrolled'].sum(), N_itc),
       smd_binary(df['hospice_enrolled'], df_itc['hospice_enrolled']))
s4_add('  Hospice LOS among enrolled, median days (IQR)',
       med_iqr(df['hospice_los_days'][df['hospice_enrolled']==1]),
       med_iqr(df_itc['hospice_los_days'][df_itc['hospice_enrolled']==1]),
       smd_continuous(df['hospice_los_days'][df['hospice_enrolled']==1],
                      df_itc['hospice_los_days'][df_itc['hospice_enrolled']==1]))
s4_add('  Died in hospital, n (%)',
       n_pct(df['in_hospital_death'].sum(), N),
       n_pct(df_itc['in_hospital_death'].sum(), N_itc),
       smd_binary(df['in_hospital_death'], df_itc['in_hospital_death']))
s4_add('  Days from last ICI dose to death, median (IQR)',
       med_iqr(df['days_last_io_to_death']),
       med_iqr(df_itc['days_last_io_to_death']),
       smd_continuous(df['days_last_io_to_death'], df_itc['days_last_io_to_death']))
s4_add('  Last ICI dose within 14 days of death, n (%)',
       n_pct(df['io_within_14d_of_death'].sum(), N),
       n_pct(df_itc['io_within_14d_of_death'].sum(), N_itc),
       smd_binary(df['io_within_14d_of_death'], df_itc['io_within_14d_of_death']))
s4_add('  Last ICI dose within 30 days of death, n (%)',
       n_pct(df['io_within_30d_of_death'].sum(), N),
       n_pct(df_itc['io_within_30d_of_death'].sum(), N_itc),
       smd_binary(df['io_within_30d_of_death'], df_itc['io_within_30d_of_death']))
s4_add('  Last ICI dose within regimen-adjusted window‡, n (%)',
       n_pct(df['io_within_regimen_window'].sum(), N),
       n_pct(df_itc['io_within_regimen_window'].sum(), N_itc),
       smd_binary(df['io_within_regimen_window'], df_itc['io_within_regimen_window']))
s4_add('  Days from next expected dose to death, median (IQR)†',
       med_iqr(df['days_past_expected_dose_to_death']),
       med_iqr(df_itc['days_past_expected_dose_to_death']),
       smd_continuous(df['days_past_expected_dose_to_death'], df_itc['days_past_expected_dose_to_death']))
s4_add('DEMOGRAPHICS', '', '')
s4_add('  Age, n (%)', '', '',
       smd_categorical(df['age_cat'], df_itc['age_cat']))
for cat in ['66-69','70-74','75-79','80-84','85+']:
    s4_add(f'    {cat}',
           n_pct((df['age_cat']==cat).sum(), N),
           n_pct((df_itc['age_cat']==cat).sum(), N_itc))
s4_add('  Male sex, n (%)',
       n_pct((df['sex']=='Male').sum(), N),
       n_pct((df_itc['sex']=='Male').sum(), N_itc),
       smd_binary(df['sex'] == 'Male', df_itc['sex'] == 'Male'))
s4_add('TUMOR SUBSITE', '', '',
       smd_categorical(df['subsite_category'], df_itc['subsite_category']))
all_subsites_s4 = sorted(set(df['subsite_category'].dropna()) |
                         set(df_itc['subsite_category'].dropna()))
for sub in all_subsites_s4:
    s4_add(f'  {sub}, n (%)',
           n_pct((df['subsite_category']==sub).sum(), N),
           n_pct((df_itc['subsite_category']==sub).sum(), N_itc))
s4_add('ICI AGENT', '', '',
       smd_categorical(df['io_agent'], df_itc['io_agent']))
all_agents_s4 = sorted(set(df['io_agent'].dropna()) |
                       set(df_itc['io_agent'].dropna()))
for ag in all_agents_s4:
    s4_add(f'  {ag}, n (%)',
           n_pct((df['io_agent']==ag).sum(), N),
           n_pct((df_itc['io_agent']==ag).sum(), N_itc))

df_s4 = pd.DataFrame(s4_rows)


# ── Excel styling ──────────────────────────────────────────────────────────────
HEADER_FILL  = PatternFill('solid', fgColor='BA0C2F')
HEADER_FONT  = Font(name='Times New Roman', bold=True, color='FFFFFF', size=11)
BODY_FONT    = Font(name='Times New Roman', size=11)
SECTION_FILL = PatternFill('solid', fgColor='F5D0D6')
SECTION_FONT = Font(name='Times New Roman', bold=True, size=11, color='7A0820')
ALT_FILL     = PatternFill('solid', fgColor='F9ECEE')
TITLE_FONT   = Font(name='Times New Roman', bold=True, size=12, color='BA0C2F')
SUB_FONT     = Font(name='Times New Roman', bold=True, size=11, color='70071C')
NOTE_FONT    = Font(name='Times New Roman', italic=True, size=11, color='555555')


def write_section(ws, title, df_data, col_widths, footnote=None,
                  start_row=1, is_sub=False, title_size=None, body_size=None):
    """Write a titled table section to ws at start_row. Returns next available row.

    title_size / body_size override the module-level font sizes for this section.
    """
    # Resolve fonts for this section (default to module-level constants)
    if title_size is not None:
        title_font_use = Font(name='Times New Roman', bold=True, size=title_size,
                              color='70071C' if is_sub else 'BA0C2F')
    else:
        title_font_use = SUB_FONT if is_sub else TITLE_FONT
    if body_size is not None:
        header_font_use  = Font(name='Times New Roman', bold=True, color='FFFFFF', size=body_size)
        section_font_use = Font(name='Times New Roman', bold=True, size=body_size, color='7A0820')
        body_font_use    = Font(name='Times New Roman', size=body_size)
        note_font_use    = Font(name='Times New Roman', italic=True, size=body_size, color='555555')
    else:
        header_font_use, section_font_use = HEADER_FONT, SECTION_FONT
        body_font_use, note_font_use      = BODY_FONT, NOTE_FONT

    r = start_row

    # Title
    cell = ws.cell(row=r, column=1, value=title)
    cell.font = title_font_use
    ws.row_dimensions[r].height = max(22, (title_size or 13) + 8)
    r += 2  # blank line after title

    # Header row
    for ci, col in enumerate(df_data.columns, 1):
        cell = ws.cell(row=r, column=ci, value=col)
        cell.font      = header_font_use
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal='left' if ci == 1 else 'center',
                                   wrap_text=True, vertical='center')
    ws.row_dimensions[r].height = 56
    r += 1

    # Data rows
    alt = 0
    for row_vals in df_data.itertuples(index=False):
        first_val    = str(row_vals[0])
        # Require ≥3 chars for a section header so single-letter labels (e.g., "N") don't match.
        is_overall   = first_val.strip().lower() in ('overall', 'total patients, n')
        is_section_r = (len(first_val.strip()) >= 3
                        and first_val.strip().isupper()
                        and not first_val.startswith(' '))
        is_dash      = first_val.strip().startswith('—')
        is_indent    = first_val.startswith('  ')
        # Determine row-level fill before the column loop
        if is_section_r or is_dash or is_overall:
            alt = 0
            row_fill = SECTION_FILL
            row_font = section_font_use
        else:
            # Both indented and non-indented data rows alternate (used by transposed S1A/S1B
            # which have flat metric labels with no leading spaces).
            alt += 1
            row_fill = ALT_FILL if alt % 2 == 0 else None
            row_font = body_font_use
        for ci, val in enumerate(row_vals, 1):
            cell = ws.cell(row=r, column=ci, value=val)
            if row_font:
                cell.font = row_font
            if row_fill:
                cell.fill = row_fill
            cell.alignment = Alignment(
                horizontal='left' if ci == 1 else 'center',
                vertical='center', wrap_text=(ci == 1))
        r += 1

    # Footnote
    if footnote:
        r += 1
        ws.cell(row=r, column=1, value=footnote).font = note_font_use
        r += 1

    # Column widths: take max of existing
    for ci, width in enumerate(col_widths, 1):
        letter   = openpyxl.utils.get_column_letter(ci)
        existing = ws.column_dimensions[letter].width
        ws.column_dimensions[letter].width = max(existing or 0, width)

    return r + 2  # leave 2-row gap before next section


# ── Merge A1 + A2 → combined timing + outcomes table ──────────────────────────
df_s12 = df_s1.merge(df_s2.drop(columns=['N']), on='Estimated Dosing Regimen', how='left')

# S1A: transpose for manuscript width — metrics → rows, regimen categories → columns
df_s12 = (
    df_s12.set_index('Estimated Dosing Regimen')
          .T
          .reset_index()
          .rename(columns={'index': 'Metric'})
)
# S1B: drop columns redundant with row label (Classification text duplicates the row group;
# "Days past expected dose" is too granular for a manuscript supp). Keep 5 columns.
df_s3 = df_s3[[COL_GROUP, 'N', COL_HOSP, COL_IHD, COL_DAYS]]

FOOTNOTE_REGIMEN = (
    'Estimated dosing regimen classified by patient-level median interdose interval (days): '
    'q2w = 11-17, q3w = 18-24, q4w = 25-31, q6w = 39-45. '
    'Patients outside these ranges classified as other/irregular. '
    'Single-dose patients had only one ICI claim in their final episode. '
    '† Days from next expected dose to death = days from last ICI dose to death minus median interdose interval. '
    'Negative values indicate the patient died before their next dose was due (still on schedule); '
    'positive values indicate the patient had already missed at least one dose. '
    'Single-dose patients use a default interval of 42 days. '
    '‡ Regimen-adjusted window = median interdose interval + 14 days tolerance. '
    'A patient within this window likely died before their next scheduled infusion.'
)
FOOTNOTE_ITC = (
    f'Primary Cohort (N={N:,}): HNC + ICI patients with documented prior curative therapy '
    f'and ≥180 days from diagnosis to ICI start. '
    f'Broadened Cohort (N={N_itc:,}): all HNC + ICI patients meeting eligibility criteria '
    f'(FFS, geography, ESRD) without curative therapy or timing restrictions. '
    f'The broadened cohort is a superset that includes all primary cohort patients. '
    f'SMD = standardized mean difference; |SMD| > 0.1 typically indicates meaningful imbalance.'
 )

print(f"Writing {OUT_PATH} ...")
wb = openpyxl.Workbook()
wb.remove(wb.active)

# ── Supplement 1: Regimen-adjusted timing sensitivity ─────────────────────────
ws1 = wb.create_sheet(title='Supplement 1 — Regimen')
r = write_section(
    ws1,
    f'Supplementary Table 1A. ICI Timing and Outcomes by Estimated Dosing Regimen — Primary Cohort (N = {N:,})',
    df_s12,
    [50, 16, 16, 16, 16, 16, 16, 16],
    footnote=None,
    start_row=1,
    title_size=18,
    body_size=14,
)
r = write_section(
    ws1,
    'Supplementary Table 1B. Patients Reclassified by Regimen-Adjusted Timing Definition',
    df_s3,
    [60, 8, 22, 22, 28],
    footnote=FOOTNOTE_REGIMEN,
    start_row=r,
    is_sub=True,
    title_size=18,
    body_size=14,
)
ws1.freeze_panes = 'B1'

# ── Table 5: Broadened cohort analysis (main manuscript table) ────────────────
ws2 = wb.create_sheet(title='Table 5 — Broadened Cohort')
r = write_section(
    ws2,
    f'Table 5. Primary cohort vs broadened cohort — patient characteristics and outcomes',
    df_s4,
    [52, 24, 24, 12],
    footnote=FOOTNOTE_ITC,
    start_row=1,
)
ws2.freeze_panes = 'B1'

wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
