"""
make_top10_admissions.py
Top 10 principal diagnoses for inpatient admissions among ICI hospice cohort
(io_analytic, N=2,527).

Sheet 1 (Table 4A): Top 10 by 3-character ICD-10 category — admissions within 30 days of death
Sheet 2 (Table 4B): Top 10 by 3-character ICD-10 category — admissions within 30 days
                    prior to hospice election (among hospice enrollees)

Output: top10_admissions.xlsx
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\tables\top10_admissions.xlsx"

# ── ICD-10 descriptions ────────────────────────────────────────────────────────
DESCRIPTIONS_FULL = {
    "A419":  "Sepsis, unspecified organism",
    "J690":  "Aspiration pneumonitis",
    "J9601": "Acute respiratory failure, unspecified",
    "J189":  "Pneumonia, unspecified organism",
    "C329":  "Malignant neoplasm of larynx, unspecified",
    "J9621": "Acute and chronic respiratory failure with hypoxia",
    "E860":  "Dehydration",
    "U071":  "COVID-19",
    "N179":  "Acute kidney failure, unspecified",
    "G893":  "Neoplasm-related pain",
}

DESCRIPTIONS_3 = {
    "A41": "Sepsis",
    "J69": "Pneumonitis due to solids and liquids (aspiration)",
    "J96": "Respiratory failure",
    "J18": "Pneumonia, unspecified organism",
    "C32": "Malignant neoplasm of larynx",
    "E86": "Volume depletion (dehydration)",
    "U07": "COVID-19",
    "N17": "Acute kidney failure",
    "G89": "Pain, not elsewhere classified",
    "C34": "Malignant neoplasm of bronchus and lung",
    "J44": "Chronic obstructive pulmonary disease",
    "R65": "Systemic inflammatory response / sepsis",
    "I50": "Heart failure",
    "K92": "Other diseases of digestive system",
    "C10": "Malignant neoplasm of oropharynx",
    "C01": "Malignant neoplasm of base of tongue",
    "C78": "Secondary malignant neoplasm of respiratory/digestive organs",
    "C79": "Secondary malignant neoplasm of other sites",
    "J95": "Postprocedural respiratory complications",
    "C02": "Malignant neoplasm of tongue (other/unspecified)",
}

# ── Colors (matches Tables 1/2/3, secular trends, sensitivity) ────────────────
SCARLET     = "BA0C2F"   # primary fill for column headers
WHITE       = "FFFFFF"
SCARLET_D40 = "70071C"   # dark scarlet — used only for separator border accent
ALT_FILL    = "F9ECEE"   # light scarlet tint — alt-row stripe (matches other tables)

THIN_BORDER = Border(
    bottom=Side(style="thin", color="DDDDDD"),
    right=Side(style="thin", color="DDDDDD"),
    left=Side(style="thin", color="DDDDDD"),
)


# ── Query ──────────────────────────────────────────────────────────────────────
print("Querying database...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")

sql_full = """
WITH total AS (
    SELECT COUNT(*) AS total_n
    FROM io_inp_claims i
    JOIN io_analytic a ON i.DSYSRTKY = a.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt
      AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
)
SELECT i.PRNCPAL_DGNS_CD AS icd_code,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / MAX(t.total_n), 1) AS pct
FROM io_inp_claims i
JOIN io_analytic a ON i.DSYSRTKY = a.DSYSRTKY
CROSS JOIN total t
WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt
  AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
GROUP BY i.PRNCPAL_DGNS_CD
ORDER BY n DESC;
"""

sql_3char = """
WITH total AS (
    SELECT COUNT(*) AS total_n
    FROM io_inp_claims i
    JOIN io_analytic a ON i.DSYSRTKY = a.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt
      AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
)
SELECT LEFT(i.PRNCPAL_DGNS_CD, 3) AS icd3,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / MAX(t.total_n), 1) AS pct
FROM io_inp_claims i
JOIN io_analytic a ON i.DSYSRTKY = a.DSYSRTKY
CROSS JOIN total t
WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt
  AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
GROUP BY LEFT(i.PRNCPAL_DGNS_CD, 3)
ORDER BY n DESC
LIMIT 10;
"""

df_full  = con.execute(sql_full).df()
df_3     = con.execute(sql_3char).df()
total_n  = con.execute("""
    SELECT COUNT(*) FROM io_inp_claims i
    JOIN io_analytic a ON i.DSYSRTKY = a.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt
      AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
""").fetchone()[0]

# ── Pre-death admission rates: among full cohort, % with admission in 30/14/7d before death ──
death_summary = con.execute("""
WITH pt_flags AS (
    SELECT
        a.DSYSRTKY,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (a.death_dt - INTERVAL 30 DAY) AND a.death_dt THEN 1 ELSE 0 END) AS d30,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (a.death_dt - INTERVAL 14 DAY) AND a.death_dt THEN 1 ELSE 0 END) AS d14,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (a.death_dt - INTERVAL 7  DAY) AND a.death_dt THEN 1 ELSE 0 END) AS d7
    FROM io_analytic a
    LEFT JOIN io_inp_claims i ON a.DSYSRTKY = i.DSYSRTKY
    GROUP BY a.DSYSRTKY
)
SELECT COUNT(*) AS n_total,
       SUM(d30) AS n_30d,
       SUM(d14) AS n_14d,
       SUM(d7)  AS n_7d
FROM pt_flags
""").fetchone()
n_cohort_total  = death_summary[0]
n_death_30d     = death_summary[1]
n_death_14d     = death_summary[2]
n_death_7d      = death_summary[3]

# ── Pre-hospice admissions: among hospice enrollees, admissions before election ─
print("Querying admissions before hospice election date...")
hospice_summary_sql = """
WITH hospice_pts AS (
    SELECT DSYSRTKY, hospice_election_date
    FROM io_analytic
    WHERE hospice_enrolled = 1 AND hospice_election_date IS NOT NULL
),
pt_flags AS (
    SELECT
        h.DSYSRTKY,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (h.hospice_election_date - INTERVAL 30 DAY)
                          AND h.hospice_election_date THEN 1 ELSE 0 END) AS had_admit_30d,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (h.hospice_election_date - INTERVAL 14 DAY)
                          AND h.hospice_election_date THEN 1 ELSE 0 END) AS had_admit_14d,
        MAX(CASE WHEN TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
                      BETWEEN (h.hospice_election_date - INTERVAL 7 DAY)
                          AND h.hospice_election_date THEN 1 ELSE 0 END) AS had_admit_7d
    FROM hospice_pts h
    LEFT JOIN io_inp_claims i ON h.DSYSRTKY = i.DSYSRTKY
    GROUP BY h.DSYSRTKY
)
SELECT COUNT(*) AS n_hospice,
       SUM(had_admit_30d) AS n_30d,
       SUM(had_admit_14d) AS n_14d,
       SUM(had_admit_7d)  AS n_7d
FROM pt_flags
"""
hosp_summary = con.execute(hospice_summary_sql).fetchone()
n_hospice_total = hosp_summary[0]
n_admit_30d     = hosp_summary[1]
n_admit_14d     = hosp_summary[2]
n_admit_7d      = hosp_summary[3]

hosp_dx_sql = """
WITH hospice_pts AS (
    SELECT DSYSRTKY, hospice_election_date
    FROM io_analytic
    WHERE hospice_enrolled = 1 AND hospice_election_date IS NOT NULL
),
total AS (
    SELECT COUNT(*) AS total_n
    FROM io_inp_claims i
    JOIN hospice_pts h ON i.DSYSRTKY = h.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
          BETWEEN (h.hospice_election_date - INTERVAL 30 DAY) AND h.hospice_election_date
      AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
)
SELECT LEFT(i.PRNCPAL_DGNS_CD, 3) AS icd3,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / MAX(t.total_n), 1) AS pct
FROM io_inp_claims i
JOIN hospice_pts h ON i.DSYSRTKY = h.DSYSRTKY
CROSS JOIN total t
WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
      BETWEEN (h.hospice_election_date - INTERVAL 30 DAY) AND h.hospice_election_date
  AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
GROUP BY LEFT(i.PRNCPAL_DGNS_CD, 3)
ORDER BY n DESC
LIMIT 10
"""
df_hosp_dx   = con.execute(hosp_dx_sql).df()
hosp_total_n = con.execute("""
    SELECT COUNT(*)
    FROM io_inp_claims i
    JOIN (SELECT DSYSRTKY, hospice_election_date FROM io_analytic
          WHERE hospice_enrolled = 1 AND hospice_election_date IS NOT NULL) h
      ON i.DSYSRTKY = h.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
          BETWEEN (h.hospice_election_date - INTERVAL 30 DAY) AND h.hospice_election_date
      AND i.PRNCPAL_DGNS_CD IS NOT NULL AND i.PRNCPAL_DGNS_CD <> ''
""").fetchone()[0]

con.close()

# Enrich hospice-dx codes
df_hosp_dx["description"] = df_hosp_dx["icd3"].map(DESCRIPTIONS_3).fillna("(see ICD-10 reference)")
df_hosp_dx = df_hosp_dx[["description", "n", "pct"]]
df_hosp_dx.insert(0, "rank", range(1, len(df_hosp_dx) + 1))
df_hosp_dx.columns = ["Rank", "Diagnosis", "Admissions, n", "% of all admissions"]

print(f"\nSheet 3 — Admissions before hospice election (N hospice = {n_hospice_total:,}):")
print(f"  Within 30d: {n_admit_30d:,} ({100*n_admit_30d/n_hospice_total:.1f}%)")
print(f"  Within 14d: {n_admit_14d:,} ({100*n_admit_14d/n_hospice_total:.1f}%)")
print(f"  Within  7d: {n_admit_7d:,} ({100*n_admit_7d/n_hospice_total:.1f}%)")
print(df_hosp_dx.to_string(index=False))

# Enrich full codes (kept for console sanity-check; not written to workbook)
df_full["description"] = df_full["icd_code"].map(DESCRIPTIONS_FULL).fillna("(see ICD-10 reference)")
df_full = df_full[["description", "n", "pct"]]
df_full.insert(0, "rank", range(1, len(df_full) + 1))
df_full.columns = ["Rank", "Diagnosis", "Admissions, n", "% of all admissions"]

# Enrich 3-char codes
df_3["description"] = df_3["icd3"].map(DESCRIPTIONS_3).fillna("(see ICD-10 reference)")
df_3 = df_3[["description", "n", "pct"]]
df_3.insert(0, "rank", range(1, len(df_3) + 1))
df_3.columns = ["Rank", "Diagnosis", "Admissions, n", "% of all admissions"]

print("\nSheet 1 — Full ICD-10 codes:")
print(df_full.to_string(index=False))
print("\nSheet 2 — Grouped by 3-character category:")
print(df_3.to_string(index=False))


# ── Helper: build an admissions sheet (identical layout for both 3A and 3B) ──
def build_admissions_sheet(ws, *, title, subtitle, info_rows, dx_df, footnote):
    """Single clean manuscript table: title → subtitle → top-10 diagnosis table → info rows (denominators) → footnote.

    info_rows: list of (label, value) tuples printed below the diagnosis table to show denominators.
    dx_df: dataframe with columns Rank, Diagnosis, Admissions n, % of all admissions.
    """
    ncols = len(dx_df.columns)
    last_col = openpyxl.utils.get_column_letter(ncols)

    # Title (matches Table 1/2/3/secular trends styling: scarlet bold text, no fill)
    ws.merge_cells(f"A1:{last_col}1")
    tc = ws["A1"]
    tc.value = title
    tc.font = Font(name="Times New Roman", bold=True, size=12, color=SCARLET)
    tc.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 24

    # Subtitle
    ws.merge_cells(f"A2:{last_col}2")
    sc = ws["A2"]
    sc.value = subtitle
    sc.font = Font(name="Times New Roman", italic=True, size=11, color="555555")
    sc.alignment = Alignment(horizontal="left")
    ws.row_dimensions[2].height = 18

    # Column header row
    hdr_row = 3
    for col, h in enumerate(dx_df.columns, 1):
        cell = ws.cell(row=hdr_row, column=col, value=h)
        cell.font = Font(name="Times New Roman", bold=True, size=11, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=SCARLET)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=Side(style="thin", color=WHITE),
                             right=Side(style="thin", color=WHITE))
    ws.row_dimensions[hdr_row].height = 20

    # Data rows
    aligns = ["center", "left", "center", "center"]
    for i, row in dx_df.iterrows():
        r = hdr_row + 1 + i
        fill = PatternFill("solid", fgColor=ALT_FILL if i % 2 == 1 else WHITE)
        for col, (val, align) in enumerate(zip(row, aligns), 1):
            if col == ncols:
                val = f"{val}%"
            elif col == ncols - 1:
                val = int(val)
            cell = ws.cell(row=r, column=col, value=val)
            cell.font = Font(name="Times New Roman", size=11)
            cell.fill = fill
            cell.alignment = Alignment(horizontal=align, vertical="center")
            cell.border = THIN_BORDER
        ws.row_dimensions[r].height = 18

    # Info rows (denominators) — placed below the diagnosis table.
    # Continue the alternating stripe pattern from the data rows; mark the
    # boundary with a thick scarlet top border that spans every column.
    info_start = hdr_row + 1 + len(dx_df)
    SEPARATOR_TOP = Side(style="medium", color=SCARLET_D40)
    for idx, (label, val) in enumerate(info_rows):
        r = info_start + idx
        stripe_idx = len(dx_df) + idx
        fill = PatternFill("solid", fgColor=ALT_FILL if stripe_idx % 2 == 1 else WHITE)
        ws.merge_cells(f"A{r}:C{r}")
        c1 = ws.cell(row=r, column=1, value=label)
        c1.font = Font(name="Times New Roman", size=11, bold=True, color="333333")
        c1.fill = fill
        c1.alignment = Alignment(horizontal="left", vertical="center")
        c2 = ws.cell(row=r, column=ncols, value=val)
        c2.font = Font(name="Times New Roman", size=11, bold=True, color="333333")
        c2.fill = fill
        c2.alignment = Alignment(horizontal="right", vertical="center")
        # Apply top separator across ALL columns of the first info row, not just
        # the merge anchor and value cell — otherwise B and C show no border.
        if idx == 0:
            for col in range(1, ncols + 1):
                ws.cell(row=r, column=col).border = Border(top=SEPARATOR_TOP)
        ws.row_dimensions[r].height = 18

    # Footnote
    fn_row = info_start + len(info_rows) + 1
    ws.merge_cells(f"A{fn_row}:{last_col}{fn_row}")
    fn = ws.cell(row=fn_row, column=1, value=footnote)
    fn.font = Font(name="Times New Roman", size=11, italic=True, color="666666")
    fn.alignment = Alignment(wrap_text=True)
    ws.row_dimensions[fn_row].height = 60

    # Column widths
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 22


# ── Build workbook ─────────────────────────────────────────────────────────────
SUBTITLE_DEATH    = f"HNC Patients Receiving ICI (N = {n_cohort_total:,})  |  Medicare 2017–2023"
SUBTITLE_HOSPICE  = f"Among hospice enrollees (n = {n_hospice_total:,})  |  HNC + ICI Patients  |  Medicare 2017–2023"

FOOTNOTE_DEATH = (
    f"The top 10 ICD-10 principal diagnosis categories are shown out of {total_n:,} total admissions "
    f"in the 30-day window. Admissions, n = admissions with that diagnosis; "
    f"% of all admissions = n / {total_n:,} (admissions, not patients — a patient may contribute multiple admissions). "
    f"Categories defined by the first 3 characters of the ICD-10 principal diagnosis code."
)
FOOTNOTE_HOSPICE = (
    f"The top 10 ICD-10 principal diagnosis categories are shown out of {hosp_total_n:,} total admissions "
    f"in the 30-day pre-election window. Admissions, n = admissions with that diagnosis; "
    f"% of all admissions = n / {hosp_total_n:,} (admissions, not patients — a patient may contribute multiple admissions). "
    f"Categories defined by the first 3 characters of the ICD-10 principal diagnosis code."
)

wb = openpyxl.Workbook()

# ── Sheet 1 (Table 3A): admissions in 30 days before death ──────────────────
ws_death = wb.active
ws_death.title = "Before Death"
build_admissions_sheet(
    ws_death,
    title="Supplementary Table 3A. Top 10 Principal Diagnoses for Inpatient Admissions Within 30 Days of Death",
    subtitle=SUBTITLE_DEATH,
    info_rows=[
        ("Total patients in cohort, N",
         f"{n_cohort_total:,}"),
        ("Patients with ≥1 inpatient admission within 30 days of death, n (%)",
         f"{n_death_30d:,} ({100*n_death_30d/n_cohort_total:.1f}%)"),
    ],
    dx_df=df_3,
    footnote=FOOTNOTE_DEATH,
)

# ── Sheet 2 (Table 3B): admissions in 30 days before hospice election ──────
ws_hosp = wb.create_sheet("Before Hospice")
build_admissions_sheet(
    ws_hosp,
    title="Supplementary Table 3B. Top 10 Principal Diagnoses for Inpatient Admissions Within 30 Days of Hospice Election",
    subtitle=SUBTITLE_HOSPICE,
    info_rows=[
        ("Total hospice enrollees, n",
         f"{n_hospice_total:,}"),
        ("Hospice enrollees with ≥1 inpatient admission within 30 days of hospice election, n (%)",
         f"{n_admit_30d:,} ({100*n_admit_30d/n_hospice_total:.1f}%)"),
    ],
    dx_df=df_hosp_dx,
    footnote=FOOTNOTE_HOSPICE,
)

wb.save(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")

from utils import export_xlsx_to_png
FIGURES_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures"
export_xlsx_to_png(OUT_PATH, FIGURES_DIR)
