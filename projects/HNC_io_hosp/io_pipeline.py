"""
IO Hospice Pipeline
Aggressiveness of End-of-Life Care Among Medicare Beneficiaries with HNC Who Received IO

Runs SQL steps 01-12 sequentially, tracks CONSORT counts, exports analytic dataset.
Skips steps whose output table already exists (checkpoint/resume support).

Usage:
    python f:/CMS/projects/HNC_io_hosp/io_pipeline.py
"""

import duckdb
import json
import subprocess
import sys
import time
from pathlib import Path

# Force line-buffered output so progress prints immediately even when piped
sys.stdout.reconfigure(line_buffering=True)

DB_PATH     = r'F:\CMS\cms_data.duckdb'
QUERIES_DIR = Path(r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\queries')
OUT_DIR     = Path(r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp')
PARQUET_OUT = OUT_DIR / 'io_analytic.parquet'
CODES_JSON  = OUT_DIR / 'codes.json'

# Normalize JSON category names to clean DB values
_CAT_NAMES = {}


def build_step4_sql():
    """Generate Step 4 SQL dynamically from codes.json icd10_codes."""
    with open(CODES_JSON, encoding='utf-8') as f:
        raw = json.load(f)['icd10_codes']

    # Normalize names; compute 3-char and 4-char prefix sets per category
    categories = {}
    for cat, codes in raw.items():
        clean = _CAT_NAMES.get(cat, cat)
        p3 = sorted({c for c in codes if len(c) < 4})
        p4 = sorted({c[:4] for c in codes if len(c) >= 4})
        categories[clean] = {'p3': p3, 'p4': p4}

    def conditions(pfx):
        parts = []
        if pfx['p3']:
            parts.append("LEFT(r.dx_prefix, 3) IN ({})".format(
                ', '.join(f"'{p}'" for p in pfx['p3'])))
        if pfx['p4']:
            parts.append("r.dx_prefix IN ({})".format(
                ', '.join(f"'{p}'" for p in pfx['p4'])))
        return ' OR '.join(parts)

    case_lines = '\n'.join(
        f"        WHEN {conditions(pfx)} THEN '{cat}'"
        for cat, pfx in categories.items()
    )

    where_parts = '\n    OR '.join(conditions(pfx) for pfx in categories.values())

    return (
        "-- Step 4: Subsite whitelist generated from PCS_Codes.json\n"
        "SET memory_limit='24GB';\n"
        "SET threads=12;\n"
        "SET temp_directory='F:\\\\CMS\\\\duckdb_temp';\n\n"
        "DROP TABLE IF EXISTS io_subsite;\n\n"
        "CREATE TABLE io_subsite AS\n\n"
        "WITH subsite_counts AS (\n"
        "    SELECT r.DSYSRTKY, r.dx_prefix, COUNT(*) AS cnt\n"
        "    FROM io_hnc_dx_raw r\n"
        "    JOIN io_hnc_confirmed c ON r.DSYSRTKY = c.DSYSRTKY\n"
        "    GROUP BY r.DSYSRTKY, r.dx_prefix\n"
        "),\n\n"
        "ranked AS (\n"
        "    SELECT DSYSRTKY, dx_prefix, cnt,\n"
        "           ROW_NUMBER() OVER (PARTITION BY DSYSRTKY ORDER BY cnt DESC, dx_prefix ASC) AS rn\n"
        "    FROM subsite_counts\n"
        ")\n\n"
        "SELECT\n"
        "    r.DSYSRTKY,\n"
        "    r.dx_prefix AS predominant_subsite,\n"
        "    CASE\n"
        f"{case_lines}\n"
        "        ELSE NULL\n"
        "    END AS subsite_category\n"
        "FROM ranked r\n"
        "WHERE r.rn = 1\n"
        "  AND (\n"
        f"    {where_parts}\n"
        "  );"
    )

def build_step7_pcs_list():
    """Return a SQL IN-list string of all PCS codes from codes.json pcs_codes."""
    with open(CODES_JSON, encoding='utf-8') as f:
        pcs = json.load(f)['pcs_codes']
    all_codes = sorted({code for codes in pcs.values() for code in codes})
    return ', '.join(f"'{c}'" for c in all_codes)


STEPS = [
    ("01_io_decedents.sql",     "io_decedents",     "Steps 1-2: Decedents age >=66"),
    ("02_io_hnc_dx.sql",        "io_hnc_dx_raw",    "Step 3a: HNC dx raw"),
    ("03_io_hnc_confirmed.sql", "io_hnc_confirmed",  "Step 3b: >=2 HNC claims on separate dates"),
    ("04_io_subsite.sql",       "io_subsite",        "Step 4: Eligible subsites"),
    ("05_io_claims.sql",        "io_claims_raw",     "Step 5: IO claims (J9271/J9299)"),
    ("06_io_episodes.sql",      "io_episodes",       "Step 6: IO episodes (gap >=120d)"),
    ("07a_io_staging.sql",      "io_hosp_claims",    "Step 7a: Stage large tables for 7K HNC+IO patients"),
    ("07_io_curative.sql",      "io_curative",       "Step 7: Curative therapy before last IO episode"),
    ("08_io_enrollment.sql",    "io_ffs_eligible",   "Steps 9-11: ESRD/Geo/FFS exclusions"),
    ("09_io_cohort_final.sql",  "io_cohort",         "Final cohort (>=180d dx-to-IO)"),
    ("10_io_elixhauser.sql",    "io_comorbidity",    "Comorbidity: Elixhauser/van Walraven"),
    ("11_io_hospice.sql",          "io_outcomes",             "Outcomes: hospice + in-hospital death"),
    ("12a_io_sensitivity.sql",    "io_sensitivity_vars",     "Sensitivity: median interdose interval"),
    ("12_io_analytic.sql",        "io_analytic",             "Analytic dataset (primary cohort)"),
    ("09b_io_cohort_itc.sql",     "io_cohort_itc",           "ITC cohort (no curative therapy req.)"),
    ("12b_io_sensitivity_itc.sql","io_sensitivity_vars_itc", "Sensitivity ITC: median interdose interval"),
    ("10b_io_comorbidity_itc.sql","io_comorbidity_itc",      "ITC comorbidity: Elixhauser/van Walraven"),
    ("11b_io_outcomes_itc.sql",   "io_outcomes_itc",         "ITC outcomes: hospice + in-hospital death"),
    ("12c_io_analytic_itc.sql",   "io_analytic_itc",         "Analytic dataset (ITC cohort)"),
]


def table_exists(con, table_name):
    n = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name]
    ).fetchone()[0]
    return n > 0


def run_pipeline():
    print(f"\n{'='*70}")
    print("IO HOSPICE PIPELINE")
    print(f"Database: {DB_PATH}")
    print(f"{'='*70}\n")

    con = duckdb.connect(DB_PATH)

    consort = []
    total_start = time.time()

    for sql_file, table_name, description in STEPS:
        sql_path = QUERIES_DIR / sql_file
        if not sql_path.exists():
            print(f"[ERROR] File not found: {sql_path}")
            sys.exit(1)

        # Checkpoint: skip if table already exists
        if table_exists(con, table_name):
            n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            consort.append((description, table_name, n))
            print(f"[SKIP] {description}")
            print(f"  -> {table_name}: {n:,} rows (already exists)\n")
            continue

        print(f"Running: {description}")
        print(f"  File: {sql_file}")

        step_start = time.time()
        if sql_file == '04_io_subsite.sql' and CODES_JSON.exists():
            sql = build_step4_sql()
        elif sql_file == '07_io_curative.sql' and CODES_JSON.exists():
            sql = sql_path.read_text(encoding='utf-8')
            sql = sql.replace('{INPATIENT_PCS_WHERE}', build_step7_pcs_list())
        else:
            sql = sql_path.read_text(encoding='utf-8')
        con.execute(sql)
        elapsed = time.time() - step_start

        n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        consort.append((description, table_name, n))
        print(f"  -> {table_name}: {n:,} rows  ({elapsed:.1f}s)\n")

    # Export analytic dataset
    print(f"Exporting io_analytic to {PARQUET_OUT} ...")
    con.execute(f"COPY io_analytic TO '{PARQUET_OUT}' (FORMAT PARQUET)")
    n_analytic = con.execute("SELECT COUNT(*) FROM io_analytic").fetchone()[0]
    print(f"  -> Exported {n_analytic:,} rows\n")

    con.close()

    total_elapsed = time.time() - total_start

    # Print CONSORT summary
    print(f"\n{'='*70}")
    print("CONSORT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Step':<55} {'Table':<25} {'N':>12}")
    print(f"{'-'*55} {'-'*25} {'-'*12}")
    for desc, tbl, n in consort:
        print(f"{desc:<55} {tbl:<25} {n:>12,}")
    print(f"\nTotal runtime: {total_elapsed:.0f}s")
    print(f"Output: {PARQUET_OUT}")
    print(f"{'='*70}\n")

    # Generate all figures and tables
    analysis_scripts = [
        'make_table1.py',
        'make_table2.py',
        'make_regression.py',
        'make_secular_trends.py',
        'make_figures.py',
        'make_flowchart.py',
        'make_top10_admissions.py',
        'make_sensitivity.py',
    ]

    print(f"\n{'='*70}")
    print("GENERATING FIGURES AND TABLES")
    print(f"{'='*70}")
    for script in analysis_scripts:
        script_path = OUT_DIR / script
        print(f"Running: {script}")
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  -> OK")
        else:
            print(f"  -> ERROR:\n{result.stderr.strip()}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_pipeline()
