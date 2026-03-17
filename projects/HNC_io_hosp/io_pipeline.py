"""
IO Hospice Pipeline
Aggressiveness of End-of-Life Care Among Medicare Beneficiaries with HNC Who Received IO

Runs SQL steps 01-12 sequentially, tracks CONSORT counts, exports analytic dataset.
Skips steps whose output table already exists (checkpoint/resume support).

Usage:
    python f:/CMS/projects/HNC_io_hosp/io_pipeline.py
"""

import duckdb
import sys
import time
from pathlib import Path

# Force line-buffered output so progress prints immediately even when piped
sys.stdout.reconfigure(line_buffering=True)

DB_PATH     = r'F:\CMS\cms_data.duckdb'
QUERIES_DIR = Path(r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\queries')
OUT_DIR     = Path(r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp')
PARQUET_OUT = OUT_DIR / 'io_analytic.parquet'

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
    ("11_io_hospice.sql",       "io_outcomes",       "Outcomes: hospice + in-hospital death"),
    ("12_io_analytic.sql",      "io_analytic",       "Analytic dataset"),
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


if __name__ == '__main__':
    run_pipeline()
