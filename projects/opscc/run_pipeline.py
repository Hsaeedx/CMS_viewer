"""
run_pipeline.py
Runs the OPSCC two-comparison pipeline in order.

Study design:
  Comparison A — TORS alone  vs  RT alone   (surgery vs radiation monotherapy)
  Comparison B — TORS + RT   vs  CT/CRT     (adjuvant-RT surgery vs chemoradiation)

  SQL steps (run via DuckDB connection):
    1. hnc_dx_raw.sql        — Raw HNC diagnosis claims across all sources
    2. hnc_confirmed.sql     — HNC confirmation (≥1 inpatient OR ≥2 outpatient ≥30d apart)
    3. hnc_universe.sql      — Universe of confirmed HNC patients
    4. opscc_universe.sql    — Filter to OPSCC primaries (C01, C09, C10, C14)
    5. opscc_cohort.sql      — Apply FFS enrollment criterion (6 continuous FFS months)
    6. opscc_ctcrt.sql       — Annotate cohort: TORS date, chemo date, RT date, metastatic flag
    7. opscc_elixhauser.sql  — Elixhauser comorbidity flags + van Walraven score
    8. opscc_propensity.sql  — Analytic dataset with 4 tx_groups, demographics, comorbidities
    9. opscc_outcomes.sql    — Dysphagia / G-tube / tracheostomy outcomes
   10. opscc_survival.sql    — Survival table: death date + Dec-31 censor per patient
   11. opscc_ffs_dates.sql   — FFS dropout date: last continuous Part A+B non-HMO month

  Python steps (run as subprocesses):
   12. iptw_analysis.py      — Two independent 1:1 PSMs; writes psm_matched_A/B to opscc_propensity
   13. survival_analysis.py  — Overall KM + Cox PH (both comparisons)
   14. subgroup_analysis.py  — Subgroup KM + Cox (age <75 / ≥75, Elixhauser tertile)
   15. outcomes_analysis.py  — Odds ratios: dysphagia, G-tube, trach (6mo/1yr/3yr/5yr/any)
   16. make_table1.py        — Table 1 demographics → table1_psm.xlsx
   17. make_outcomes_table.py — OR table export → outcomes_tables.xlsx
   18. make_figures.py       — KM + forest plots for both comparisons → PNG files
   19. make_flowchart.py     — Cohort flowchart → cohort_flowchart.png

Pass a start step number to resume from a specific step, e.g.:
    python run_pipeline.py 10
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up: opscc -> projects -> root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DB_PATH     = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
CMS_DIR     = Path(os.getenv("CMS_directory", ""))
PROJECT_DIR = Path(__file__).resolve().parent
QUERIES_DIR = PROJECT_DIR / "queries"

STEPS_SQL = {
    1:  ("1  - HNC Dx Raw",        QUERIES_DIR / "hnc_dx_raw.sql"),
    2:  ("2  - HNC Confirmed",     QUERIES_DIR / "hnc_confirmed.sql"),
    3:  ("3  - HNC Universe",      QUERIES_DIR / "hnc_universe.sql"),
    4:  ("4  - OPSCC Universe",    QUERIES_DIR / "opscc_universe.sql"),
    5:  ("5  - Cohort",            QUERIES_DIR / "opscc_cohort.sql"),
    6:  ("6  - TORS/CT-CRT Dates", QUERIES_DIR / "opscc_ctcrt.sql"),
    7:  ("7  - Elixhauser",        QUERIES_DIR / "opscc_elixhauser.sql"),
    8:  ("8  - Propensity",        QUERIES_DIR / "opscc_propensity.sql"),
    9:  ("9  - Outcomes",          QUERIES_DIR / "opscc_outcomes.sql"),
    10: ("10 - Survival Table",    QUERIES_DIR / "opscc_survival.sql"),
    11: ("11 - FFS Dates",         QUERIES_DIR / "opscc_ffs_dates.sql"),
}

STEPS_PY = {
    12: ("12 - PSM Matching",       PROJECT_DIR / "iptw_analysis.py"),
    13: ("13 - Survival Analysis",  PROJECT_DIR / "survival_analysis.py"),
    14: ("14 - Subgroup Survival",  PROJECT_DIR / "subgroup_analysis.py"),
    15: ("15 - Outcomes Analysis",  PROJECT_DIR / "outcomes_analysis.py"),
    16: ("16 - Table 1",            PROJECT_DIR / "make_table1.py"),
    17: ("17 - Outcomes Table",     PROJECT_DIR / "make_outcomes_table.py"),
    18: ("18 - Figures",            PROJECT_DIR / "make_figures.py"),
    19: ("19 - Flowchart",          PROJECT_DIR / "make_flowchart.py"),
}


def split_sql(text):
    """Split SQL on semicolons, stripping single-line comments first."""
    text = re.sub(r"--[^\n]*", "", text)
    return [s.strip() for s in text.split(";") if s.strip()]


def run_step(con, label, path):
    print(f"\n{'='*70}")
    print(f"  STEP {label}")
    print(f"{'='*70}")

    stmts = split_sql(path.read_text(encoding="utf-8"))
    t0 = time.time()

    for i, stmt in enumerate(stmts, 1):
        first_word = stmt.lstrip().split()[0].upper() if stmt.strip() else ""
        try:
            result = con.execute(stmt)
            if first_word == "SELECT":
                df = result.df()
                print(f"\n  [Query {i}]")
                print(df.to_string(index=False))
        except Exception as e:
            print(f"\n  [ERROR in statement {i}]: {e}")
            print(f"  Statement preview: {stmt[:200]}")
            raise

    print(f"\n  Done in {time.time() - t0:.1f}s")


def connect():
    import duckdb
    temp_dir = (CMS_DIR / "duckdb_temp").as_posix()
    con = duckdb.connect(str(DB_PATH), read_only=False)
    con.execute(
        f"SET memory_limit='24GB'; "
        f"SET threads=12; "
        f"SET temp_directory='{temp_dir}';"
    )
    return con


def run_subprocess(step_label, script_path):
    print(f"\n{'='*70}")
    print(f"  STEP {step_label}  (subprocess)")
    print(f"{'='*70}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)
    print(f"  Done in {time.time() - t0:.1f}s")


def main():
    start_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting OPSCC pipeline from step {start_step} ...")
    print(f"  DB:      {DB_PATH}")
    print(f"  Scripts: {PROJECT_DIR}")

    # SQL steps 1-11 run inside a single DuckDB connection
    sql_steps = [s for s in range(1, 12) if s >= start_step]
    py_steps  = [s for s in range(12, 20) if s >= start_step]

    if sql_steps:
        con = connect()
        print("Connected.\n")
        for step in sql_steps:
            run_step(con, *STEPS_SQL[step])
        con.close()

    for step in py_steps:
        run_subprocess(*STEPS_PY[step])

    # Final summary
    con = connect()
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}\n")

    summary_tables = [
        "hnc_dx_raw",
        "hnc_confirmed",
        "hnc_universe",
        "opscc_universe",
        "opscc_cohort",
        "opscc_comorbidity",
        "opscc_propensity",
        "opscc_outcomes",
        "opscc_survival",
        "opscc_ffs_dates",
    ]
    for tbl in summary_tables:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:30s}  {n:>10,} rows")
        except Exception as e:
            print(f"  {tbl:30s}  ERROR: {e}")

    con.close()


if __name__ == "__main__":
    main()
