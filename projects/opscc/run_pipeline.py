"""
run_pipeline.py
Runs the OPSCC three-comparison pipeline in order.

Study design:
  Comparison A — TORS alone  vs  RT alone   (surgery vs radiation monotherapy)
  Comparison B — TORS + RT   vs  CRT        (adjuvant-RT surgery vs chemoradiation)
  Comparison C — TORS + CRT  vs  TORS + RT  (triple modality vs surgery+RT)

  SQL steps (run via DuckDB connection):
    1. hnc_dx_raw.sql        — Raw HNC diagnosis claims across all sources
    2. hnc_confirmed.sql     — HNC confirmation (>=1 inpatient OR >=2 outpatient >=30d apart)
    3. hnc_universe.sql      — Universe of confirmed HNC patients
    4. opscc_universe.sql    — Filter to OPSCC primaries (C01, C09, C10, C14)
    5. opscc_cohort.sql      — Apply FFS enrollment (at dx month) + prior cancer exclusion (12-month lookback)
    6. opscc_ctcrt.sql       — Annotate cohort: TORS date, chemo date, RT date, metastatic flag
    7. opscc_elixhauser.sql  — Elixhauser comorbidity flags + van Walraven score
    8. opscc_propensity.sql  — Analytic dataset with 5 tx_groups, demographics, comorbidities
    9. opscc_outcomes.sql    — Dysphagia outcomes
   10. opscc_survival.sql    — Survival table: death date + Dec-31 censor per patient
   11. opscc_ffs_dates.sql   — FFS dropout date: last continuous Part A+B non-HMO month

  Python + post-PSM SQL steps:
   12. iptw_analysis.py             — Three 1:1 PSMs; writes psm_matched_{A,B,C} to opscc_propensity
   13. opscc_slp.sql                — Multi-interval SLP utilization for matched patients
   13a. opscc_gtube_dependence.sql  — Multi-interval G-tube utilization for matched patients
   14. survival_analysis.py         — Overall KM + Cox PH (all comparisons)
   15. subgroup_analysis.py         — Subgroup KM + Cox
   16. outcomes_analysis.py         — Cumulative-incidence rates: G-tube, SLP, dysphagia
   17. make_table1.py               — Table 1 demographics
   18. make_outcomes_table.py       — Outcomes tables export
   19. make_figures.py              — KM + forest + cumulative-incidence plots
   20. make_flowchart.py            — Cohort flowchart

Pass a start step number to resume from a specific step, e.g.:
    python run_pipeline.py 10

Output artifacts (Excel tables, PNG figures) for all steps >= start_step are
deleted before those steps run, so no stale files can ever persist.
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
    15: ("15 - Survival Analysis",  PROJECT_DIR / "survival_analysis.py"),
    16: ("16 - Subgroup Survival",  PROJECT_DIR / "subgroup_analysis.py"),
    17: ("17 - Outcomes Analysis",  PROJECT_DIR / "outcomes_analysis.py"),
    18: ("18 - Table 1",            PROJECT_DIR / "make_table1.py"),
    19: ("19 - Outcomes Table",     PROJECT_DIR / "make_outcomes_table.py"),
    20: ("20 - Figures",            PROJECT_DIR / "make_figures.py"),
    21: ("21 - Flowchart",          PROJECT_DIR / "make_flowchart.py"),
}

# SQL steps that run after PSM matching (depend on psm_matched_A/B/C flags)
STEPS_SQL_POST = {
    13: ("13 - SLP Utilization",       QUERIES_DIR / "opscc_slp.sql"),
    14: ("14 - G-tube Dependence",     QUERIES_DIR / "opscc_gtube_dependence.sql"),
}


# Output artifacts produced by steps 18-21; keyed by the first step that creates them
OUTPUT_FILES = {
    18: [PROJECT_DIR / "figures" / "table1_psm.xlsx"],
    19: [PROJECT_DIR / "figures" / "outcomes_tables.xlsx"],
    20: sorted((PROJECT_DIR / "figures").glob("fig_*.png")) if (PROJECT_DIR / "figures").exists() else [],
    21: [PROJECT_DIR / "figures" / "cohort_flowchart.png"],
}


def clean_outputs(start_step):
    """Delete all output files for steps >= start_step so nothing stale remains."""
    for step, files in OUTPUT_FILES.items():
        if step < start_step:
            continue
        # Re-evaluate glob at runtime so we catch any files present now
        if step == 20:
            files = list((PROJECT_DIR / "figures").glob("fig_*.png")) if (PROJECT_DIR / "figures").exists() else []
        for f in files:
            f = Path(f)
            if f.exists():
                f.unlink()
                print(f"  Deleted stale output: {f.name}")


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

    # Remove stale output artifacts before regenerating them
    clean_outputs(start_step)

    # SQL steps 1-11 run inside a single DuckDB connection
    sql_steps      = [s for s in range(1, 12) if s >= start_step]
    sql_post_steps = [s for s in sorted(STEPS_SQL_POST) if s >= start_step]
    py_steps       = [s for s in sorted(STEPS_PY) if s >= start_step]

    if sql_steps:
        con = connect()
        print("Connected.\n")
        for step in sql_steps:
            run_step(con, *STEPS_SQL[step])
        con.close()

    # Interleave post-PSM SQL steps and Python steps in order
    all_remaining = sorted(set(sql_post_steps + py_steps))
    for step in all_remaining:
        if step in STEPS_SQL_POST:
            con = connect()
            run_step(con, *STEPS_SQL_POST[step])
            con.close()
        elif step in STEPS_PY:
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
        "opscc_slp",
        "opscc_gtube_dependence",
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
