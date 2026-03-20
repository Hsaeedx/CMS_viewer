"""
run_pipeline.py
Runs the stroke+SLP pipeline in order:
  1. stroke_cohort.sql
  2. stroke_slp.sql
  3. build_stroke_comorbidity.py  (subprocess)
  4. stroke_propensity.sql
  5. stroke_outcomes.sql
  6. stroke_psm.py                (subprocess)
  7. stroke_analysis.py           (subprocess)

Each SQL file is split on semicolons and executed statement-by-statement.
SELECT results are printed after each step.

Pass a start step number to resume from a specific step, e.g.:
    python run_pipeline.py 6
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up: stroke_SLP -> projects -> root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DB_PATH     = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
CMS_DIR     = Path(os.getenv("CMS_directory", ""))
PROJECT_DIR = Path(__file__).resolve().parent
QUERIES_DIR = PROJECT_DIR / "queries"

STEPS_SQL = {
    1: ("1 - Cohort",       QUERIES_DIR / "stroke_cohort.sql"),
    2: ("2 - SLP Exposure", QUERIES_DIR / "stroke_slp.sql"),
    4: ("4 - Propensity",   QUERIES_DIR / "stroke_propensity.sql"),
    5: ("5 - Outcomes",     QUERIES_DIR / "stroke_outcomes.sql"),
}

STEPS_PY = {
    3: ("3 - Comorbidity", PROJECT_DIR / "build_stroke_comorbidity.py"),
    6: ("6 - PSM",         PROJECT_DIR / "stroke_psm.py"),
    7: ("7 - Analysis",    PROJECT_DIR / "stroke_analysis.py"),
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
    subprocess.run([sys.executable, str(script_path)], check=True)
    print(f"  Done in {time.time() - t0:.1f}s")


def main():
    start_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting pipeline from step {start_step} ...")
    print(f"  DB:      {DB_PATH}")
    print(f"  Scripts: {PROJECT_DIR}")

    con = connect()
    print("Connected.\n")

    for step in [1, 2]:
        if start_step <= step:
            run_step(con, *STEPS_SQL[step])

    if start_step <= 3:
        con.close()
        run_subprocess(*STEPS_PY[3])
        con = connect()

    for step in [4, 5]:
        if start_step <= step:
            run_step(con, *STEPS_SQL[step])

    con.close()

    for step in [6, 7]:
        if start_step <= step:
            run_subprocess(*STEPS_PY[step])

    con = connect()
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}\n")

    for tbl in ["stroke_cohort", "stroke_slp", "stroke_comorbidity",
                "stroke_propensity", "stroke_outcomes"]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:30s}  {n:>10,} rows")
        except Exception as e:
            print(f"  {tbl:30s}  ERROR: {e}")

    con.close()


if __name__ == "__main__":
    main()
