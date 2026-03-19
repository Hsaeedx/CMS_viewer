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
    python run_pipeline.py 4
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import re
import subprocess
import time

DB_PATH = r"F:\CMS\cms_data.duckdb"

STEPS_PRE = [
    ("1 - Cohort",       r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\queries\stroke_cohort.sql"),
    ("2 - SLP Exposure", r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\queries\stroke_slp.sql"),
]

COMORBIDITY_SCRIPT = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\build_stroke_comorbidity.py"
PSM_SCRIPT         = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\stroke_psm.py"
ANALYSIS_SCRIPT    = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\stroke_analysis.py"

STEPS_POST = [
    ("4 - Propensity",   r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\queries\stroke_propensity.sql"),
    ("5 - Outcomes",     r"C:\Users\hsaee\Desktop\CMS_viewer\projects\stroke_SLP\queries\stroke_outcomes.sql"),
]


def split_sql(text):
    """Split SQL on semicolons, comment-aware to avoid splitting inside -- comments."""
    # Strip single-line comments before splitting — avoids splitting on ';' in comments
    # and avoids DuckDB parser issues with unicode box-drawing chars in comments.
    text = re.sub(r'--[^\n]*', '', text)
    stmts = []
    for raw in text.split(";"):
        s = raw.strip()
        if s:
            stmts.append(s)
    return stmts


def run_step(con, label, path):
    print(f"\n{'='*70}")
    print(f"  STEP {label}")
    print(f"{'='*70}")

    with open(path, encoding="utf-8") as f:
        sql = f.read()

    stmts = split_sql(sql)
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
            print(f"  Statement preview: {stmt[:200].encode('ascii', errors='replace').decode()}")
            raise

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


def connect():
    con = duckdb.connect(DB_PATH, read_only=False)
    con.execute(
        "SET memory_limit='24GB'; "
        "SET threads=12; "
        "SET temp_directory='F:\\\\CMS\\\\duckdb_temp';"
    )
    return con


def run_subprocess(step_label, script_path):
    print(f"\n{'='*70}")
    print(f"  STEP {step_label}  (subprocess)")
    print(f"{'='*70}")
    t0 = time.time()
    subprocess.run([sys.executable, script_path], check=True)
    print(f"  Done in {time.time() - t0:.1f}s")


def main():
    start_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting pipeline from step {start_step} ...")

    con = connect()
    print("Connected.\n")

    if start_step <= 1:
        run_step(con, "1 - Cohort",       STEPS_PRE[0][1])
    if start_step <= 2:
        run_step(con, "2 - SLP Exposure", STEPS_PRE[1][1])

    if start_step <= 3:
        con.close()
        run_subprocess("3 - Comorbidity", COMORBIDITY_SCRIPT)
        con = connect()

    if start_step <= 4:
        run_step(con, "4 - Propensity", STEPS_POST[0][1])
    if start_step <= 5:
        run_step(con, "5 - Outcomes",   STEPS_POST[1][1])

    con.close()

    if start_step <= 6:
        run_subprocess("6 - PSM",      PSM_SCRIPT)
    if start_step <= 7:
        run_subprocess("7 - Analysis", ANALYSIS_SCRIPT)

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
