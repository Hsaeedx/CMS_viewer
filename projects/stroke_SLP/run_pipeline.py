"""
run_pipeline.py
Runs the stroke+SLP SQL pipeline in order:
  1. stroke_cohort.sql
  2. stroke_slp.sql
  3. stroke_comorbidity.sql
  4. stroke_propensity.sql
  5. stroke_outcomes.sql

Each file is split on semicolons and executed statement-by-statement.
SELECT results are printed after each step.
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import re
import subprocess
import time

DB_PATH = r"F:\CMS\cms_data.duckdb"

STEPS_PRE = [
    ("1 - Cohort",       r"F:\CMS\projects\stroke_SLP\queries\stroke_cohort.sql"),
    ("2 - SLP Exposure", r"F:\CMS\projects\stroke_SLP\queries\stroke_slp.sql"),
]

COMORBIDITY_SCRIPT = r"F:\CMS\projects\stroke_SLP\build_stroke_comorbidity.py"

STEPS_POST = [
    ("4 - Propensity",   r"F:\CMS\projects\stroke_SLP\queries\stroke_propensity.sql"),
    ("5 - Outcomes",     r"F:\CMS\projects\stroke_SLP\queries\stroke_outcomes.sql"),
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


def main():
    print(f"Connecting to {DB_PATH} ...")
    con = connect()
    print("Connected.\n")

    for label, path in STEPS_PRE:
        run_step(con, label, path)

    # Step 3 — Comorbidity runs as a separate process (needs its own DB connection)
    con.close()
    print(f"\n{'='*70}")
    print("  STEP 3 - Comorbidity  (subprocess)")
    print(f"{'='*70}")
    t0 = time.time()
    subprocess.run([sys.executable, COMORBIDITY_SCRIPT], check=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    con = connect()
    for label, path in STEPS_POST:
        run_step(con, label, path)

    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}\n")

    # Quick row-count sanity check across all output tables
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
