"""
rerun_slp_propensity.py

Re-runs only the steps affected by the updated SLP exposure definition
and the HHA/hospice discharge exclusion:
  Step 2 — stroke_slp.sql     (new HCPCS codes)
  Step 4 — stroke_propensity.sql  (new exclusion + reads from updated stroke_slp)

Cohort, comorbidity, and outcomes tables are unchanged and are NOT re-run.
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb

DB_PATH     = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
CMS_DIR     = Path(os.getenv("CMS_directory", ""))
PROJECT_DIR = Path(__file__).resolve().parent
QUERIES_DIR = PROJECT_DIR / "queries"

STEPS = [
    ("2 - SLP Exposure", QUERIES_DIR / "stroke_slp.sql"),
    ("4 - Propensity",   QUERIES_DIR / "stroke_propensity.sql"),
]


def split_sql(text):
    text = re.sub(r'--[^\n]*', '', text)
    return [s.strip() for s in text.split(";") if s.strip()]


def run_step(con, label, path):
    print(f"\n{'='*70}")
    print(f"  STEP {label}")
    print(f"{'='*70}")
    with open(str(path), encoding="utf-8") as f:
        sql = f.read()
    stmts = split_sql(sql)
    t0 = time.time()
    for i, stmt in enumerate(stmts, 1):
        first_word = stmt.lstrip().split()[0].upper() if stmt.strip() else ""
        try:
            result = con.execute(stmt)
            if first_word == "SELECT":
                print(f"\n  [Query {i}]")
                print(result.df().to_string(index=False))
        except Exception as e:
            print(f"\n  [ERROR in statement {i}]: {e}")
            print(f"  Statement preview: {stmt[:200]}")
            raise
    print(f"\n  Done in {time.time()-t0:.1f}s")


def main():
    print(f"Connecting to {DB_PATH} ...")
    con = duckdb.connect(str(DB_PATH), read_only=False)
    temp_dir = (CMS_DIR / "duckdb_temp").as_posix()
    con.execute(f"SET memory_limit='24GB'; SET threads=12; SET temp_directory='{temp_dir}';")
    print("Connected.\n")

    for label, path in STEPS:
        run_step(con, label, path)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

    for tbl in ["stroke_slp", "stroke_propensity"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:30s}  {n:>10,} rows")

    con.close()


if __name__ == "__main__":
    main()
