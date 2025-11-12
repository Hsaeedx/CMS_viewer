#!/usr/bin/env python3
# (content abbreviated here in this notebook output)
# See full script in file. This script sets up DuckDB views over your CSVs,
# builds an H&N cohort by ICD-10, and computes basic costs.
import duckdb as dd
import json
from pathlib import Path

CMS_DIR = Path("/Volumes/hssd_2tb/CMS")
HEADERS_PATH = Path("headers.json")
with open(HEADERS_PATH, "r") as f:
    HEADERS = json.load(f)

FILES = [
    "mbsf",
    "inp_claimsk",
    "inp_revenuek",
    "out_claimsk",
    "out_revenuek",
    "car_claimsk",
    "car_linek",
]


con = dd.connect("cms.duckdb")

def csv_view(name, file, year):
    p = Path(f"{file}_{year}.csv")
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    if not p.exists():
        print(f"[WARN] Missing: {file}_{year}.csv[.gz]")
        return

    # Try to get header list from your JSON
    try:
        header = HEADERS[name]
    except KeyError:
        print(f"[WARN] No header entry for: {name}")
        return

    # Build SQL list literal if we have headers
    names_sql = ", ".join("'" + c.replace("'", "''") + "'" for c in header)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {name}_{year} AS
        SELECT * FROM read_csv_auto('{str(p)}',
            header = FALSE,
            names  = [{names_sql}],
            sample_size = -1
        );
        """
    )

    print(f"[OK] {name} -> {p}")


if __name__ == "__main__":
    # Iterate over all years in CMS_DIR
    for year_dir in CMS_DIR.iterdir():
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        print(f"=== YEAR: {year} ===")
        year_path = year_dir
        for k in FILES:
            csv_view(k, year_path / k, year)

    print(con.sql("SHOW TABLES").df())
