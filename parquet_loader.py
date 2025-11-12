#!/usr/bin/env python3
"""
Convert CSV / CSV.GZ files to Parquet using DuckDB.
- Uses headers.json when a matching header set is found
- Falls back to the file's own header row if not
- Increases max_line_size to tolerate very long CMS lines
"""

import duckdb
import json
from pathlib import Path

CMS_DIR = Path("/Volumes/hssd_2tb/CMS/2019")
HEADERS_PATH = Path("headers.json")

with HEADERS_PATH.open("r") as f:
    HEADERS = json.load(f)  # e.g., {"inp_claimsk": [...], "out_claimsk": [...], ...}

con = duckdb.connect()
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA preserve_insertion_order=false;")
con.execute("PRAGMA temp_directory='/Volumes/hssd_2tb/tmp_duckdb';")
con.execute("PRAGMA memory_limit='6GB';")

def pick_header_key(filename: str) -> str | None:
    # Try to find a key like "inp_claimsk", "out_claimsk", "car_line", etc. in the filename
    lower = filename.lower()
    for k in HEADERS.keys():
        if k.lower() in lower:
            return k
    return None


for csv_path in CMS_DIR.rglob("*.csv"):
    # if "OUT" not in csv_path.name.upper():
    #     continue

    pq_path = csv_path.with_suffix(".parquet")
    print(f"Converting {csv_path}  →  {pq_path}")

    header_key = pick_header_key(csv_path.name)
    header_list = HEADERS.get(header_key)

    try:
        # Choose header mode + column list
        if header_list:
            names_sql = ", ".join("'" + c.replace("'", "''") + "'" for c in header_list)
            header_flag = "false"
            col_def = f"names=[{names_sql}], header={header_flag}"
        else:
            header_flag = "true"
            col_def = f"header={header_flag}"

        con.execute(f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path}',
                    delim=',',
                    header=true,
                    quote='"',
                    escape='"',
                    compression='auto',
                    null_padding=true,
                    max_line_size=100000000,
                    all_varchar=true,     -- 👈 key line: disables type inference
                    sample_size=-1,       -- scan entire file to align columns properly
                    strict_mode=false
                )
            )
            TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
    except Exception as e:
        print(f"[ERROR] Failed on {csv_path}: {e}")
        continue

print("✅ Done converting all CSVs to Parquet.")
