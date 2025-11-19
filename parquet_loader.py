#!/usr/bin/env python3
"""
Convert CSV / CSV.GZ files to Parquet using DuckDB.
- Uses headers.json when a matching header set is found
- Falls back to the file's own header row if not
- Increases max_line_size to tolerate very long CMS lines
"""

import duckdb
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CMS_DIR = Path(os.getenv("CMS_directory"))
PARQUET_DIR = Path(os.getenv("parquet_directory"))
HEADERS_PATH = Path("headers.json")

if not CMS_DIR or not CMS_DIR.exists():
    print("Error: CMS_directory is not set or does not exist.")
    sys.exit(1)
if not PARQUET_DIR.exists():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
if not HEADERS_PATH.exists():
    print("Error: headers.json file not found.")
    sys.exit(1)

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
    header_key = pick_header_key(csv_path.name)

    # Skip files you don't want
    if header_key is None:
        print(f"⏭️ Skipping {csv_path.name} (no matching headers.json key)")
        continue

    pq_path = PARQUET_DIR / f"{csv_path.stem}.parquet"
    print(f"Converting {csv_path} → {pq_path} using '{header_key}' headers")

    header_list = HEADERS[header_key]
    if csv_path.name[:3].lower() == "inp" or csv_path.name[:3].lower() == "out":
        header_list = None
        print(f"Using file's own header row for {csv_path.name}")
    continue
    try:
        names_sql = ", ".join("'" + c.replace("'", "''") + "'" for c in header_list)
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path}',
                    delim=',',
                    names=[{names_sql}],
                    header=false,
                    quote='"',
                    escape='"',
                    compression='auto',
                    null_padding=true,
                    max_line_size=100000000,
                    all_varchar=true,
                    sample_size=-1,
                    strict_mode=false
                )
            )
            TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    except Exception as e:
        print(f"[ERROR] Failed on {csv_path}: {e}")
        continue


print("✅ Done converting all CSVs to Parquet.")
