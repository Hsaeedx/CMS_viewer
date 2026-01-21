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
HEADERS_PATH = Path(os.getenv("headers_path", "headers.json"))

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
con.execute(f"PRAGMA temp_directory='{PARQUET_DIR}/tmp_duckdb';")
con.execute("PRAGMA memory_limit='6GB';")

def pick_header_key(filename: str) -> str | None:
    # Try to find a key like "inp_claimsk", "out_claimsk", "car_line", etc. in the filename
    lower = filename.lower()
    for k in HEADERS.keys():
        if k.lower() in lower:
            return k
    return None


def detect_and_skip_header(csv_path: Path, expected_header_count: int) -> bool:
    """
    Detect if CSV file has a header row that needs to be skipped.
    Returns True if the file has a header row, False otherwise.
    """
    try:
        # Read first line
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()

        # Count columns in first row
        first_row_cols = len(first_line.split(','))

        # If column count matches expected and first value looks like a header name
        # (contains letters, not just numbers/dates), assume it's a header
        first_value = first_line.split(',')[0].strip('"').strip()

        # Check if it looks like a header (non-numeric, not a date pattern)
        looks_like_header = (
            first_row_cols == expected_header_count and
            first_value and
            not first_value.replace('-', '').replace('/', '').isdigit() and
            any(c.isalpha() for c in first_value)
        )

        return looks_like_header
    except Exception as e:
        print(f"  [WARN] Could not detect header in {csv_path.name}: {e}")
        return False


for csv_path in CMS_DIR.rglob("*.csv"):
    header_key = pick_header_key(csv_path.name)

    # Skip files you don't want
    if header_key is None:
        print(f"⏭️ Skipping {csv_path.name} (no matching headers.json key)")
        continue

    pq_path = PARQUET_DIR / f"{csv_path.stem}.parquet"
    print(f"Converting {csv_path} → {pq_path} using '{header_key}' headers")

    # Always use headers from headers.json (extract keys from the dict)
    header_list = list(HEADERS[header_key].keys())
    expected_col_count = len(header_list)

    # Detect if file has a header row that needs to be skipped
    has_header = detect_and_skip_header(csv_path, expected_col_count)

    if has_header:
        print(f"  ✓ Detected header row - will skip and use headers.json")
    else:
        print(f"  ✓ No header row detected - will apply headers.json")

    try:
        names_sql = ", ".join("'" + c.replace("'", "''") + "'" for c in header_list)

        con.execute(f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path}',
                    delim=',',
                    names=[{names_sql}],
                    header={has_header},
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
