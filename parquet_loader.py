#!/usr/bin/env python3
"""
Convert CSV / CSV.GZ files to Parquet using DuckDB, filtering to only rows whose DSYSRTKY
appears in keys.txt.

- Uses headers.json when a matching header set is found
- Detects whether the CSV already has a header row and skips it if present
- Increases max_line_size to tolerate very long CMS lines
- Reads all columns as VARCHAR (all_varchar=true) like your original script
"""

import duckdb
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CMS_DIR = Path(os.getenv("CMS_directory", ""))
PARQUET_DIR = Path(os.getenv("parquet_directory", "parquet"))
HEADERS_PATH = Path(os.getenv("headers_path", "headers.json"))
KEYS_PATH = Path(os.getenv("keys_path", "keys.txt"))
KEY_COL = os.getenv("key_column", "DSYSRTKY")  # column in the CSV to filter on

# ---- validate paths ----
if not CMS_DIR or not CMS_DIR.exists():
    print("Error: CMS_directory is not set or does not exist.")
    sys.exit(1)

PARQUET_DIR.mkdir(parents=True, exist_ok=True)

if not HEADERS_PATH.exists():
    print(f"Error: headers.json file not found at {HEADERS_PATH}")
    sys.exit(1)

if not KEYS_PATH.exists():
    print(f"Error: keys file not found at {KEYS_PATH}")
    sys.exit(1)

with HEADERS_PATH.open("r") as f:
    HEADERS = json.load(f)  # e.g., {"inp_claimsk": {...}, "out_claimsk": {...}, ...}

# ---- duckdb connection ----
con = duckdb.connect()
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA preserve_insertion_order=false;")
con.execute(f"PRAGMA temp_directory='{(PARQUET_DIR / 'tmp_duckdb').as_posix()}';")
con.execute("PRAGMA memory_limit='6GB';")

# ---- load keys table once ----
con.execute("DROP TABLE IF EXISTS keys;")
con.execute(f"""
    CREATE TEMP TABLE keys AS
    SELECT column0 AS dsysrtky
    FROM read_csv('{KEYS_PATH.as_posix()}', header=false, delim=',', all_varchar=true);
""")
con.execute("CREATE INDEX IF NOT EXISTS keys_dsysrtky_idx ON keys(dsysrtky);")
n_keys = con.execute("SELECT COUNT(*) FROM keys;").fetchone()[0]
print(f"Loaded {n_keys:,} keys from {KEYS_PATH}")

def pick_header_key(filename: str) -> str | None:
    """Try to find a key like 'inp_claimsk', 'out_claimsk', 'car_line', etc. in the filename."""
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
        # NOTE: This reads raw text. For .csv.gz it will not work reliably.
        # Most CMS drops are plain .csv without headers; keep as your original logic.
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()

        first_row_cols = len(first_line.split(","))
        first_value = first_line.split(",")[0].strip('"').strip()

        looks_like_header = (
            first_row_cols == expected_header_count
            and first_value
            and not first_value.replace("-", "").replace("/", "").isdigit()
            and any(c.isalpha() for c in first_value)
        )
        return looks_like_header
    except Exception as e:
        print(f"  [WARN] Could not detect header in {csv_path.name}: {e}")
        return False

def header_has_col(header_list: list[str], col: str) -> bool:
    wanted = col.upper()
    return any(h.upper() == wanted for h in header_list)

# ---- convert ----
# Support both .csv and .csv.gz
csv_files = list(CMS_DIR.rglob("*.csv")) + list(CMS_DIR.rglob("*.csv.gz"))

for csv_path in csv_files:
    header_key = pick_header_key(csv_path.name)

    # Skip files you don't want
    if header_key is None:
        print(f"⏭️ Skipping {csv_path.name} (no matching headers.json key)")
        continue

    # Always use headers from headers.json (extract keys from the dict)
    header_list = list(HEADERS[header_key].keys())

    # Skip files that do not have the key column
    if not header_has_col(header_list, KEY_COL):
        print(f"⏭️ Skipping {csv_path.name} (no {KEY_COL} column in {header_key} headers)")
        continue

    # Output path: strip .gz if present
    stem = csv_path.name
    if stem.endswith(".csv.gz"):
        out_stem = stem[:-7]
    elif stem.endswith(".csv"):
        out_stem = stem[:-4]
    else:
        out_stem = csv_path.stem

    pq_path = PARQUET_DIR / f"{out_stem}.parquet"
    print(f"Converting {csv_path} → {pq_path} using '{header_key}' headers (filtering on {KEY_COL})")

    expected_col_count = len(header_list)

    # Detect if file has a header row that needs to be skipped
    # (Will likely be False for CMS; for .gz this detection may fail, but header=False is usually correct anyway)
    has_header = detect_and_skip_header(csv_path, expected_col_count) if csv_path.suffix != ".gz" else False

    if has_header:
        print("  Detected header row - will skip and use headers.json")
    else:
        print("  No header row detected - will apply headers.json")

    try:
        names_sql = ", ".join("'" + c.replace("'", "''") + "'" for c in header_list)

        con.execute(f"""
            COPY (
                SELECT src.*
                FROM read_csv(
                    '{csv_path.as_posix()}',
                    delim=',',
                    names=[{names_sql}],
                    header={str(has_header).lower()},
                    quote='"',
                    escape='"',
                    compression='auto',
                    null_padding=true,
                    max_line_size=100000000,
                    all_varchar=true,
                    sample_size=-1,
                    strict_mode=false
                ) AS src
                WHERE EXISTS (
                    SELECT 1
                    FROM keys k
                    WHERE k.dsysrtky = src.{KEY_COL}
                )
            )
            TO '{pq_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    except Exception as e:
        print(f"[ERROR] Failed on {csv_path}: {e}")
        continue

print("Done converting all CSVs to filtered Parquet.")
