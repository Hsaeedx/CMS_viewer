#!/usr/bin/env python3
"""
Sets up DuckDB views over your Parquet files.
Groups similar files (e.g., all INP files) into combined tables.
Uses environment variables for configuration.
"""
import duckdb as dd
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

PARQUET_DIR = Path(os.getenv("parquet_directory"))
DB_PATH = os.getenv("duckdb_database", "cms.duckdb")

if not PARQUET_DIR or not PARQUET_DIR.exists():
    print("Error: parquet_directory is not set or does not exist.")
    print(f"Current value: {PARQUET_DIR}")
    sys.exit(1)


print(f"[INFO] Creating DuckDB database at: {DB_PATH}")
con = dd.connect(DB_PATH)
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA memory_limit='6GB';")
print(f"[INFO] Database connected successfully\n")


def extract_table_type(filename):
    """
    Extract table type from filename.
    Examples:
      - "mbsf_2023.parquet" -> "mbsf"
      - "inp_claimsk_2019.parquet" -> "inp_claimsk"
      - "OUT_claimsk_2023.parquet" -> "out_claimsk"
    """
    # Remove .parquet extension
    name = filename.stem.lower()

    # Remove year pattern (4 digits at the end)
    name = re.sub(r'_\d{4}$', '', name)

    return name


if __name__ == "__main__":
    print(f"[INFO] Recursively scanning for Parquet files in: {PARQUET_DIR}")
    print("="*80)

    # Find all Parquet files recursively
    parquet_files = sorted(PARQUET_DIR.rglob("*.parquet"))

    if not parquet_files:
        print(f"[WARN] No Parquet files found in {PARQUET_DIR}")
        sys.exit(1)

    print(f"[INFO] Found {len(parquet_files)} Parquet files\n")

    # Group files by table type
    file_groups = defaultdict(list)
    for pq_file in parquet_files:
        table_type = extract_table_type(pq_file)
        file_groups[table_type].append(pq_file)

    print(f"[INFO] Grouped into {len(file_groups)} table types\n")

    # Create individual year views
    print("Creating individual year views...")
    print("-" * 80)
    for pq_file in parquet_files:
        view_name = pq_file.stem
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{str(pq_file)}');
            """
        )
        print(f"[OK] {view_name} -> {pq_file.name}")

    # Create combined views for each table type
    print("\n" + "="*80)
    print("Creating combined views (all years)...")
    print("-" * 80)

    for table_type, files in sorted(file_groups.items()):
        if len(files) > 1:
            # Multiple files - create combined view
            file_paths = "', '".join(str(f) for f in files)
            combined_view_name = f"{table_type}_all"

            con.execute(
                f"""
                CREATE OR REPLACE VIEW {combined_view_name} AS
                SELECT * FROM read_parquet(['{file_paths}']);
                """
            )
            print(f"[OK] {combined_view_name} -> {len(files)} files combined")
        else:
            # Single file - create alias
            single_view_name = f"{table_type}_all"
            original_view = files[0].stem

            con.execute(
                f"""
                CREATE OR REPLACE VIEW {single_view_name} AS
                SELECT * FROM {original_view};
                """
            )
            print(f"[OK] {single_view_name} -> alias to {original_view}")

    print("\n" + "="*80)
    print("[INFO] All views created successfully!")
    print("="*80)
    print("\nAvailable tables:")
    tables_df = con.sql("SHOW TABLES").df()
    print(tables_df)

    print(f"\n[INFO] Database saved to: {DB_PATH}")
    print("\n[INFO] Usage:")
    print("  - Individual years: SELECT * FROM mbsf_2023")
    print("  - All years combined: SELECT * FROM mbsf_all")
