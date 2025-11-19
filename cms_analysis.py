import duckdb
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv


def get_patients_with_icd10(con, codes, table="all"):
    """
    Return a DataFrame of unique patients (DSYSRTKY)
    with any diagnosis matching the ICD-10 list.

    codes:
        ["C01", "C02"] or [("C00","C14"), ("C30","C32")]

    table:
        "inp", "outp", "car", or "all" (default)
    """

    # Build SQL conditions
    conditions = []
    for code in codes:
        if isinstance(code, tuple):
            lo, hi = code
            conditions.append(f"(SUBSTRING(dx,1,3) BETWEEN '{lo}' AND '{hi}')")
        else:
            conditions.append(f"(SUBSTRING(dx,1,{len(code)}) = '{code}')")

    where_clause = " OR ".join(conditions)

    # Diagnosis extraction CTE for any table
    def dx_cte(tbl):
        return f"""
        SELECT DSYSRTKY,
        UNNEST([
            ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4,
            ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7, ICD_DGNS_CD8,
            ICD_DGNS_CD9, ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12
        ]) AS dx
        FROM {tbl}
        """

    # Build FROM clause based on table parameter
    if table == "all":
        from_clause = f"""
        {dx_cte('inp')}
        UNION ALL
        {dx_cte('outp')}
        UNION ALL
        {dx_cte('car')}
        """
    elif table in ("inp", "outp", "car"):
        from_clause = dx_cte(table)
    else:
        raise ValueError("table must be one of: 'inp', 'outp', 'car', 'all'")

    # Final query
    query = f"""
    WITH all_diags AS (
        {from_clause}
    )
    SELECT DISTINCT DSYSRTKY
    FROM all_diags
    WHERE {where_clause};
    """

    return con.execute(query).fetchdf()




load_dotenv()

HEADERS_PATH = Path("headers.json")

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------

# Change this path if needed
CMS_DIR = Path(os.getenv("CMS_directory"))
PARQUET_DIR = Path(os.getenv("parquet_directory"))

FILES = {
    "car":      "car_claimsk_2019.parquet",
    "car_line": "car_line_2019.parquet",
    "inp":      "inp_claimsk_2019.parquet",
    "outp":     "OUT_claimsk_2019.parquet",
    "mbsf":     "mbsf_2019.parquet",
}

# -------------------------------------------------------------------
# 2. CONNECT TO DUCKDB
# -------------------------------------------------------------------

# :memory: = clean runtime DB
# If you want persistence, replace with "cms.duckdb"
con = duckdb.connect(database="cms.duckdb", read_only=False)
print("[INFO] Connected to DuckDB\n")

# Optional – increase memory (you probably don't need this)
con.execute("PRAGMA memory_limit='6GB';")
con.execute("PRAGMA threads=4;")


# -------------------------------------------------------------------
# 3. REGISTER PARQUET FILES AS SQL VIEWS
# -------------------------------------------------------------------

print("[INFO] Registering Parquet files as views...")

for view_name, filename in FILES.items():
    path = PARQUET_DIR / filename
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path}');")
    print(f"  - View '{view_name}' → {path}")

print()

# -------------------------------------------------------------------


#-
# Count number of patients with HNC
#- 

HNC_codes = [
    ("C00", "C14"),
    ("C30", "C32")
]

df = get_patients_with_icd10(con, HNC_codes, "all")
print("Number of unique patients with head and neck cancer (HNC):", len(df))
print(df.head())





