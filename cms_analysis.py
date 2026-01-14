#!/usr/bin/env python3
"""
CMS Data Analysis Functions

Functions for querying and analyzing CMS Medicare claims data stored in DuckDB.
Includes utilities for finding patients by ICD-10 diagnosis codes.
"""

import duckdb
import json
import os
from pathlib import Path
from dotenv import load_dotenv


def get_patients_with_icd10(con, codes, table="all"):
    """
    Return a DataFrame of unique patients (DSYSRTKY)
    with any diagnosis matching the ICD-10 list.

    Parameters:
    -----------
    con : duckdb.Connection
        DuckDB connection
    codes : list
        List of ICD-10 codes. Can be strings for exact matches or tuples for ranges.
        Examples: ["C01", "C02"] or [("C00","C14"), ("C30","C32")]
    table : str
        Which claims to search:
        - "inp" : inpatient claims only
        - "outp" : outpatient claims only
        - "car" : carrier (physician) claims only
        - "all" : all claim types (default)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with unique DSYSRTKY (patient IDs)

    Example:
    --------
    # Find patients with head/neck cancer codes
    codes = [("C00", "C14"), ("C30", "C32")]
    df = get_patients_with_icd10(con, codes, "all")
    print(f"Found {len(df)} patients")
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

    # Diagnosis extraction CTE for inpatient/outpatient (25 diagnosis columns)
    def dx_cte_facility(tbl):
        return f"""
        SELECT DSYSRTKY,
        UNNEST([
            ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4,
            ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7, ICD_DGNS_CD8,
            ICD_DGNS_CD9, ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15, ICD_DGNS_CD16,
            ICD_DGNS_CD17, ICD_DGNS_CD18, ICD_DGNS_CD19, ICD_DGNS_CD20,
            ICD_DGNS_CD21, ICD_DGNS_CD22, ICD_DGNS_CD23, ICD_DGNS_CD24,
            ICD_DGNS_CD25
        ]) AS dx
        FROM {tbl}
        """

    # Diagnosis extraction CTE for carrier (12 diagnosis columns + principal)
    def dx_cte_carrier(tbl):
        return f"""
        SELECT DSYSRTKY,
        UNNEST([
            PRNCPAL_DGNS_CD,
            ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4,
            ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7, ICD_DGNS_CD8,
            ICD_DGNS_CD9, ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12
        ]) AS dx
        FROM {tbl}
        """

    # Build FROM clause based on table parameter
    # Note: Use _all views for combined multi-year tables
    if table == "all":
        from_clause = f"""
        {dx_cte_facility('inp_claimsk_all')}
        UNION ALL
        {dx_cte_facility('out_claimsk_all')}
        UNION ALL
        {dx_cte_carrier('car_claimsk_all')}
        """
    elif table == "inp":
        from_clause = dx_cte_facility('inp_claimsk_all')
    elif table == "outp":
        from_clause = dx_cte_facility('out_claimsk_all')
    elif table == "car":
        from_clause = dx_cte_carrier('car_claimsk_all')
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


def get_patients_by_diagnosis_group(con, group_name, data_source="both", diagnoses_file="diagnoses.json"):
    """
    Get unique patients with diagnoses from a named group in diagnoses.json.

    Parameters:
    -----------
    con : duckdb.Connection
        DuckDB connection
    group_name : str
        Name of diagnosis group from diagnoses.json (e.g., "dysphagia_core", "aspiration_pneumonitis")
    data_source : str
        Where to search for diagnoses:
        - "inp" : inpatient claims only
        - "out" : outpatient claims only
        - "both" : both inpatient and outpatient (default)
    diagnoses_file : str
        Path to diagnoses.json file (default: "diagnoses.json")

    Returns:
    --------
    pandas.DataFrame
        DataFrame with unique DSYSRTKY (patient IDs)

    Example:
    --------
    # Find patients with dysphagia from both inpatient and outpatient
    df = get_patients_by_diagnosis_group(con, "dysphagia_core", data_source="both")
    print(f"Found {len(df)} patients with dysphagia")

    # Find patients with aspiration pneumonia from inpatient only
    df = get_patients_by_diagnosis_group(con, "aspiration_pneumonitis", data_source="inp")
    print(f"Found {len(df)} inpatient aspiration pneumonia cases")
    """

    # Load diagnosis codes from JSON
    with open(diagnoses_file, 'r') as f:
        diagnoses = json.load(f)

    if group_name not in diagnoses:
        available = [k for k in diagnoses.keys() if k != "notes"]
        raise ValueError(f"Group '{group_name}' not found in {diagnoses_file}. Available: {available}")

    group = diagnoses[group_name]

    # Skip notes section
    if group_name == "notes":
        raise ValueError("'notes' is not a diagnosis group")

    # Map data_source to table parameter
    table_map = {
        "inp": "inp",
        "out": "outp",
        "both": "all"
    }

    if data_source not in table_map:
        raise ValueError(f"data_source must be 'inp', 'out', or 'both'. Got: {data_source}")

    table = table_map[data_source]

    # Handle different structures in the JSON
    codes = []

    # Simple structure with "values" key
    if "values" in group:
        codes.extend(group["values"])

    # Complex structure (like head_neck_cancer_related with nested malignancy_range)
    elif "malignancy_range" in group:
        sub_group = group["malignancy_range"]
        codes.extend(sub_group["values"])

    # Nested structure (like neurologic_risk_factors)
    else:
        # Collect all values from nested groups
        for value in group.values():
            if isinstance(value, dict) and "values" in value:
                codes.extend(value["values"])

    if not codes:
        raise ValueError(f"No diagnosis codes found for group '{group_name}'")

    # Use existing function with the collected codes
    return get_patients_with_icd10(con, codes, table)


# =============================================================================
# Main execution / Example usage
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    HEADERS_PATH = Path("headers.json")

    # Configuration
    CMS_DIR = Path(os.getenv("CMS_directory"))
    PARQUET_DIR = Path(os.getenv("parquet_directory"))

    FILES = {
        "car":      "car_claimsk_2019.parquet",
        "car_line": "car_line_2019.parquet",
        "inp":      "inp_claimsk_2019.parquet",
        "outp":     "OUT_claimsk_2019.parquet",
        "mbsf":     "mbsf_2019.parquet",
    }

    # Connect to DuckDB
    con = duckdb.connect(database="cms.duckdb", read_only=False)
    print("[INFO] Connected to DuckDB\n")

    # Optional – increase memory
    con.execute("PRAGMA memory_limit='6GB';")
    con.execute("PRAGMA threads=4;")

    # Register Parquet files as SQL views
    print("[INFO] Registering Parquet files as views...")
    for view_name, filename in FILES.items():
        path = PARQUET_DIR / filename
        con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path}');")
        print(f"  - View '{view_name}' → {path}")

    print()

    # Count unique patients in mbsf
    print("[INFO] Counting unique patients in 'mbsf'...")
    query = """
    SELECT COUNT(DISTINCT DSYSRTKY) AS unique_patients
    FROM mbsf;
    """
    result = con.execute(query).fetchone()
    print(f"Number of unique patients in 'mbsf': {result[0]:,}")

    print()

    # Example: Find patients with head and neck cancer
    print("[INFO] Example: Finding patients with head/neck cancer...")
    HNC_codes = [
        ("C00", "C14"),
        ("C30", "C32")
    ]
    df = get_patients_with_icd10(con, HNC_codes, "all")
    print(f"Number of unique patients with head and neck cancer (HNC): {len(df):,}")

    print()

    # Example: Using diagnosis groups from diagnoses.json
    print("[INFO] Example: Using diagnosis groups from diagnoses.json...")
    try:
        # Find patients with dysphagia
        dysphagia_patients = get_patients_by_diagnosis_group(con, "dysphagia_core", data_source="both")
        print(f"Patients with dysphagia: {len(dysphagia_patients):,}")

        # Find patients with aspiration pneumonitis
        aspiration_patients = get_patients_by_diagnosis_group(con, "aspiration_pneumonitis", data_source="both")
        print(f"Patients with aspiration pneumonitis: {len(aspiration_patients):,}")

    except FileNotFoundError:
        print("  Note: diagnoses.json not found. Skipping diagnosis group examples.")
    except ValueError as e:
        print(f"  Note: {e}")

    print("\n[INFO] Analysis complete!")