#!/usr/bin/env python3
"""
Build Cohort Tables

Creates permanent cohort tables in cms_data.duckdb for a specific diagnosis group.
Tables can be filtered by year and/or sample group for efficient subset analysis.

Cohorts include only patients with full-time enrollment:
  - 12 months Part A (A_MO_CNT = '12')
  - 12 months Part B (B_MO_CNT = '12')
  - No HMO months (HMO_MO = '0', fee-for-service only)

Usage:
    python build_cohort.py dysphagia_core                    # All years, all patients
    python build_cohort.py dysphagia_core --sample 5         # All years, 5% sample
    python build_cohort.py dysphagia_core --year 2019        # Single year, all patients
    python build_cohort.py dysphagia_core -y 2019 -s 5       # Single year, 5% sample
"""

import argparse
import duckdb
import json
from dotenv import load_dotenv

# Available years in the database
AVAILABLE_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]


def get_sample_filter(sample_pct):
    """
    Get the SAMPLE_GROUP filter for a given sample percentage.

    CMS sample groups:
    - '01', '04': 5% sample
    - '15': Additional 15% (combined with above = 20%)
    """
    if sample_pct is None:
        return None, ""
    elif sample_pct == 5:
        return "m.SAMPLE_GROUP IN ('01', '04')", "_5pct"
    elif sample_pct == 20:
        return "m.SAMPLE_GROUP IN ('01', '04', '15')", "_20pct"
    else:
        raise ValueError(f"Invalid sample size: {sample_pct}. Use 5 or 20.")


def load_diagnosis_codes(group_name, diagnoses_file="diagnoses.json"):
    """Load diagnosis codes from JSON file for a given group."""
    with open(diagnoses_file, 'r') as f:
        diagnoses = json.load(f)

    if group_name not in diagnoses:
        available = [k for k in diagnoses.keys() if k != "notes"]
        raise ValueError(f"Group '{group_name}' not found. Available: {available}")

    if group_name == "notes":
        raise ValueError("'notes' is not a diagnosis group")

    group = diagnoses[group_name]
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
        for value in group.values():
            if isinstance(value, dict) and "values" in value:
                codes.extend(value["values"])

    if not codes:
        raise ValueError(f"No diagnosis codes found for group '{group_name}'")

    return codes


def build_icd_where_clause_for_column(codes, col_name):
    """Build SQL WHERE clause for ICD-10 code matching on a single column."""
    conditions = []
    for code in codes:
        if isinstance(code, tuple):
            lo, hi = code
            conditions.append(f"(SUBSTRING({col_name},1,3) BETWEEN '{lo}' AND '{hi}')")
        else:
            conditions.append(f"(SUBSTRING({col_name},1,{len(code)}) = '{code}')")
    return "(" + " OR ".join(conditions) + ")"


def build_icd_where_clause_inpatient(codes):
    """Build WHERE clause checking all diagnosis columns for inpatient claims."""
    cols = ["PRNCPAL_DGNS_CD", "ADMTG_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 26)]
    col_conditions = [build_icd_where_clause_for_column(codes, col) for col in cols]
    return " OR ".join(col_conditions)


def build_icd_where_primary_clause_inpatient(codes):
    """Build WHERE clause checking only primary diagnosis column for inpatient claims."""
    return build_icd_where_clause_for_column(codes, "PRNCPAL_DGNS_CD")


def build_icd_where_clause_outpatient(codes):
    """Build WHERE clause checking all diagnosis columns for outpatient claims."""
    cols = ["PRNCPAL_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 26)]
    col_conditions = [build_icd_where_clause_for_column(codes, col) for col in cols]
    return " OR ".join(col_conditions)


def build_icd_where_clause_carrier(codes):
    """Build WHERE clause checking all 13 diagnosis columns for carrier claims."""
    cols = ["PRNCPAL_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 13)]
    col_conditions = [build_icd_where_clause_for_column(codes, col) for col in cols]
    return " OR ".join(col_conditions)


def build_icd_where_clause_snf(codes):
    """Build WHERE clause checking all diagnosis columns for SNF claims.
    SNF has same structure as inpatient: PRNCPAL_DGNS_CD + ADMTG_DGNS_CD + ICD_DGNS_CD1-25."""
    cols = ["PRNCPAL_DGNS_CD", "ADMTG_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 26)]
    col_conditions = [build_icd_where_clause_for_column(codes, col) for col in cols]
    return " OR ".join(col_conditions)


def build_icd_where_clause_hha(codes):
    """Build WHERE clause checking all diagnosis columns for HHA claims.
    HHA has same structure as outpatient: PRNCPAL_DGNS_CD + ICD_DGNS_CD1-25 (no ADMTG_DGNS_CD)."""
    cols = ["PRNCPAL_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 26)]
    col_conditions = [build_icd_where_clause_for_column(codes, col) for col in cols]
    return " OR ".join(col_conditions)


def build_cohort_table_for_year(con, table_name, codes, year, sample_filter=None):
    """
    Create a cohort table for a specific year using pure SQL.
    Uses direct column checks (no UNNEST) for better performance.
    Searches across all claim types: inpatient, outpatient, carrier, SNF, and HHA.
    """
    inpatient_where = build_icd_where_clause_inpatient(codes)
    outpatient_where = build_icd_where_clause_outpatient(codes)
    carrier_where = build_icd_where_clause_carrier(codes)
    snf_where = build_icd_where_clause_snf(codes)
    hha_where = build_icd_where_clause_hha(codes)

    # Build table names for specific year
    inp_table = f"inp_claimsk_{year}"
    out_table = f"out_claimsk_{year}"
    car_table = f"car_claimsk_{year}"
    snf_table = f"snf_claimsk_{year}"
    hha_table = f"hha_claimsk_{year}"
    mbsf_table = f"mbsf_{year}"

    # Check which tables exist for this year
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    available_tables = {t[0].lower() for t in tables}

    # Build UNION query for available claim tables - now with direct WHERE on columns
    unions = []

    if inp_table.lower() in available_tables:
        unions.append(f"""
            SELECT DISTINCT DSYSRTKY FROM {inp_table}
            WHERE {inpatient_where}
        """)

    if out_table.lower() in available_tables:
        unions.append(f"""
            SELECT DISTINCT DSYSRTKY FROM {out_table}
            WHERE {outpatient_where}
        """)

    if car_table.lower() in available_tables:
        unions.append(f"""
            SELECT DISTINCT DSYSRTKY FROM {car_table}
            WHERE {carrier_where}
        """)

    if snf_table.lower() in available_tables:
        unions.append(f"""
            SELECT DISTINCT DSYSRTKY FROM {snf_table}
            WHERE {snf_where}
        """)

    if hha_table.lower() in available_tables:
        unions.append(f"""
            SELECT DISTINCT DSYSRTKY FROM {hha_table}
            WHERE {hha_where}
        """)

    if not unions:
        print(f"  [WARN] No claims tables found for year {year}")
        return None

    claims_union = "\n        UNION\n        ".join(unions)

    # Build enrollment filter
    enrollment_filter = "m.A_MO_CNT = '12' AND m.B_MO_CNT = '12' AND m.HMO_MO = '0'"
    if sample_filter:
        where_clause = f"{enrollment_filter} AND {sample_filter}"
    else:
        where_clause = enrollment_filter

    # Drop and create in one shot - all in SQL
    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    query = f"""
    CREATE TABLE {table_name} AS
    WITH dx_patients AS (
        {claims_union}
    )
    SELECT DISTINCT d.DSYSRTKY
    FROM dx_patients d
    INNER JOIN {mbsf_table} m ON d.DSYSRTKY = m.DSYSRTKY
    WHERE {where_clause}
    """

    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return count


def build_cohort_table_all_years(con, table_name, codes, sample_filter=None):
    """
    Create a cohort table across all years using pure SQL.
    Uses the _all combined views with direct column checks (no UNNEST).
    Searches across all claim types: inpatient, outpatient, carrier, SNF, and HHA.
    """
    inpatient_where = build_icd_where_clause_inpatient(codes)
    outpatient_where = build_icd_where_clause_outpatient(codes)
    carrier_where = build_icd_where_clause_carrier(codes)
    snf_where = build_icd_where_clause_snf(codes)
    hha_where = build_icd_where_clause_hha(codes)

    # Build enrollment filter
    enrollment_filter = "m.A_MO_CNT = '12' AND m.B_MO_CNT = '12' AND m.HMO_MO = '0'"
    if sample_filter:
        where_clause = f"{enrollment_filter} AND {sample_filter}"
    else:
        where_clause = enrollment_filter

    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    query = f"""
    CREATE TABLE {table_name} AS
    WITH dx_patients AS (
        SELECT DISTINCT DSYSRTKY FROM inp_claimsk_all
        WHERE {inpatient_where}
        UNION
        SELECT DISTINCT DSYSRTKY FROM out_claimsk_all
        WHERE {outpatient_where}
        UNION
        SELECT DISTINCT DSYSRTKY FROM car_claimsk_all
        WHERE {carrier_where}
        UNION
        SELECT DISTINCT DSYSRTKY FROM snf_claimsk_all
        WHERE {snf_where}
        UNION
        SELECT DISTINCT DSYSRTKY FROM hha_claimsk_all
        WHERE {hha_where}
    )
    SELECT DISTINCT d.DSYSRTKY
    FROM dx_patients d
    INNER JOIN mbsf_all m ON d.DSYSRTKY = m.DSYSRTKY
    WHERE {where_clause}
    """

    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build cohort tables in DuckDB for a diagnosis group",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_cohort.py dysphagia_core                    # All years, all patients
  python build_cohort.py dysphagia_core --sample 5         # All years, 5% sample
  python build_cohort.py dysphagia_core --year 2019        # Single year, all patients
  python build_cohort.py dysphagia_core -y 2019 -s 5       # Single year, 5% sample

Available diagnosis groups (from diagnoses.json):
  dysphagia_core, aspiration_pneumonitis, pneumonia_sensitivity,
  nutritional_compromise, functional_decline_optional,
  head_neck_cancer_related, neurologic_risk_factors,
  esophageal_contributors_secondary, not_primary_exposure
        """
    )
    parser.add_argument(
        "diagnosis",
        type=str,
        help="Diagnosis group from diagnoses.json"
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        choices=[5, 20],
        default=None,
        help="Sample size percentage (5 or 20). Omit for all patients."
    )
    parser.add_argument(
        "-y", "--year",
        type=int,
        choices=AVAILABLE_YEARS,
        default=None,
        help="Specific year (2016-2022). Omit to build tables for all years."
    )

    args = parser.parse_args()

    load_dotenv()

    # Load diagnosis codes
    print(f"[INFO] Loading diagnosis codes for '{args.diagnosis}'...")
    codes = load_diagnosis_codes(args.diagnosis)
    print(f"  Found {len(codes)} codes")

    # Get sample filter and suffix
    sample_filter, sample_suffix = get_sample_filter(args.sample)
    if sample_filter:
        print(f"[INFO] Filtering to sample: {sample_filter}")

    # Connect to DuckDB
    con = duckdb.connect(database="cms_data.duckdb", read_only=False)
    print("[INFO] Connected to cms_data.duckdb\n")

    results = {}

    if args.year:
        # Single year
        print(f"[INFO] Building cohort for year {args.year}...")
        table_name = f"{args.diagnosis}_{args.year}{sample_suffix}"
        count = build_cohort_table_for_year(con, table_name, codes, args.year, sample_filter)
        if count is not None:
            results[table_name] = count
            print(f"  Created table '{table_name}' with {count:,} patients")
        else:
            print(f"  No patients found for year {args.year}")
    else:
        # All years: build per-year tables AND a total table
        print("[INFO] Building per-year cohort tables...")
        print("=" * 60)

        for year in AVAILABLE_YEARS:
            print(f"\n[{year}] Building cohort...")
            table_name = f"{args.diagnosis}_{year}{sample_suffix}"
            count = build_cohort_table_for_year(con, table_name, codes, year, sample_filter)
            if count is not None:
                results[table_name] = count
                print(f"  Created '{table_name}': {count:,} patients")
            else:
                print(f"  No claims tables for {year}")

        # Build total table using _all views
        # print(f"\n[TOTAL] Building combined cohort from all years...")
        # total_table_name = f"{args.diagnosis}_all{sample_suffix}"
        # total_count = build_cohort_table_all_years(con, total_table_name, codes, sample_filter)
        # results[total_table_name] = total_count
        # print(f"  Created '{total_table_name}': {total_count:,} patients")

    # Summary
    print("\n" + "=" * 60)
    print("Summary of created tables:")
    print("-" * 60)
    for table_name, count in results.items():
        print(f"  {table_name}: {count:,} patients")

    con.close()
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
