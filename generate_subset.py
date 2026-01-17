#!/usr/bin/env python3
"""
Generate Patient Cohort Subsets

Creates cohort tables for specific diagnosis groups to enable efficient
subset analysis without repeatedly scanning the full claims data.

Usage:
    python generate_subset.py                  # All patients (100%)
    python generate_subset.py --sample 5       # 5% sample
    python generate_subset.py --sample 20      # 20% sample (5% + 15%)
    python generate_subset.py -s 5             # Short form
"""

import argparse
import duckdb
import os
from pathlib import Path
from dotenv import load_dotenv
from cms_analysis import get_patients_by_diagnosis_group


def create_cohort_table(con, cohort_name, diagnosis_group, data_source="both",
                       temporary=True, diagnoses_file="diagnoses.json"):
    """
    Create a cohort table for repeated analysis on a specific patient subset.

    Parameters:
    -----------
    con : duckdb.Connection
        DuckDB connection
    cohort_name : str
        Name for the cohort table (e.g., 'dysphagia_cohort')
    diagnosis_group : str
        Diagnosis group from diagnoses.json (e.g., 'dysphagia_core')
    data_source : str
        "inp" (inpatient only), "out" (outpatient only), or "both" (default)
    temporary : bool
        If True, creates TEMP table (lasts for session). If False, creates permanent table.
    diagnoses_file : str
        Path to diagnoses.json file (default: "diagnoses.json")

    Returns:
    --------
    int : Number of patients in cohort

    Example:
    --------
    # Create temporary cohort for dysphagia patients
    num_patients = create_cohort_table(con, 'dysphagia_cohort', 'dysphagia_core', 'both')
    print(f"Created cohort with {num_patients:,} patients")

    # Now run fast queries against the cohort
    result = con.execute('''
        SELECT COUNT(*) FROM inp_claimsk_all i
        INNER JOIN dysphagia_cohort d ON i.DSYSRTKY = d.DSYSRTKY
    ''').fetchone()
    """
    print(f"[INFO] Identifying {diagnosis_group} patients from {data_source} data...")

    # Get the patients (this scans the claims data)
    patients = get_patients_by_diagnosis_group(con, diagnosis_group, data_source, diagnoses_file)
    num_patients = len(patients)

    print(f"[INFO] Found {num_patients:,} unique patients")

    # Drop existing table if it exists
    con.execute(f"DROP TABLE IF EXISTS {cohort_name}")

    # Create the cohort table
    temp_clause = "TEMP " if temporary else ""
    print(f"[INFO] Creating {temp_clause}table '{cohort_name}'...")

    # Register the DataFrame and create table from it
    con.register('temp_patients_df', patients)
    con.execute(f"CREATE {temp_clause}TABLE {cohort_name} AS SELECT * FROM temp_patients_df")
    con.unregister('temp_patients_df')

    print(f"[INFO] Cohort table '{cohort_name}' created successfully")

    return num_patients


def create_multiple_cohorts(con, cohort_configs, temporary=True):
    """
    Create multiple cohort tables at once.

    Parameters:
    -----------
    con : duckdb.Connection
        DuckDB connection
    cohort_configs : list of dict
        List of cohort configurations, each with keys:
        - 'cohort_name': str
        - 'diagnosis_group': str
        - 'data_source': str (optional, default 'both')
    temporary : bool
        If True, creates TEMP tables. If False, creates permanent tables.

    Returns:
    --------
    dict : Mapping of cohort_name to patient count

    Example:
    --------
    cohorts = [
        {'cohort_name': 'dysphagia_cohort', 'diagnosis_group': 'dysphagia_core'},
        {'cohort_name': 'aspiration_cohort', 'diagnosis_group': 'aspiration_pneumonitis'},
        {'cohort_name': 'stroke_cohort', 'diagnosis_group': 'neurologic_risk_factors'}
    ]
    results = create_multiple_cohorts(con, cohorts)
    """
    results = {}

    print(f"\n[INFO] Creating {len(cohort_configs)} cohort tables...")
    print("=" * 60)

    for i, config in enumerate(cohort_configs, 1):
        cohort_name = config['cohort_name']
        diagnosis_group = config['diagnosis_group']
        data_source = config.get('data_source', 'both')

        print(f"\n[{i}/{len(cohort_configs)}] Processing {cohort_name}...")

        count = create_cohort_table(
            con=con,
            cohort_name=cohort_name,
            diagnosis_group=diagnosis_group,
            data_source=data_source,
            temporary=temporary
        )

        results[cohort_name] = count

    print("\n" + "=" * 60)
    print("[INFO] All cohorts created successfully!")
    print("\nSummary:")
    for cohort_name, count in results.items():
        print(f"  - {cohort_name}: {count:,} patients")

    return results


def get_sample_filter(sample_pct):
    """
    Get the SAMPLE_GROUP filter for a given sample percentage.

    CMS sample groups:
    - '01', '04': 5% sample
    - '15': Additional 15% (combined with above = 20%)
    - NULL or other: Remaining population

    Parameters:
    -----------
    sample_pct : int or None
        Sample percentage (5, 20, or None for all)

    Returns:
    --------
    tuple: (sql_where_clause, description)
    """
    if sample_pct is None or sample_pct == 100:
        return None, "all patients (100%)"
    elif sample_pct == 5:
        return "SAMPLE_GROUP IN ('01', '04')", "5% sample"
    elif sample_pct == 20:
        return "SAMPLE_GROUP IN ('01', '04', '15')", "20% sample"
    else:
        raise ValueError(f"Invalid sample size: {sample_pct}. Use 5, 20, or omit for all patients.")


def run_cohort_analysis(con, diagnosis_group, sample_pct=None):
    """
    Run cohort analysis for a diagnosis group with optional sample filtering.

    Parameters:
    -----------
    con : duckdb.Connection
        DuckDB connection
    diagnosis_group : str
        Diagnosis group from diagnoses.json
    sample_pct : int or None
        Sample percentage (5, 20, or None for all)
    """
    sample_filter, sample_desc = get_sample_filter(sample_pct)

    print(f"Analysis Configuration:")
    print(f"  Diagnosis group: {diagnosis_group}")
    print(f"  Sample: {sample_desc}")
    print("=" * 60)

    # Get total population count
    if sample_filter:
        # Create sample table
        print(f"\nCreating {sample_desc} beneficiary table...")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE sample_population AS
            SELECT DISTINCT DSYSRTKY
            FROM mbsf_all
            WHERE {sample_filter}
        """)
        total_population = con.execute("SELECT COUNT(*) FROM sample_population").fetchone()[0]
        print(f"  Sample contains {total_population:,} unique beneficiaries")
    else:
        # Count all unique beneficiaries
        print("\nCounting all beneficiaries...")
        total_population = con.execute("SELECT COUNT(DISTINCT DSYSRTKY) FROM mbsf_all").fetchone()[0]
        print(f"  Total population: {total_population:,} unique beneficiaries")

    # Create the diagnosis cohort
    print(f"\nIdentifying {diagnosis_group} patients...")
    print("-" * 60)
    num_full_cohort = create_cohort_table(
        con=con,
        cohort_name='diagnosis_cohort',
        diagnosis_group=diagnosis_group,
        data_source='both',
        temporary=True
    )

    # Filter to sample if needed
    if sample_filter:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE analysis_cohort AS
            SELECT d.DSYSRTKY
            FROM diagnosis_cohort d
            INNER JOIN sample_population s ON d.DSYSRTKY = s.DSYSRTKY
        """)
        cohort_count = con.execute("SELECT COUNT(*) FROM analysis_cohort").fetchone()[0]
        print(f"\nFull cohort: {num_full_cohort:,} patients")
        print(f"Filtered to {sample_desc}: {cohort_count:,} patients")
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE analysis_cohort AS SELECT * FROM diagnosis_cohort")
        cohort_count = num_full_cohort
        print(f"\nCohort size: {cohort_count:,} patients")

    # Run summary statistics
    print(f"\nSummary Statistics ({sample_desc})")
    print("-" * 60)

    result = con.execute("""
        SELECT
            COUNT(DISTINCT d.DSYSRTKY) as unique_patients,
            COUNT(*) as total_claims,
            COUNT(DISTINCT i.CLAIMNO) as unique_claims,
            ROUND(AVG(CAST(i.TOT_CHRG AS DECIMAL)), 2) as avg_total_charge,
            ROUND(SUM(CAST(i.TOT_CHRG AS DECIMAL)), 2) as total_charges,
            COUNT(CASE WHEN i.PTNTSTUS = 'B' THEN 1 END) as deaths
        FROM inp_claimsk_all i
        INNER JOIN analysis_cohort d ON i.DSYSRTKY = d.DSYSRTKY
    """).fetchall()

    if result:
        row = result[0]
        print(f"  Total patients in sample: {total_population:,}")
        print(f"  {diagnosis_group} patients: {cohort_count:,} ({100*cohort_count/total_population:.2f}%)")
        print(f"  Patients with inpatient claims: {row[0]:,}")
        print(f"  Total inpatient claim records: {row[1]:,}")
        print(f"  Unique inpatient claim IDs: {row[2]:,}")
        if row[3]:
            print(f"  Average total charge: ${row[3]:,.2f}")
            print(f"  Total charges: ${row[4]:,.2f}")
        if row[0] and row[0] > 0:
            print(f"  Deaths (PTNTSTUS='B'): {row[5]:,} ({100*row[5]/row[0]:.1f}%)")

    print("\n" + "=" * 60)
    return cohort_count


# =============================================================================
# Main execution / CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate patient cohort subsets for CMS claims analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_subset.py                      # All patients (100%)
  python generate_subset.py --sample 5           # 5% sample
  python generate_subset.py --sample 20          # 20% sample
  python generate_subset.py -s 5 -d aspiration_pneumonitis
        """
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        choices=[5, 20],
        default=None,
        help="Sample size percentage (5 or 20). Omit for all patients."
    )
    parser.add_argument(
        "-d", "--diagnosis",
        type=str,
        default="dysphagia_core",
        help="Diagnosis group from diagnoses.json (default: dysphagia_core)"
    )

    args = parser.parse_args()

    load_dotenv()

    # Connect to DuckDB
    con = duckdb.connect(database="cms_data.duckdb", read_only=False)
    print("[INFO] Connected to cms_data.duckdb\n")

    try:
        run_cohort_analysis(con, args.diagnosis, args.sample)
    finally:
        con.close()

    print("\n[INFO] Analysis complete!")
