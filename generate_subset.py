#!/usr/bin/env python3
"""
Generate Patient Cohort Subsets

Creates cohort tables for specific diagnosis groups to enable efficient
subset analysis without repeatedly scanning the full claims data.
"""

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


# =============================================================================
# Main execution / Example usage
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    # Connect to DuckDB
    con = duckdb.connect(database="cms_viewer.duckdb", read_only=False)
    print("[INFO] Connected to cms_viewer.duckdb\n")

    # Example 1: Create a single cohort
    print("Example 1: Creating dysphagia cohort")
    print("-" * 60)
    num_dysphagia = create_cohort_table(
        con=con,
        cohort_name='dysphagia_cohort',
        diagnosis_group='dysphagia_core',
        data_source='both',
        temporary=True
    )

    print(f"\nCreated dysphagia_cohort with {num_dysphagia:,} patients")

    # Example query using the cohort
    print("\nExample query: Count inpatient admissions for dysphagia patients")
    result = con.execute("""
        SELECT COUNT(*) as num_admissions
        FROM inp_claimsk_all i
        INNER JOIN dysphagia_cohort d ON i.DSYSRTKY = d.DSYSRTKY
    """).fetchone()
    print(f"Number of inpatient admissions: {result[0]:,}")

    print("\n" + "=" * 60)

    # Example 2: Create multiple cohorts at once
    print("\nExample 2: Creating multiple cohorts")
    print("-" * 60)

    cohort_configs = [
        {
            'cohort_name': 'aspiration_cohort',
            'diagnosis_group': 'aspiration_pneumonitis',
            'data_source': 'both'
        },
        {
            'cohort_name': 'pneumonia_cohort',
            'diagnosis_group': 'pneumonia_sensitivity',
            'data_source': 'both'
        }
    ]

    cohort_counts = create_multiple_cohorts(con, cohort_configs, temporary=True)

    # Example: Find overlap between dysphagia and aspiration
    print("\n" + "=" * 60)
    print("\nExample analysis: Overlap between dysphagia and aspiration")
    overlap = con.execute("""
        SELECT COUNT(*) as overlap_count
        FROM dysphagia_cohort d
        INNER JOIN aspiration_cohort a ON d.DSYSRTKY = a.DSYSRTKY
    """).fetchone()

    print(f"Patients with both dysphagia and aspiration: {overlap[0]:,}")
    print(f"Percentage of dysphagia patients: {100 * overlap[0] / num_dysphagia:.1f}%")

    con.close()
    print("\n[INFO] Analysis complete!")
