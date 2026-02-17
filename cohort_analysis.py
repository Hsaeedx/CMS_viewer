#!/usr/bin/env python3
"""
Cohort Analysis

Analyzes cohort tables from cms_data.duckdb and produces statistics, graphs, and CSV output.

Usage:
    python cohort_analysis.py prevalence dysphagia_core
    python cohort_analysis.py prevalence dysphagia_core --sample 5
    python cohort_analysis.py prevalence dysphagia_core -o ./output
    python cohort_analysis.py prevalence dysphagia_core --lookback 2
    python cohort_analysis.py inpatient dysphagia_core --lookback 2
    python cohort_analysis.py inpatient_pop dysphagia_core -p stroke_core --lookback 2

Available analyses:
    prevalence    - Yearly prevalence rates with graph and CSV output
    inpatient     - Inpatient utilization (admission rate, admits/person, LOS)
    inpatient_pop - Compare inpatient metrics within a population (requires -p)
                    Includes 95% CI and p-values for comparisons
"""

import argparse
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from scipy import stats

# Available years in the database
AVAILABLE_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]


def get_connection(read_only=True):
    """Get DuckDB connection."""
    return duckdb.connect(database="cms_data.duckdb", read_only=read_only)


def get_sample_suffix(sample_pct):
    """Get table suffix for sample percentage."""
    if sample_pct is None:
        return ""
    elif sample_pct == 5:
        return "_5pct"
    elif sample_pct == 20:
        return "_20pct"
    else:
        raise ValueError(f"Invalid sample size: {sample_pct}. Use 5 or 20.")


def get_sample_filter(sample_pct):
    """Get the SAMPLE_GROUP filter for a given sample percentage."""
    if sample_pct is None:
        return None
    elif sample_pct == 5:
        return "SAMPLE_GROUP IN ('01', '04')"
    elif sample_pct == 20:
        return "SAMPLE_GROUP IN ('01', '04', '15')"
    else:
        raise ValueError(f"Invalid sample size: {sample_pct}. Use 5 or 20.")


def check_cohort_tables_exist(con, cohort_name, sample_suffix):
    """Check if required cohort tables exist."""
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    available = {t[0].lower() for t in tables}

    missing = []
    for year in AVAILABLE_YEARS:
        table_name = f"{cohort_name}_{year}{sample_suffix}".lower()
        if table_name not in available:
            missing.append(f"{cohort_name}_{year}{sample_suffix}")

    return missing


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_prevalence(cohort_name, sample_pct=None, output_dir=None, lookback_window=0):
    """
    Calculate yearly prevalence for a cohort.

    Prevalence = (patients with diagnosis in current or prior N years who are alive and enrolled)
                 / (total enrolled population)

    Args:
        cohort_name: Name of the cohort (e.g., 'dysphagia_core')
        sample_pct: Sample percentage (5, 20, or None for all)
        output_dir: Directory for output files (default: current directory)
        lookback_window: Number of prior years to include (default: 0, meaning current year only)

    Returns:
        DataFrame with yearly prevalence data
    """
    con = get_connection()
    sample_suffix = get_sample_suffix(sample_pct)
    sample_filter = get_sample_filter(sample_pct)

    # Check if cohort tables exist
    missing = check_cohort_tables_exist(con, cohort_name, sample_suffix)
    if missing:
        print(f"[ERROR] Missing cohort tables: {missing}")
        print(f"[INFO] Run: python build_cohort.py {cohort_name}" +
              (f" --sample {sample_pct}" if sample_pct else ""))
        con.close()
        return None

    # Set up output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Calculating yearly prevalence for '{cohort_name}'...")
    if lookback_window == 0:
        print(f"[INFO] Using current year only (no lookback)")
    else:
        print(f"[INFO] Using {lookback_window}-year lookback (diagnosis in current year OR prior {lookback_window} years)")
    print(f"[INFO] Patients must be in current year cohort, alive, and enrolled in the reporting year")
    if sample_pct:
        print(f"[INFO] Using {sample_pct}% sample")

    results = []

    # Years to report (need lookback_window years of prior data)
    min_year = min(AVAILABLE_YEARS) + lookback_window
    report_years = [y for y in AVAILABLE_YEARS if y >= min_year]

    for year in report_years:
        mbsf_table = f"mbsf_{year}"

        # Lookback years: current year and N prior years based on lookback_window
        lookback_years = [y for y in range(year - lookback_window, year + 1) if y in AVAILABLE_YEARS]

        # Get enrollment filter
        enrollment_filter = "A_MO_CNT = '12' AND B_MO_CNT = '12' AND HMO_MO = '0'"
        if sample_filter:
            enrollment_filter += f" AND {sample_filter}"

        # Build union of all cohort tables for the lookback period (current year + prior years)
        # This gives us patients diagnosed in current year OR any prior year within the window
        cohort_unions = " UNION ".join([
            f"SELECT DSYSRTKY FROM {cohort_name}_{y}{sample_suffix}"
            for y in lookback_years
        ])

        # Count patients who had diagnosis in current year OR prior years within lookback
        # AND are alive AND enrolled this year
        # Alive = no death date OR death date is in or after this year
        prevalent_count = con.execute(f"""
            SELECT COUNT(DISTINCT c.DSYSRTKY)
            FROM ({cohort_unions}) c
            INNER JOIN {mbsf_table} m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE {enrollment_filter}
              AND (m.DEATH_DT IS NULL OR m.DEATH_DT = '' OR SUBSTRING(m.DEATH_DT, 1, 4) >= '{year}')
        """).fetchone()[0]

        # Get total enrolled population for this year
        total_enrolled = con.execute(f"""
            SELECT COUNT(*) FROM {mbsf_table}
            WHERE {enrollment_filter}
        """).fetchone()[0]

        # Calculate prevalence (per 1000)
        prevalence_per_1000 = (prevalent_count / total_enrolled * 1000) if total_enrolled > 0 else 0

        results.append({
            'year': year,
            'cohort_count': prevalent_count,
            'total_enrolled': total_enrolled,
            'prevalence_per_1000': round(prevalence_per_1000, 2)
        })

        print(f"  {year}: {prevalent_count:,} / {total_enrolled:,} = {prevalence_per_1000:.2f} per 1,000 (lookback: {lookback_years})")

    con.close()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Generate output file names
    lookback_suffix = f"_lw_{lookback_window}" if lookback_window > 0 else ""
    base_name = f"{cohort_name}_prevalence{sample_suffix}{lookback_suffix}"
    csv_path = output_dir / f"{base_name}.csv"
    png_path = output_dir / f"{base_name}.png"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved CSV: {csv_path}")

    # Generate plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for counts
    color_cohort = '#2E86AB'
    ax1.bar(df['year'], df['cohort_count'], color=color_cohort, alpha=0.7, label='Cohort Count')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cohort Count', color=color_cohort, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_cohort)
    ax1.set_xticks(df['year'])
    ax1.set_ylim(bottom=0, top=200000)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Line chart for prevalence on secondary axis
    ax2 = ax1.twinx()
    color_prev = '#E94F37'
    ax2.plot(df['year'], df['prevalence_per_1000'], color=color_prev, marker='o',
             linewidth=2, markersize=8, label='Prevalence per 1,000')
    ax2.set_ylabel('Prevalence per 1,000', color=color_prev, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_prev)
    ax2.set_ylim(bottom=0, top=5)

    # Title and formatting
    title = f"Yearly Prevalence: {cohort_name.replace('_', ' ').title()}"
    title_parts = []
    if lookback_window > 0:
        title_parts.append(f"{lookback_window}-yr lookback")
    if sample_pct:
        title_parts.append(f"{sample_pct}% Sample")
    if title_parts:
        title += f" ({', '.join(title_parts)})"
    plt.title(title, fontsize=14, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved graph: {png_path}")

    return df


def analyze_inpatient(cohort_name, sample_pct=None, output_dir=None, lookback_window=0):
    """
    Calculate inpatient metrics for a cohort as a share of all inpatients.

    Metrics:
        - Cohort inpatient rate: % of all inpatients who are in the cohort
        - Cohort admissions per person: Average admissions among cohort inpatients
        - Cohort avg LOS: Average length of stay for cohort inpatients

    Args:
        cohort_name: Name of the cohort (e.g., 'dysphagia_core')
        sample_pct: Sample percentage (5, 20, or None for all)
        output_dir: Directory for output files (default: current directory)
        lookback_window: Number of prior years to include (default: 0, meaning current year only)

    Returns:
        DataFrame with yearly inpatient data
    """
    con = get_connection()
    sample_suffix = get_sample_suffix(sample_pct)
    sample_filter = get_sample_filter(sample_pct)

    # Check if cohort tables exist
    missing = check_cohort_tables_exist(con, cohort_name, sample_suffix)
    if missing:
        print(f"[ERROR] Missing cohort tables: {missing}")
        print(f"[INFO] Run: python build_cohort.py {cohort_name}" +
              (f" --sample {sample_pct}" if sample_pct else ""))
        con.close()
        return None

    # Set up output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Calculating inpatient share for '{cohort_name}'...")
    if lookback_window == 0:
        print(f"[INFO] Using current year only (no lookback)")
    else:
        print(f"[INFO] Using {lookback_window}-year lookback (diagnosis in current year OR prior {lookback_window} years)")
    if sample_pct:
        print(f"[INFO] Using {sample_pct}% sample")

    results = []

    # Years to report (need lookback_window years of prior data)
    min_year = min(AVAILABLE_YEARS) + lookback_window
    report_years = [y for y in AVAILABLE_YEARS if y >= min_year]

    # Track aggregates across all years
    total_all_inpatients = 0
    total_cohort_inpatients = 0
    total_cohort_admissions = 0
    total_cohort_los_days = 0

    for year in report_years:
        inp_table = f"INP_claimsk_{year}"

        # Lookback years: current year and N prior years based on lookback_window
        lookback_years = [y for y in range(year - lookback_window, year + 1) if y in AVAILABLE_YEARS]

        # Build union of all cohort tables for the lookback period
        cohort_unions = " UNION ".join([
            f"SELECT DSYSRTKY FROM {cohort_name}_{y}{sample_suffix}"
            for y in lookback_years
        ])

        # Get total unique inpatients for this year
        all_inpatients = con.execute(f"""
            SELECT COUNT(DISTINCT DSYSRTKY) FROM {inp_table}
        """).fetchone()[0]

        # Get cohort inpatient stats for this year
        cohort_stats = con.execute(f"""
            SELECT
                COUNT(DISTINCT i.DSYSRTKY) as cohort_inpatients,
                COUNT(*) as cohort_admissions,
                SUM(CAST(COALESCE(NULLIF(i.UTIL_DAY, ''), '0') AS INTEGER)) as cohort_los_days
            FROM {inp_table} i
            INNER JOIN ({cohort_unions}) c ON i.DSYSRTKY = c.DSYSRTKY
        """).fetchone()

        cohort_inpatients = cohort_stats[0] or 0
        cohort_admissions = cohort_stats[1] or 0
        cohort_los_days = cohort_stats[2] or 0

        # Calculate metrics
        cohort_rate = (cohort_inpatients / all_inpatients * 100) if all_inpatients > 0 else 0
        admissions_per_person = (cohort_admissions / cohort_inpatients) if cohort_inpatients > 0 else 0
        avg_los = (cohort_los_days / cohort_admissions) if cohort_admissions > 0 else 0

        results.append({
            'year': year,
            'all_inpatients': all_inpatients,
            'cohort_inpatients': cohort_inpatients,
            'cohort_admissions': cohort_admissions,
            'cohort_los_days': cohort_los_days,
            'cohort_rate_pct': round(cohort_rate, 2),
            'admissions_per_person': round(admissions_per_person, 2),
            'avg_los_days': round(avg_los, 2)
        })

        # Update aggregates
        total_all_inpatients += all_inpatients
        total_cohort_inpatients += cohort_inpatients
        total_cohort_admissions += cohort_admissions
        total_cohort_los_days += cohort_los_days

        print(f"  {year}: {cohort_inpatients:,}/{all_inpatients:,} inpatients ({cohort_rate:.1f}%), "
              f"{admissions_per_person:.2f} admits/person, {avg_los:.1f} days avg LOS (lookback: {lookback_years})")

    con.close()

    # Add aggregate row
    agg_cohort_rate = (total_cohort_inpatients / total_all_inpatients * 100) if total_all_inpatients > 0 else 0
    agg_admissions_per_person = (total_cohort_admissions / total_cohort_inpatients) if total_cohort_inpatients > 0 else 0
    agg_avg_los = (total_cohort_los_days / total_cohort_admissions) if total_cohort_admissions > 0 else 0

    results.append({
        'year': 'ALL',
        'all_inpatients': total_all_inpatients,
        'cohort_inpatients': total_cohort_inpatients,
        'cohort_admissions': total_cohort_admissions,
        'cohort_los_days': total_cohort_los_days,
        'cohort_rate_pct': round(agg_cohort_rate, 2),
        'admissions_per_person': round(agg_admissions_per_person, 2),
        'avg_los_days': round(agg_avg_los, 2)
    })

    print(f"\n  AGGREGATE: {total_cohort_inpatients:,}/{total_all_inpatients:,} inpatients ({agg_cohort_rate:.1f}%), "
          f"{agg_admissions_per_person:.2f} admits/person, {agg_avg_los:.1f} days avg LOS")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Generate output file names
    lookback_suffix = f"_lw_{lookback_window}" if lookback_window > 0 else ""
    base_name = f"{cohort_name}_inpatient{sample_suffix}{lookback_suffix}"
    csv_path = output_dir / f"{base_name}.csv"
    png_path = output_dir / f"{base_name}.png"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved CSV: {csv_path}")

    # Generate plot (exclude aggregate row)
    df_plot = df[df['year'] != 'ALL'].copy()
    df_plot['year'] = df_plot['year'].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Cohort Inpatient Count
    ax1 = axes[0, 0]
    color1 = '#6A4C93'
    ax1.bar(df_plot['year'], df_plot['cohort_inpatients'], color=color1, alpha=0.7)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Cohort Inpatients', fontsize=11)
    ax1.set_title('Inpatients with Condition', fontsize=12, fontweight='bold')
    ax1.set_xticks(df_plot['year'])
    ax1.set_ylim(bottom=0, top=df_plot['cohort_inpatients'].max() * 1.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Plot 2: Cohort Rate (% of inpatients in cohort)
    ax2 = axes[0, 1]
    color2 = '#2E86AB'
    ax2.bar(df_plot['year'], df_plot['cohort_rate_pct'], color=color2, alpha=0.7)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Cohort Rate (%)', fontsize=11)
    ax2.set_title('% of Inpatients in Cohort', fontsize=12, fontweight='bold')
    ax2.set_xticks(df_plot['year'])
    ax2.set_ylim(bottom=0, top=df_plot['cohort_rate_pct'].max() * 1.4)

    # Plot 3: Admissions per Person
    ax3 = axes[1, 0]
    color3 = '#E94F37'
    ax3.bar(df_plot['year'], df_plot['admissions_per_person'], color=color3, alpha=0.7)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Admissions per Person', fontsize=11)
    ax3.set_title('Admissions per Person', fontsize=12, fontweight='bold')
    ax3.set_xticks(df_plot['year'])
    ax3.set_ylim(bottom=0, top=df_plot['admissions_per_person'].max() * 1.4)

    # Plot 4: Average LOS
    ax4 = axes[1, 1]
    color4 = '#8AC926'
    ax4.bar(df_plot['year'], df_plot['avg_los_days'], color=color4, alpha=0.7)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Average LOS (days)', fontsize=11)
    ax4.set_title('Average Length of Stay', fontsize=12, fontweight='bold')
    ax4.set_xticks(df_plot['year'])
    ax4.set_ylim(bottom=0, top=df_plot['avg_los_days'].max() * 1.4)

    # Main title
    main_title = f"Inpatient Analysis: {cohort_name.replace('_', ' ').title()}"
    title_parts = []
    if lookback_window > 0:
        title_parts.append(f"{lookback_window}-yr lookback")
    if sample_pct:
        title_parts.append(f"{sample_pct}% Sample")
    if title_parts:
        main_title += f" ({', '.join(title_parts)})"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved graph: {png_path}")

    return df


def analyze_inpatient_population(cohort_name, sample_pct=None, output_dir=None, lookback_window=0, population=None):
    """
    Analyze inpatient metrics comparing cohort within a population vs population without cohort.

    Example: dysphagia within stroke population - compares stroke+dysphagia vs stroke-only inpatients.

    Args:
        cohort_name: Name of the cohort (e.g., 'dysphagia_core')
        sample_pct: Sample percentage (5, 20, or None for all)
        output_dir: Directory for output files (default: current directory)
        lookback_window: Number of prior years to include (default: 0, meaning current year only)
        population: Population cohort to analyze within (e.g., 'stroke_core') - REQUIRED

    Returns:
        DataFrame with yearly comparison data including 95% CI and p-values
    """
    if not population:
        print("[ERROR] Population (-p) is required for inpatient_pop analysis")
        print("[INFO] Example: python cohort_analysis.py inpatient_pop dysphagia_core -p stroke_core")
        return None

    con = get_connection()
    sample_suffix = get_sample_suffix(sample_pct)

    # Check if both cohort tables exist
    missing_cohort = check_cohort_tables_exist(con, cohort_name, sample_suffix)
    missing_pop = check_cohort_tables_exist(con, population, sample_suffix)

    if missing_cohort:
        print(f"[ERROR] Missing cohort tables: {missing_cohort}")
        con.close()
        return None
    if missing_pop:
        print(f"[ERROR] Missing population tables: {missing_pop}")
        con.close()
        return None

    # Set up output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_label = cohort_name.split('_')[0].title()
    pop_label = population.split('_')[0].title()

    print(f"[INFO] Analyzing '{cohort_label}' within '{pop_label}' population...")
    if lookback_window == 0:
        print(f"[INFO] Using current year only (no lookback)")
    else:
        print(f"[INFO] Using {lookback_window}-year lookback")
    if sample_pct:
        print(f"[INFO] Using {sample_pct}% sample")

    results = []

    # Years to report
    min_year = min(AVAILABLE_YEARS) + lookback_window
    report_years = [y for y in AVAILABLE_YEARS if y >= min_year]

    # Track aggregates for statistical comparison
    all_with_cohort_admissions = []
    all_without_cohort_admissions = []
    all_with_cohort_los = []
    all_without_cohort_los = []

    for year in report_years:
        inp_table = f"INP_claimsk_{year}"

        # Lookback years
        lookback_years = [y for y in range(year - lookback_window, year + 1) if y in AVAILABLE_YEARS]

        # Build unions for both cohorts
        cohort_unions = " UNION ".join([
            f"SELECT DSYSRTKY FROM {cohort_name}_{y}{sample_suffix}"
            for y in lookback_years
        ])
        pop_unions = " UNION ".join([
            f"SELECT DSYSRTKY FROM {population}_{y}{sample_suffix}"
            for y in lookback_years
        ])

        # Get population inpatients who are ALSO in cohort (e.g., stroke patients WITH dysphagia)
        with_cohort_stats = con.execute(f"""
            SELECT
                COUNT(DISTINCT i.DSYSRTKY) as n_patients,
                COUNT(*) as n_admissions,
                SUM(CAST(COALESCE(NULLIF(i.UTIL_DAY, ''), '0') AS INTEGER)) as total_los
            FROM {inp_table} i
            INNER JOIN ({pop_unions}) p ON i.DSYSRTKY = p.DSYSRTKY
            INNER JOIN ({cohort_unions}) c ON i.DSYSRTKY = c.DSYSRTKY
        """).fetchone()

        # Get population inpatients who are NOT in cohort (e.g., stroke patients WITHOUT dysphagia)
        without_cohort_stats = con.execute(f"""
            SELECT
                COUNT(DISTINCT i.DSYSRTKY) as n_patients,
                COUNT(*) as n_admissions,
                SUM(CAST(COALESCE(NULLIF(i.UTIL_DAY, ''), '0') AS INTEGER)) as total_los
            FROM {inp_table} i
            INNER JOIN ({pop_unions}) p ON i.DSYSRTKY = p.DSYSRTKY
            WHERE i.DSYSRTKY NOT IN (SELECT DSYSRTKY FROM ({cohort_unions}))
        """).fetchone()

        # Get per-patient data for statistical tests
        with_cohort_per_patient = con.execute(f"""
            SELECT
                i.DSYSRTKY,
                COUNT(*) as n_admissions,
                SUM(CAST(COALESCE(NULLIF(i.UTIL_DAY, ''), '0') AS INTEGER)) as total_los
            FROM {inp_table} i
            INNER JOIN ({pop_unions}) p ON i.DSYSRTKY = p.DSYSRTKY
            INNER JOIN ({cohort_unions}) c ON i.DSYSRTKY = c.DSYSRTKY
            GROUP BY i.DSYSRTKY
        """).fetchdf()

        without_cohort_per_patient = con.execute(f"""
            SELECT
                i.DSYSRTKY,
                COUNT(*) as n_admissions,
                SUM(CAST(COALESCE(NULLIF(i.UTIL_DAY, ''), '0') AS INTEGER)) as total_los
            FROM {inp_table} i
            INNER JOIN ({pop_unions}) p ON i.DSYSRTKY = p.DSYSRTKY
            WHERE i.DSYSRTKY NOT IN (SELECT DSYSRTKY FROM ({cohort_unions}))
            GROUP BY i.DSYSRTKY
        """).fetchdf()

        # Extract values
        with_n = with_cohort_stats[0] or 0
        with_admissions = with_cohort_stats[1] or 0
        with_los = with_cohort_stats[2] or 0

        without_n = without_cohort_stats[0] or 0
        without_admissions = without_cohort_stats[1] or 0
        without_los = without_cohort_stats[2] or 0

        total_pop = with_n + without_n

        # Calculate rates
        cohort_rate = (with_n / total_pop * 100) if total_pop > 0 else 0
        with_admits_pp = (with_admissions / with_n) if with_n > 0 else 0
        without_admits_pp = (without_admissions / without_n) if without_n > 0 else 0

        # Statistical tests and CI calculations
        admits_p = None
        los_p = None

        def mean_ci(data):
            if len(data) == 0:
                return (0, 0, 0)
            mean = np.mean(data)
            se = stats.sem(data) if len(data) > 1 else 0
            ci_low = mean - 1.96 * se
            ci_high = mean + 1.96 * se
            return (mean, ci_low, ci_high)

        if len(with_cohort_per_patient) > 0 and len(without_cohort_per_patient) > 0:
            # Mann-Whitney U test for admissions
            admits_stat, admits_p = stats.mannwhitneyu(
                with_cohort_per_patient['n_admissions'],
                without_cohort_per_patient['n_admissions'],
                alternative='two-sided'
            )

            # LOS per admission for each patient
            with_los_per_admit = with_cohort_per_patient['total_los'] / with_cohort_per_patient['n_admissions']
            without_los_per_admit = without_cohort_per_patient['total_los'] / without_cohort_per_patient['n_admissions']

            los_stat, los_p = stats.mannwhitneyu(
                with_los_per_admit,
                without_los_per_admit,
                alternative='two-sided'
            )

            # Collect for aggregate stats
            all_with_cohort_admissions.extend(with_cohort_per_patient['n_admissions'].tolist())
            all_without_cohort_admissions.extend(without_cohort_per_patient['n_admissions'].tolist())
            all_with_cohort_los.extend(with_los_per_admit.tolist())
            all_without_cohort_los.extend(without_los_per_admit.tolist())

        # Calculate CIs
        with_admits_mean, with_admits_ci_low, with_admits_ci_high = mean_ci(
            with_cohort_per_patient['n_admissions'].tolist() if len(with_cohort_per_patient) > 0 else []
        )
        without_admits_mean, without_admits_ci_low, without_admits_ci_high = mean_ci(
            without_cohort_per_patient['n_admissions'].tolist() if len(without_cohort_per_patient) > 0 else []
        )

        with_los_data = (with_cohort_per_patient['total_los'] / with_cohort_per_patient['n_admissions']).tolist() if len(with_cohort_per_patient) > 0 else []
        without_los_data = (without_cohort_per_patient['total_los'] / without_cohort_per_patient['n_admissions']).tolist() if len(without_cohort_per_patient) > 0 else []

        with_los_mean, with_los_ci_low, with_los_ci_high = mean_ci(with_los_data)
        without_los_mean, without_los_ci_low, without_los_ci_high = mean_ci(without_los_data)

        results.append({
            'year': year,
            'pop_total_inpatients': total_pop,
            'with_cohort_n': with_n,
            'without_cohort_n': without_n,
            'cohort_rate_pct': round(cohort_rate, 2),
            'with_cohort_admits_pp': round(with_admits_mean, 2),
            'with_cohort_admits_ci': f"({with_admits_ci_low:.2f}-{with_admits_ci_high:.2f})",
            'without_cohort_admits_pp': round(without_admits_mean, 2),
            'without_cohort_admits_ci': f"({without_admits_ci_low:.2f}-{without_admits_ci_high:.2f})",
            'admits_p_value': f"{admits_p:.4f}" if admits_p is not None else "N/A",
            'with_cohort_avg_los': round(with_los_mean, 2),
            'with_cohort_los_ci': f"({with_los_ci_low:.2f}-{with_los_ci_high:.2f})",
            'without_cohort_avg_los': round(without_los_mean, 2),
            'without_cohort_los_ci': f"({without_los_ci_low:.2f}-{without_los_ci_high:.2f})",
            'los_p_value': f"{los_p:.4f}" if los_p is not None else "N/A"
        })

        print(f"  {year}: {pop_label} inpatients: {total_pop:,}")
        print(f"       With {cohort_label}: {with_n:,} ({cohort_rate:.1f}%)")
        p_str = f" (p={admits_p:.4f})" if admits_p is not None else ""
        print(f"       Admits/person: {with_admits_mean:.2f} vs {without_admits_mean:.2f}{p_str}")
        p_str = f" (p={los_p:.4f})" if los_p is not None else ""
        print(f"       Avg LOS: {with_los_mean:.1f} vs {without_los_mean:.1f} days{p_str}")

    con.close()

    # Aggregate statistics
    if len(all_with_cohort_admissions) > 0 and len(all_without_cohort_admissions) > 0:
        agg_admits_stat, agg_admits_p = stats.mannwhitneyu(
            all_with_cohort_admissions, all_without_cohort_admissions, alternative='two-sided'
        )
        agg_los_stat, agg_los_p = stats.mannwhitneyu(
            all_with_cohort_los, all_without_cohort_los, alternative='two-sided'
        )

        def mean_ci_agg(data):
            mean = np.mean(data)
            se = stats.sem(data) if len(data) > 1 else 0
            return (mean, mean - 1.96 * se, mean + 1.96 * se)

        with_admits_agg = mean_ci_agg(all_with_cohort_admissions)
        without_admits_agg = mean_ci_agg(all_without_cohort_admissions)
        with_los_agg = mean_ci_agg(all_with_cohort_los)
        without_los_agg = mean_ci_agg(all_without_cohort_los)

        total_with = sum(r['with_cohort_n'] for r in results)
        total_without = sum(r['without_cohort_n'] for r in results)
        total_pop_agg = total_with + total_without

        results.append({
            'year': 'ALL',
            'pop_total_inpatients': total_pop_agg,
            'with_cohort_n': total_with,
            'without_cohort_n': total_without,
            'cohort_rate_pct': round(total_with / total_pop_agg * 100, 2) if total_pop_agg > 0 else 0,
            'with_cohort_admits_pp': round(with_admits_agg[0], 2),
            'with_cohort_admits_ci': f"({with_admits_agg[1]:.2f}-{with_admits_agg[2]:.2f})",
            'without_cohort_admits_pp': round(without_admits_agg[0], 2),
            'without_cohort_admits_ci': f"({without_admits_agg[1]:.2f}-{without_admits_agg[2]:.2f})",
            'admits_p_value': f"{agg_admits_p:.4f}",
            'with_cohort_avg_los': round(with_los_agg[0], 2),
            'with_cohort_los_ci': f"({with_los_agg[1]:.2f}-{with_los_agg[2]:.2f})",
            'without_cohort_avg_los': round(without_los_agg[0], 2),
            'without_cohort_los_ci': f"({without_los_agg[1]:.2f}-{without_los_agg[2]:.2f})",
            'los_p_value': f"{agg_los_p:.4f}"
        })

        print(f"\n  AGGREGATE:")
        print(f"       {pop_label} inpatients: {total_pop_agg:,}")
        print(f"       With {cohort_label}: {total_with:,} ({total_with/total_pop_agg*100:.1f}%)")
        print(f"       Admits/person: {with_admits_agg[0]:.2f} vs {without_admits_agg[0]:.2f} (p={agg_admits_p:.4f})")
        print(f"       Avg LOS: {with_los_agg[0]:.1f} vs {without_los_agg[0]:.1f} days (p={agg_los_p:.4f})")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Generate output file names
    lookback_suffix = f"_lw_{lookback_window}" if lookback_window > 0 else ""
    base_name = f"{cohort_name}_in_{population}_inpatient{sample_suffix}{lookback_suffix}"
    csv_path = output_dir / f"{base_name}.csv"
    png_path = output_dir / f"{base_name}.png"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved CSV: {csv_path}")

    # Generate plot (exclude aggregate row)
    df_plot = df[df['year'] != 'ALL'].copy()
    df_plot['year'] = df_plot['year'].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Count of population with cohort condition
    ax1 = axes[0, 0]
    color1 = '#6A4C93'
    ax1.bar(df_plot['year'], df_plot['with_cohort_n'], color=color1, alpha=0.7)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel(f'{pop_label} with {cohort_label}', fontsize=11)
    ax1.set_title(f'{pop_label} Inpatients with {cohort_label}', fontsize=12, fontweight='bold')
    ax1.set_xticks(df_plot['year'])
    ax1.set_ylim(bottom=0, top=df_plot['with_cohort_n'].max() * 1.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Plot 2: Rate of cohort within population
    ax2 = axes[0, 1]
    color2 = '#2E86AB'
    ax2.bar(df_plot['year'], df_plot['cohort_rate_pct'], color=color2, alpha=0.7)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel(f'% with {cohort_label}', fontsize=11)
    ax2.set_title(f'% of {pop_label} with {cohort_label}', fontsize=12, fontweight='bold')
    ax2.set_xticks(df_plot['year'])
    ax2.set_ylim(bottom=0, top=df_plot['cohort_rate_pct'].max() * 1.4)

    # Plot 3: Admissions per person comparison
    ax3 = axes[1, 0]
    width = 0.35
    x = np.arange(len(df_plot['year']))
    ax3.bar(x - width/2, df_plot['with_cohort_admits_pp'], width, label=f'With {cohort_label}', color='#E94F37', alpha=0.7)
    ax3.bar(x + width/2, df_plot['without_cohort_admits_pp'], width, label=f'Without {cohort_label}', color='#1982C4', alpha=0.7)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Admissions per Person', fontsize=11)
    ax3.set_title('Admissions per Person Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_plot['year'])
    max_admits = max(df_plot['with_cohort_admits_pp'].max(), df_plot['without_cohort_admits_pp'].max())
    ax3.set_ylim(bottom=0, top=max_admits * 1.4)
    ax3.legend()

    # Plot 4: Average LOS comparison
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, df_plot['with_cohort_avg_los'], width, label=f'With {cohort_label}', color='#8AC926', alpha=0.7)
    ax4.bar(x + width/2, df_plot['without_cohort_avg_los'], width, label=f'Without {cohort_label}', color='#6A4C93', alpha=0.7)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Average LOS (days)', fontsize=11)
    ax4.set_title('Average Length of Stay Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_plot['year'])
    max_los = max(df_plot['with_cohort_avg_los'].max(), df_plot['without_cohort_avg_los'].max())
    ax4.set_ylim(bottom=0, top=max_los * 1.4)
    ax4.legend()

    # Main title
    main_title = f"{cohort_label} in {pop_label} Inpatients"
    title_parts = []
    if lookback_window > 0:
        title_parts.append(f"{lookback_window}-yr lookback")
    if sample_pct:
        title_parts.append(f"{sample_pct}% Sample")
    if title_parts:
        main_title += f" ({', '.join(title_parts)})"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved graph: {png_path}")

    return df


# =============================================================================
# ANALYSIS REGISTRY - Add new analysis functions here
# =============================================================================

ANALYSES = {
    'prevalence': {
        'func': analyze_prevalence,
        'help': 'Calculate yearly prevalence rates for a cohort'
    },
    'inpatient': {
        'func': analyze_inpatient,
        'help': 'Calculate inpatient utilization (admission rate, admits/person, LOS)'
    },
    'inpatient_pop': {
        'func': analyze_inpatient_population,
        'help': 'Compare inpatient metrics within a population (requires -p)',
        'requires_population': True
    },
}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cohort tables from cms_data.duckdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnosis_analysis.py prevalence dysphagia_core
  python diagnosis_analysis.py prevalence dysphagia_core --sample 5
  python diagnosis_analysis.py prevalence stroke_core -o ./results

Available analyses:
""" + "\n".join(f"  {name}: {info['help']}" for name, info in ANALYSES.items())
    )

    parser.add_argument(
        "analysis",
        type=str,
        choices=list(ANALYSES.keys()),
        help="Type of analysis to run"
    )
    parser.add_argument(
        "cohort",
        type=str,
        help="Name of the cohort table (e.g., dysphagia_core, stroke_core)"
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        choices=[5, 20],
        default=None,
        help="Sample size percentage (5 or 20). Omit for all patients."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for CSV and graph files (default: current directory)"
    )
    parser.add_argument(
        "-l", "--lookback",
        type=int,
        default=0,
        help="Lookback window in years (default: 0, current year only)"
    )
    parser.add_argument(
        "-p", "--population",
        type=str,
        default=None,
        help="Population cohort for comparison (required for inpatient_pop analysis)"
    )

    args = parser.parse_args()

    load_dotenv()

    # Get and run the analysis function
    analysis_info = ANALYSES[args.analysis]
    analysis_func = analysis_info['func']

    print(f"\n{'='*60}")
    print(f"Running {args.analysis} analysis for: {args.cohort}")
    print(f"{'='*60}\n")

    # Build kwargs for analysis function
    kwargs = {
        'cohort_name': args.cohort,
        'sample_pct': args.sample,
        'output_dir': args.output,
        'lookback_window': args.lookback
    }

    # Add population if the analysis supports it
    if analysis_info.get('requires_population'):
        kwargs['population'] = args.population

    result = analysis_func(**kwargs)

    if result is not None:
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
