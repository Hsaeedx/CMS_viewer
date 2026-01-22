#!/usr/bin/env python3
"""
Diagnosis Analysis

Analyzes cohort tables from cms_data.duckdb and produces statistics, graphs, and CSV output.

Usage:
    python cohort_analysis.py prevalence dysphagia_core
    python cohort_analysis.py prevalence dysphagia_core --sample 5
    python cohort_analysis.py prevalence dysphagia_core -o ./output
    python cohort_analysis.py prevalence dysphagia_core --lookback 2

Available analyses:
    prevalence  - Yearly prevalence rates with graph and CSV output
"""

import argparse
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

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
    max_count = df['cohort_count'].max()
    ax1.set_ylim(bottom=0, top=max_count * 1.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Line chart for prevalence on secondary axis
    ax2 = ax1.twinx()
    color_prev = '#E94F37'
    ax2.plot(df['year'], df['prevalence_per_1000'], color=color_prev, marker='o',
             linewidth=2, markersize=8, label='Prevalence per 1,000')
    ax2.set_ylabel('Prevalence per 1,000', color=color_prev, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_prev)
    max_prev = df['prevalence_per_1000'].max()
    ax2.set_ylim(bottom=0, top=max_prev * 1.4)

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


# =============================================================================
# ANALYSIS REGISTRY - Add new analysis functions here
# =============================================================================

ANALYSES = {
    'prevalence': {
        'func': analyze_prevalence,
        'help': 'Calculate yearly prevalence rates for a cohort'
    },
    # Add more analyses here:
    # 'demographics': {
    #     'func': analyze_demographics,
    #     'help': 'Analyze demographic breakdown of cohort'
    # },
    # 'comorbidities': {
    #     'func': analyze_comorbidities,
    #     'help': 'Analyze common comorbidities in cohort'
    # },
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

    args = parser.parse_args()

    load_dotenv()

    # Get and run the analysis function
    analysis_info = ANALYSES[args.analysis]
    analysis_func = analysis_info['func']

    print(f"\n{'='*60}")
    print(f"Running {args.analysis} analysis for: {args.cohort}")
    print(f"{'='*60}\n")

    result = analysis_func(
        cohort_name=args.cohort,
        sample_pct=args.sample,
        output_dir=args.output,
        lookback_window=args.lookback
    )

    if result is not None:
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
