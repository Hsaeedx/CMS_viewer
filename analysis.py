#!/usr/bin/env python3
"""
Analysis utilities for CMS claims data.

Provides functions for building event-level tables from claims data
using diagnosis groups defined in diagnoses.json.

Usage:
    from analysis import build_event_table

    # Build stroke events from inpatient and outpatient claims for 2019
    df = build_event_table('stroke', 2019, ['inp', 'out'])

    # Build dysphagia events from all sources, all years
    df = build_event_table('dysphagia', 'all', ['inp', 'out', 'car', 'snf', 'hha'])

    # Custom date field policy
    df = build_event_table('stroke', 2019, ['inp', 'snf'], date_field_policy={'inp': 'DSCHRGDT', 'snf': 'DSCHRGDT'})
"""

import duckdb
import pandas as pd

from build_cohort import (
    load_diagnosis_codes,
    build_icd_where_clause_inpatient,
    build_icd_where_primary_clause_inpatient,
    build_icd_where_clause_outpatient,
    build_icd_where_clause_carrier,
    build_icd_where_clause_snf,
    build_icd_where_clause_hha,
)

AVAILABLE_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

DEFAULT_DATE_POLICY = {
    'inp': 'ADMSN_DT',
    'inp_primary': 'ADMSN_DT',
    'out': 'THRU_DT',
    'car': 'THRU_DT',
    'snf': 'ADMSN_DT',
    'hha': 'THRU_DT',
}

SOURCE_CONFIG = {
    'inp': {
        'table_prefix': 'inp_claimsk',
        'where_builder': build_icd_where_clause_inpatient,
    },
    'inp_primary': {
        'table_prefix': 'inp_claimsk',
        'where_builder': build_icd_where_primary_clause_inpatient,
    },
    'out': {
        'table_prefix': 'out_claimsk',
        'where_builder': build_icd_where_clause_outpatient,
    },
    'car': {
        'table_prefix': 'car_claimsk',
        'where_builder': build_icd_where_clause_carrier,
    },
    'snf': {
        'table_prefix': 'snf_claimsk',
        'where_builder': build_icd_where_clause_snf,
    },
    'hha': {
        'table_prefix': 'hha_claimsk',
        'where_builder': build_icd_where_clause_hha,
    },
}


def _get_sample_filter(sample_pct):
    """Get SAMPLE_GROUP filter and table suffix for a sample percentage."""
    if sample_pct is None:
        return None, ""
    elif sample_pct == 5:
        return "SAMPLE_GROUP IN ('01', '04')", "_5pct"
    elif sample_pct == 20:
        return "SAMPLE_GROUP IN ('01', '04', '15')", "_20pct"
    else:
        raise ValueError(f"Invalid sample size: {sample_pct}. Use 5 or 20.")


def build_event_table(group, year, sources, date_field_policy=None, sample_pct=None):
    """
    Build an event-level table from claims data for a diagnosis group.

    Scans the specified claim sources for claims matching the diagnosis codes,
    extracts one row per claim event, and persists the result to DuckDB.

    Args:
        group: Diagnosis group key from diagnoses.json (e.g., 'stroke', 'dysphagia')
        year: Year to scan (int, e.g. 2019) or 'all' for all available years
        sources: List of claim sources to scan (e.g., ['inp', 'out', 'car', 'snf', 'hha'])
        date_field_policy: Dict mapping source -> date column name. If None, uses defaults:
                           inp=ADMSN_DT, out=THRU_DT, car=THRU_DT, snf=ADMSN_DT, hha=THRU_DT
        sample_pct: Sample percentage (5, 20, or None for all patients)

    Returns:
        DataFrame with columns: DSYSRTKY, event_date, source
        Also persists as a DuckDB table named {group}_events_{year}
    """
    # Validate sources
    invalid = [s for s in sources if s not in SOURCE_CONFIG]
    if invalid:
        raise ValueError(f"Invalid sources: {invalid}. Valid: {list(SOURCE_CONFIG.keys())}")

    # Resolve date field policy
    policy = dict(DEFAULT_DATE_POLICY)
    if date_field_policy:
        policy.update(date_field_policy)

    # Load diagnosis codes
    codes = load_diagnosis_codes(group)
    print(f"[INFO] Loaded {len(codes)} diagnosis codes for '{group}'")

    # Determine years to scan
    if year == 'all':
        years = AVAILABLE_YEARS
    else:
        years = [int(year)]

    # Sample filter
    sample_filter, sample_suffix = _get_sample_filter(sample_pct)

    # Connect to DuckDB (read/write to persist table)
    con = duckdb.connect(database="cms_data.duckdb", read_only=False)

    # Get available tables
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    available_tables = {t[0].lower() for t in tables}

    # Build union queries across all years and sources
    unions = []
    skipped = []

    for y in years:
        for src in sources:
            config = SOURCE_CONFIG[src]
            table_name = f"{config['table_prefix']}_{y}"

            if table_name.lower() not in available_tables:
                skipped.append(f"{src}/{y}")
                continue

            dx_where = config['where_builder'](codes)
            date_field = policy[src]

            date_filter = f"AND SUBSTRING({date_field}, 1, 4) = '{y}'"

            if sample_filter:
                mbsf_table = f"mbsf_{y}"
                unions.append(f"""
                    SELECT c.DSYSRTKY,
                           STRPTIME(c.{date_field}, '%Y%m%d')::DATE AS event_date,
                           '{src}' AS source
                    FROM {table_name} c
                    INNER JOIN {mbsf_table} m ON c.DSYSRTKY = m.DSYSRTKY
                    WHERE ({dx_where}) AND {sample_filter}
                      {date_filter.replace(date_field, 'c.' + date_field)}
                """)
            else:
                unions.append(f"""
                    SELECT DSYSRTKY,
                           STRPTIME({date_field}, '%Y%m%d')::DATE AS event_date,
                           '{src}' AS source
                    FROM {table_name}
                    WHERE ({dx_where})
                      {date_filter}
                """)

    if not unions:
        print(f"[WARN] No matching tables found for sources={sources}, year={year}")
        con.close()
        return pd.DataFrame(columns=['DSYSRTKY', 'event_date', 'source'])

    if skipped:
        print(f"[WARN] Skipped (table not found): {', '.join(skipped)}")

    # Build the full query
    full_query = "\nUNION ALL\n".join(unions)

    # Persist to DuckDB
    year_label = 'all' if year == 'all' else str(year)
    output_table = f"{group}_events_{year_label}{sample_suffix}"

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(f"CREATE TABLE {output_table} AS {full_query}")

    # Get count and fetch as DataFrame
    count = con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
    # df = con.execute(f"SELECT * FROM {output_table}").fetchdf()

    con.close()

    print(f"[INFO] Created table '{output_table}': {count:,} events")
    print(f"[INFO] Sources scanned: {', '.join(sources)}")
    print(f"[INFO] Years: {', '.join(str(y) for y in years)}")

    # return df
