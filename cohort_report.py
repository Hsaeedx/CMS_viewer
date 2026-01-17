#!/usr/bin/env python3
"""cohort_report_optimized_fixed.py

Fixes vs your pasted version:
- **NO UNNEST** in incident-cohort building (this was the main OOM cause).
- Incident cohort is built from yearly claim tables using OR-across-dx-columns predicates,
  then dedup + LAG gap logic to define new episodes (365d clean).
- Non-dx comparison exclusion is built without UNNEST and using anti-join.
- Comorbidity dx unnesting is restricted to ONLY the cohort+comparison patients.
- PRAGMAs: preserve_insertion_order=false; low threads during cohort build; temp_directory points to a real folder.

Usage:
  python cohort_report_optimized_fixed.py --diagnosis dysphagia_core --memory 8GB --threads 2
"""

import argparse
import duckdb
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATABASE = "cms_data.duckdb"
DEFAULT_DIAGNOSES_FILE = "diagnoses.json"
DEFAULT_OUTPUT_DIR = "output"

LOOKBACK_DAYS = 365
MIN_YEAR_FOR_INCIDENT = 2017

CONT_SAMPLE_N = 200_000

PATIENT_ID = "DSYSRTKY"


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate analysis for a CMS diagnosis cohort (optimized + fixed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--diagnosis", "-d", required=True,
                        help="Diagnosis group name from diagnoses.json")
    parser.add_argument("--compare", "-c", default=None,
                        help="Comparison diagnosis group (incident cohort). If omitted, uses non-diagnosis population.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--database", "-db", default=DEFAULT_DATABASE,
                        help=f"DuckDB database path (default: {DEFAULT_DATABASE})")
    parser.add_argument("--diagnoses-file", "-df", default=DEFAULT_DIAGNOSES_FILE,
                        help=f"diagnoses.json path (default: {DEFAULT_DIAGNOSES_FILE})")
    parser.add_argument("--memory", "-m", default="6GB", help="DuckDB memory_limit")
    parser.add_argument("--threads", "-t", type=int, default=4, help="DuckDB threads")

    parser.add_argument("--no-coverage-filter", action="store_true",
                        help="Disable full-year FFS A/B coverage requirement (A_MO_CNT,B_MO_CNT,HMO_MO).")

    parser.add_argument("--cont-sample", type=int, default=CONT_SAMPLE_N,
                        help=f"Sample size for continuous variable tests (default: {CONT_SAMPLE_N})")

    parser.add_argument("--temp-dir", default=None,
                        help="DuckDB temp_directory. Strongly recommended to point to a large SSD path.")
    parser.add_argument("--max-temp", default="200GiB",
                        help="DuckDB max_temp_directory_size (default: 200GiB)")

    return parser.parse_args()


# =============================================================================
# DB
# =============================================================================

def connect_database(db_path: str,
                     memory_limit: str = "6GB",
                     threads: int = 4,
                     read_only: bool = True,
                     temp_dir: str | None = None,
                     max_temp: str = "200GiB",
                     low_thread_mode: bool = False):
    """Connect and set PRAGMAs.

    low_thread_mode=True is recommended for cohort building (reduces peak memory).
    """
    try:
        con = duckdb.connect(database=db_path, read_only=read_only)

        # Must be early
        con.execute("PRAGMA preserve_insertion_order=false;")

        # Temp spill settings
        if temp_dir:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{temp_dir}';")
            con.execute(f"PRAGMA max_temp_directory_size='{max_temp}';")

        # Memory + threads
        con.execute(f"PRAGMA memory_limit='{memory_limit}';")
        if low_thread_mode:
            con.execute("PRAGMA threads=2;")
        else:
            con.execute(f"PRAGMA threads={threads};")

        return con
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        return None


def check_data_availability(con) -> dict:
    required_views = ['inp_claimsk_all', 'out_claimsk_all', 'car_claimsk_all', 'mbsf_all']
    out = {}
    for v in required_views:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
            out[v] = (True, n)
        except Exception:
            out[v] = (False, None)
    return out


def print_data_status(availability: dict) -> bool:
    print("\n[INFO] Data Availability:")
    ok = True
    for view, (avail, cnt) in availability.items():
        print(f"  - {view}: {'OK ('+format(cnt,',')+' rows)' if avail else 'NOT AVAILABLE'}")
        ok = ok and avail
    if not ok:
        print("\n[ERROR] Some required views are missing.")
    return ok


# =============================================================================
# Diagnoses helpers
# =============================================================================

def load_diagnoses(diagnoses_file: str) -> dict:
    with open(diagnoses_file, 'r') as f:
        return json.load(f)


def get_diagnosis_codes(diagnoses: dict, group_name: str) -> tuple[list, str]:
    if group_name not in diagnoses:
        available = [k for k in diagnoses.keys() if k != "notes"]
        raise ValueError(f"Group '{group_name}' not found. Available: {available}")

    group = diagnoses[group_name]
    codes: list[str] = []
    match_type = "exact"

    if isinstance(group, dict) and "values" in group:
        codes = group["values"]
        match_type = group.get("match", "exact")
    elif isinstance(group, dict) and "malignancy_range" in group:
        sub = group["malignancy_range"]
        codes = sub.get("values", [])
        match_type = sub.get("match", "prefix")
    else:
        for _, value in (group or {}).items():
            if isinstance(value, dict) and "values" in value:
                codes.extend(value["values"])
                if value.get("match") == "prefix":
                    match_type = "prefix"

    if not codes:
        raise ValueError(f"No codes found for group '{group_name}'")

    return codes, match_type


def build_dx_match_expr(codes: list[str], match_type: str, column: str) -> str:
    if match_type == "prefix":
        return " OR ".join([f"SUBSTRING({column},1,{len(code)})='{code}'" for code in codes])
    quoted = "', '".join(codes)
    return f"{column} IN ('{quoted}')"


def build_any_dx_predicate(codes: list[str], match_type: str, cols: list[str]) -> str:
    # (match(col1) OR match(col2) OR ...)
    parts = [f"({build_dx_match_expr(codes, match_type, c)})" for c in cols]
    return " OR ".join(parts)


# =============================================================================
# Stats helpers
# =============================================================================

def chi_squared_test(counts1: list[int], counts2: list[int]) -> dict:
    try:
        table = np.array([counts1, counts2])
        chi2, p, dof, _ = chi2_contingency(table)
        n = table.sum()
        min_dim = min(table.shape) - 1
        v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
        return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": v}
    except Exception as e:
        return {"chi2": None, "p_value": None, "dof": None, "cramers_v": None, "error": str(e)}


def proportion_diff_ci(n1: int, x1: int, n2: int, x2: int, confidence: float = 0.95) -> dict:
    if n1 == 0 or n2 == 0:
        return {"diff": None, "ci_lower": None, "ci_upper": None, "p_value": None}
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    zcrit = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_l, ci_u = diff - zcrit * se, diff + zcrit * se
    if se_pool > 0:
        z = diff / se_pool
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p = 1.0
    return {"diff": float(diff), "ci_lower": float(ci_l), "ci_upper": float(ci_u), "p_value": float(p)}


def welch_t_test(values1: list[float], values2: list[float]) -> dict:
    v1 = [v for v in values1 if v is not None and not np.isnan(v)]
    v2 = [v for v in values2 if v is not None and not np.isnan(v)]
    if len(v1) < 2 or len(v2) < 2:
        return {"mean1": None, "mean2": None, "diff": None, "p_value": None}
    t, p = ttest_ind(v1, v2, equal_var=False)
    return {
        "mean1": float(np.mean(v1)),
        "mean2": float(np.mean(v2)),
        "diff": float(np.mean(v1) - np.mean(v2)),
        "statistic": float(t),
        "p_value": float(p),
        "n1": int(len(v1)),
        "n2": int(len(v2)),
    }


def format_p(p):
    if p is None:
        return "N/A"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


# =============================================================================
# Cohort builders
# =============================================================================

def create_cohort_table(con, df: pd.DataFrame, table_name: str):
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.register("_temp_df", df)
    con.execute(f"CREATE TEMP TABLE {table_name} AS SELECT * FROM _temp_df")
    con.unregister("_temp_df")
    print(f"[INFO] Created temp table '{table_name}' with {len(df):,} rows")


def build_incident_cohort(con, diagnosis_group: str, diagnoses: dict,
                          require_coverage: bool = True,
                          years=range(2016, 2023)) -> pd.DataFrame:
    """Incident definition:
    - Use claim-level dx match without UNNEST (avoids row explosion)
    - Dedup by (patient, date)
    - New episode if previous dx_date is NULL or gap > 365 days
    - Keep earliest qualifying episode >= MIN_YEAR_FOR_INCIDENT
    - Optional MBSF observability proxy in prior + index year
    """
    codes, match_type = get_diagnosis_codes(diagnoses, diagnosis_group)

    inp_cols = [f"ICD_DGNS_CD{i}" for i in range(1, 11)]
    out_cols = [f"ICD_DGNS_CD{i}" for i in range(1, 11)]
    car_cols = ["PRNCPAL_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 7)]

    inp_pred = build_any_dx_predicate(codes, match_type, inp_cols)
    out_pred = build_any_dx_predicate(codes, match_type, out_cols)
    car_pred = build_any_dx_predicate(codes, match_type, car_cols)

    print(f"\n[INFO] Building incident cohort for '{diagnosis_group}' (NO-UNNEST, year-chunked)...")

    con.execute("DROP TABLE IF EXISTS dx_claims")
    con.execute("CREATE TEMP TABLE dx_claims (DSYSRTKY VARCHAR, dx_date DATE, source VARCHAR)")

    for year in years:
        print(f"       scanning {year}...")

        con.execute(f"""
            INSERT INTO dx_claims
            SELECT DSYSRTKY, CAST(strptime(THRU_DT,'%Y%m%d') AS DATE) AS dx_date, 'inp' AS source
            FROM inp_claimsk_{year}
            WHERE ({inp_pred})
        """)

        con.execute(f"""
            INSERT INTO dx_claims
            SELECT DSYSRTKY, CAST(strptime(THRU_DT,'%Y%m%d') AS DATE) AS dx_date, 'out' AS source
            FROM out_claimsk_{year}
            WHERE ({out_pred})
        """)

        con.execute(f"""
            INSERT INTO dx_claims
            SELECT DSYSRTKY, CAST(strptime(THRU_DT,'%Y%m%d') AS DATE) AS dx_date, 'car' AS source
            FROM car_claimsk_{year}
            WHERE ({car_pred})
        """)

    # Dedup reduces window sort size
    con.execute("DROP TABLE IF EXISTS dx_claims_dedup")
    con.execute("""
        CREATE TEMP TABLE dx_claims_dedup AS
        SELECT DSYSRTKY, dx_date, MIN(source) AS source
        FROM dx_claims
        GROUP BY DSYSRTKY, dx_date
    """)
    con.execute("DROP TABLE dx_claims")

    base = f"""
        WITH ordered AS (
            SELECT
                DSYSRTKY,
                dx_date,
                source,
                LAG(dx_date) OVER (PARTITION BY DSYSRTKY ORDER BY dx_date) AS prev_dx_date
            FROM dx_claims_dedup
        ),
        episodes AS (
            SELECT
                DSYSRTKY,
                dx_date,
                source,
                CAST(EXTRACT(year FROM dx_date) AS INT) AS INDEX_YEAR
            FROM ordered
            WHERE
                CAST(EXTRACT(year FROM dx_date) AS INT) >= {MIN_YEAR_FOR_INCIDENT}
                AND (prev_dx_date IS NULL OR dx_date - prev_dx_date > INTERVAL '{LOOKBACK_DAYS} days')
        ),
        first_ep AS (
            SELECT DSYSRTKY, MIN(dx_date) AS dx_date
            FROM episodes
            GROUP BY DSYSRTKY
        )
        SELECT
            f.DSYSRTKY,
            strftime(f.dx_date, '%Y%m%d') AS INDEX_DATE,
            (SELECT e.source FROM episodes e
             WHERE e.DSYSRTKY=f.DSYSRTKY AND e.dx_date=f.dx_date
             LIMIT 1) AS DATA_SOURCE,
            CAST(EXTRACT(year FROM f.dx_date) AS INT) AS INDEX_YEAR
        FROM first_ep f
    """

    if not require_coverage:
        df = con.execute(base).fetchdf()
        print(f"       Found {len(df):,} incident cases.")
        return df

    # Apply MBSF observability proxy in prior + index year
    query = f"""
        WITH base AS (
            {base}
        )
        SELECT b.*
        FROM base b
        WHERE
            EXISTS (
                SELECT 1 FROM mbsf_all m
                WHERE m.DSYSRTKY=b.DSYSRTKY AND m.RFRNC_YR=CAST(b.INDEX_YEAR AS VARCHAR)
                  AND CAST(COALESCE(NULLIF(m.A_MO_CNT,''),'0') AS INT)=12
                  AND CAST(COALESCE(NULLIF(m.B_MO_CNT,''),'0') AS INT)=12
                  AND CAST(COALESCE(NULLIF(m.HMO_MO,''),'0') AS INT)=0
            )
            AND EXISTS (
                SELECT 1 FROM mbsf_all m
                WHERE m.DSYSRTKY=b.DSYSRTKY AND m.RFRNC_YR=CAST(b.INDEX_YEAR-1 AS VARCHAR)
                  AND CAST(COALESCE(NULLIF(m.A_MO_CNT,''),'0') AS INT)=12
                  AND CAST(COALESCE(NULLIF(m.B_MO_CNT,''),'0') AS INT)=12
                  AND CAST(COALESCE(NULLIF(m.HMO_MO,''),'0') AS INT)=0
            )
    """

    df = con.execute(query).fetchdf()
    print(f"       Found {len(df):,} incident cases.")
    return df


def build_patients_with_dx(con, dx_group: str, diagnoses: dict,
                           years=range(2016, 2023),
                           include_carrier: bool = True) -> None:
    """Creates TEMP TABLE _patients_with_dx (DSYSRTKY) without UNNEST."""
    codes, match_type = get_diagnosis_codes(diagnoses, dx_group)

    inp_cols = [f"ICD_DGNS_CD{i}" for i in range(1, 11)]
    out_cols = [f"ICD_DGNS_CD{i}" for i in range(1, 11)]
    car_cols = ["PRNCPAL_DGNS_CD"] + [f"ICD_DGNS_CD{i}" for i in range(1, 7)]

    inp_pred = build_any_dx_predicate(codes, match_type, inp_cols)
    out_pred = build_any_dx_predicate(codes, match_type, out_cols)
    car_pred = build_any_dx_predicate(codes, match_type, car_cols)

    con.execute("DROP TABLE IF EXISTS _patients_with_dx_raw")
    con.execute("CREATE TEMP TABLE _patients_with_dx_raw (DSYSRTKY VARCHAR)")

    for year in years:
        con.execute(f"""
            INSERT INTO _patients_with_dx_raw
            SELECT DISTINCT DSYSRTKY FROM inp_claimsk_{year} WHERE ({inp_pred})
        """)
        con.execute(f"""
            INSERT INTO _patients_with_dx_raw
            SELECT DISTINCT DSYSRTKY FROM out_claimsk_{year} WHERE ({out_pred})
        """)
        if include_carrier:
            con.execute(f"""
                INSERT INTO _patients_with_dx_raw
                SELECT DISTINCT DSYSRTKY FROM car_claimsk_{year} WHERE ({car_pred})
            """)

    con.execute("DROP TABLE IF EXISTS _patients_with_dx")
    con.execute("""
        CREATE TEMP TABLE _patients_with_dx AS
        SELECT DISTINCT DSYSRTKY FROM _patients_with_dx_raw
    """)
    con.execute("DROP TABLE _patients_with_dx_raw")


def build_nondx_comparison(con, dx_group: str, diagnoses: dict, cohort_table: str,
                           require_coverage: bool = True) -> pd.DataFrame:
    """Comparison cohort = people outside dx_group.

    Matches the index-year distribution of the cohort.
    """
    print("\n[INFO] Building comparison cohort = patients outside diagnosis group...")

    # exclusion set
    build_patients_with_dx(con, dx_group, diagnoses, include_carrier=True)

    year_counts = con.execute(f"""
        SELECT INDEX_YEAR, COUNT(*) AS n
        FROM {cohort_table}
        GROUP BY INDEX_YEAR
        ORDER BY INDEX_YEAR
    """).fetchdf()

    chunks = []
    for _, row in year_counts.iterrows():
        year = int(row["INDEX_YEAR"])
        n = int(row["n"])
        if n <= 0:
            continue

        cov_sql = ""
        if require_coverage:
            cov_sql = """
              AND CAST(COALESCE(NULLIF(m.A_MO_CNT,''),'0') AS INT) = 12
              AND CAST(COALESCE(NULLIF(m.B_MO_CNT,''),'0') AS INT) = 12
              AND CAST(COALESCE(NULLIF(m.HMO_MO,''),'0') AS INT) = 0
            """

        # Use anti-join instead of NOT IN (more stable)
        df_year = con.execute(f"""
            SELECT
                m.DSYSRTKY,
                '{year}0701' AS INDEX_DATE,
                'comparison' AS DATA_SOURCE,
                {year} AS INDEX_YEAR
            FROM mbsf_all m
            LEFT JOIN _patients_with_dx d
              ON m.DSYSRTKY = d.DSYSRTKY
            WHERE m.RFRNC_YR = '{year}'
              AND d.DSYSRTKY IS NULL
              {cov_sql}
            USING SAMPLE {min(max(n*5, 5000), 2_000_000)}
            LIMIT {n}
        """).fetchdf()

        chunks.append(df_year)

    if not chunks:
        return pd.DataFrame(columns=["DSYSRTKY", "INDEX_DATE", "DATA_SOURCE", "INDEX_YEAR"])

    return pd.concat(chunks, ignore_index=True)


# =============================================================================
# Analyses (with comparisons)
# =============================================================================

def analyze_cohort_summary(con, cohort_table: str) -> dict:
    res = {}
    res["summary"] = con.execute(f"""
        SELECT COUNT(*) AS total_patients,
               MIN(INDEX_DATE) AS earliest_index,
               MAX(INDEX_DATE) AS latest_index
        FROM {cohort_table}
    """).fetchdf()

    res["by_source"] = con.execute(f"""
        SELECT DATA_SOURCE,
               COUNT(*) AS patient_count,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM {cohort_table}
        GROUP BY DATA_SOURCE
        ORDER BY patient_count DESC
    """).fetchdf()

    res["by_year"] = con.execute(f"""
        SELECT INDEX_YEAR,
               COUNT(*) AS incident_cases,
               SUM(COUNT(*)) OVER (ORDER BY INDEX_YEAR) AS cumulative
        FROM {cohort_table}
        GROUP BY INDEX_YEAR
        ORDER BY INDEX_YEAR
    """).fetchdf()

    return res


def analyze_demographics(con, cohort_table: str, comparison_table: str, cont_sample_n: int) -> dict:
    res = {}

    def demo(table: str) -> dict:
        age = con.execute(f"""
            SELECT
                CASE
                    WHEN CAST(m.AGE AS INT) < 65 THEN '<65'
                    WHEN CAST(m.AGE AS INT) BETWEEN 65 AND 74 THEN '65-74'
                    WHEN CAST(m.AGE AS INT) BETWEEN 75 AND 84 THEN '75-84'
                    ELSE '85+'
                END AS age_group,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
            FROM {table} c
            JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE m.RFRNC_YR = SUBSTRING(c.INDEX_DATE, 1, 4)
            GROUP BY 1
            ORDER BY CASE age_group WHEN '<65' THEN 1 WHEN '65-74' THEN 2 WHEN '75-84' THEN 3 ELSE 4 END
        """).fetchdf()

        age_stats = con.execute(f"""
            SELECT
                ROUND(AVG(CAST(m.AGE AS INT)), 1) AS mean_age,
                MEDIAN(CAST(m.AGE AS INT)) AS median_age,
                ROUND(STDDEV(CAST(m.AGE AS INT)), 1) AS std_age,
                COUNT(*) AS n
            FROM {table} c
            JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE m.RFRNC_YR = SUBSTRING(c.INDEX_DATE, 1, 4)
        """).fetchdf()

        ages_raw = con.execute(f"""
            SELECT CAST(m.AGE AS INT) AS age
            FROM {table} c
            JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE m.RFRNC_YR = SUBSTRING(c.INDEX_DATE, 1, 4)
            USING SAMPLE {cont_sample_n}
        """).fetchdf()["age"].tolist()

        sex = con.execute(f"""
            SELECT
                CASE WHEN m.SEX='1' THEN 'Male'
                     WHEN m.SEX='2' THEN 'Female'
                     ELSE 'Unknown' END AS sex,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
            FROM {table} c
            JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE m.RFRNC_YR = SUBSTRING(c.INDEX_DATE, 1, 4)
            GROUP BY 1
            ORDER BY patient_count DESC
        """).fetchdf()

        race = con.execute(f"""
            SELECT
                CASE
                    WHEN m.RACE = '0' THEN 'Unknown'
                    WHEN m.RACE = '1' THEN 'White'
                    WHEN m.RACE = '2' THEN 'Black'
                    WHEN m.RACE = '3' THEN 'Other'
                    WHEN m.RACE = '4' THEN 'Asian'
                    WHEN m.RACE = '5' THEN 'Hispanic'
                    WHEN m.RACE = '6' THEN 'Native American'
                    ELSE 'Unknown'
                END AS race,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
            FROM {table} c
            JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
            WHERE m.RFRNC_YR = SUBSTRING(c.INDEX_DATE, 1, 4)
            GROUP BY 1
            ORDER BY patient_count DESC
        """).fetchdf()

        return {"age": age, "age_stats": age_stats, "ages_raw": ages_raw, "sex": sex, "race": race}

    c = demo(cohort_table)
    k = demo(comparison_table)

    res["age"] = c["age"]
    res["age_stats"] = c["age_stats"]
    res["sex"] = c["sex"]
    res["race"] = c["race"]

    res["comparison_age"] = k["age"]
    res["comparison_age_stats"] = k["age_stats"]
    res["comparison_sex"] = k["sex"]
    res["comparison_race"] = k["race"]

    res["age_comparison"] = welch_t_test(c["ages_raw"], k["ages_raw"])

    sex_order = ["Male", "Female"]
    c_counts = [int(c["sex"].loc[c["sex"].sex == s, "patient_count"].sum()) for s in sex_order]
    k_counts = [int(k["sex"].loc[k["sex"].sex == s, "patient_count"].sum()) for s in sex_order]
    res["sex_comparison"] = chi_squared_test(c_counts, k_counts)
    res["female_proportion_comparison"] = proportion_diff_ci(sum(c_counts), c_counts[1], sum(k_counts), k_counts[1])

    race_order = ["White", "Black", "Hispanic", "Asian", "Native American", "Other", "Unknown"]
    c_r = [int(c["race"].loc[c["race"].race == r, "patient_count"].sum()) for r in race_order]
    k_r = [int(k["race"].loc[k["race"].race == r, "patient_count"].sum()) for r in race_order]
    res["race_comparison"] = chi_squared_test(c_r, k_r)

    return res


def analyze_mortality(con, cohort_table: str, comparison_table: str) -> dict:
    res = {}

    def mort(table: str) -> pd.DataFrame:
        return con.execute(f"""
            WITH md AS (
                SELECT
                    c.DSYSRTKY,
                    c.INDEX_DATE,
                    m.DEATH_DT,
                    CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != '' THEN 1 ELSE 0 END AS died,
                    CASE
                        WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                         AND strptime(m.DEATH_DT,'%Y%m%d') <= strptime(c.INDEX_DATE,'%Y%m%d') + INTERVAL '365 days'
                        THEN 1 ELSE 0
                    END AS died_1yr
                FROM {table} c
                JOIN mbsf_all m ON c.DSYSRTKY = m.DSYSRTKY
                WHERE m.RFRNC_YR = '2022'
            )
            SELECT
                COUNT(*) AS total_patients,
                SUM(died) AS total_deaths,
                ROUND(100.0 * SUM(died)/COUNT(*), 2) AS overall_mortality_pct,
                SUM(died_1yr) AS deaths_1yr,
                ROUND(100.0 * SUM(died_1yr)/COUNT(*), 2) AS mortality_1yr_pct
            FROM md
        """).fetchdf()

    c = mort(cohort_table).iloc[0]
    k = mort(comparison_table).iloc[0]

    res["cohort"] = pd.DataFrame([c])
    res["comparison"] = pd.DataFrame([k])

    res["overall_mortality_comparison"] = proportion_diff_ci(int(c.total_patients), int(c.total_deaths),
                                                             int(k.total_patients), int(k.total_deaths))
    res["mortality_1yr_comparison"] = proportion_diff_ci(int(c.total_patients), int(c.deaths_1yr),
                                                         int(k.total_patients), int(k.deaths_1yr))

    return res


def analyze_inpatient_utilization(con, cohort_table: str, comparison_table: str, cont_sample_n: int) -> dict:
    res = {}

    def admission_stats(table: str) -> pd.DataFrame:
        return con.execute(f"""
            WITH admissions AS (
                SELECT
                    c.DSYSRTKY,
                    COUNT(*) AS num_admissions,
                    AVG(date_diff('day', strptime(i.ADMSN_DT,'%Y%m%d'), strptime(i.DSCHRGDT,'%Y%m%d'))) AS avg_los
                FROM {table} c
                JOIN inp_claimsk_all i ON c.DSYSRTKY = i.DSYSRTKY
                WHERE i.THRU_DT >= c.INDEX_DATE
                GROUP BY c.DSYSRTKY
            )
            SELECT
                (SELECT COUNT(*) FROM {table}) AS total_cohort,
                COUNT(*) AS patients_admitted,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {table}), 2) AS admission_rate_pct,
                SUM(num_admissions) AS total_admissions,
                ROUND(AVG(num_admissions), 2) AS mean_admissions_per_patient,
                MEDIAN(num_admissions) AS median_admissions,
                ROUND(AVG(avg_los), 1) AS mean_los_days,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_los) AS median_los_days
            FROM admissions
        """).fetchdf()

    res["admission_stats"] = admission_stats(cohort_table)
    res["comparison_admission_stats"] = admission_stats(comparison_table)

    c = res["admission_stats"].iloc[0]
    k = res["comparison_admission_stats"].iloc[0]
    res["admission_rate_comparison"] = proportion_diff_ci(int(c.total_cohort), int(c.patients_admitted),
                                                          int(k.total_cohort), int(k.patients_admitted))

    c_vals = con.execute(f"""
        SELECT
            COUNT(*) AS num_adm,
            AVG(date_diff('day', strptime(i.ADMSN_DT,'%Y%m%d'), strptime(i.DSCHRGDT,'%Y%m%d'))) AS los
        FROM {cohort_table} c
        JOIN inp_claimsk_all i ON c.DSYSRTKY = i.DSYSRTKY
        WHERE i.THRU_DT >= c.INDEX_DATE
        GROUP BY c.DSYSRTKY
        USING SAMPLE {cont_sample_n}
    """).fetchdf()

    k_vals = con.execute(f"""
        SELECT
            COUNT(*) AS num_adm,
            AVG(date_diff('day', strptime(i.ADMSN_DT,'%Y%m%d'), strptime(i.DSCHRGDT,'%Y%m%d'))) AS los
        FROM {comparison_table} c
        JOIN inp_claimsk_all i ON c.DSYSRTKY = i.DSYSRTKY
        WHERE i.THRU_DT >= c.INDEX_DATE
        GROUP BY c.DSYSRTKY
        USING SAMPLE {cont_sample_n}
    """).fetchdf()

    res["num_admissions_test"] = welch_t_test(c_vals["num_adm"].tolist(), k_vals["num_adm"].tolist())
    res["los_test"] = welch_t_test(c_vals["los"].tolist(), k_vals["los"].tolist())

    return res


def analyze_costs(con, cohort_table: str, comparison_table: str, cont_sample_n: int) -> dict:
    res = {}

    def by_type(table: str) -> pd.DataFrame:
        df = con.execute(f"""
            WITH cohort_size AS (SELECT COUNT(*) AS n FROM {table}),
            inp AS (
                SELECT SUM(CAST(COALESCE(NULLIF(i.PMT_AMT,''),'0') AS DECIMAL(18,2))) AS total
                FROM {table} c JOIN inp_claimsk_all i ON c.DSYSRTKY=i.DSYSRTKY
                WHERE i.THRU_DT >= c.INDEX_DATE
            ),
            out AS (
                SELECT SUM(CAST(COALESCE(NULLIF(o.PMT_AMT,''),'0') AS DECIMAL(18,2))) AS total
                FROM {table} c JOIN out_claimsk_all o ON c.DSYSRTKY=o.DSYSRTKY
                WHERE o.THRU_DT >= c.INDEX_DATE
            ),
            car AS (
                SELECT SUM(CAST(COALESCE(NULLIF(r.PMT_AMT,''),'0') AS DECIMAL(18,2))) AS total
                FROM {table} c JOIN car_claimsk_all r ON c.DSYSRTKY=r.DSYSRTKY
                WHERE r.THRU_DT >= c.INDEX_DATE
            )
            SELECT 'Inpatient' AS cost_type, inp.total AS total_spending, ROUND(inp.total / cohort_size.n, 2) AS per_capita FROM inp, cohort_size
            UNION ALL
            SELECT 'Outpatient', out.total, ROUND(out.total / cohort_size.n, 2) FROM out, cohort_size
            UNION ALL
            SELECT 'Carrier', car.total, ROUND(car.total / cohort_size.n, 2) FROM car, cohort_size
        """).fetchdf()

        total_spend = float(df["total_spending"].sum())
        total_pc = float(df["per_capita"].sum())
        return pd.concat([df, pd.DataFrame([{ "cost_type": "TOTAL", "total_spending": total_spend, "per_capita": total_pc }])], ignore_index=True)

    res["by_type"] = by_type(cohort_table)
    res["comparison_by_type"] = by_type(comparison_table)

    c_costs = con.execute(f"""
        WITH per_patient AS (
            SELECT c.DSYSRTKY,
                   SUM(CAST(COALESCE(NULLIF(i.PMT_AMT,''),'0') AS DOUBLE)) AS inp,
                   0.0 AS outp,
                   0.0 AS car
            FROM {cohort_table} c
            JOIN inp_claimsk_all i ON c.DSYSRTKY=i.DSYSRTKY
            WHERE i.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY

            UNION ALL

            SELECT c.DSYSRTKY,
                   0.0,
                   SUM(CAST(COALESCE(NULLIF(o.PMT_AMT,''),'0') AS DOUBLE)),
                   0.0
            FROM {cohort_table} c
            JOIN out_claimsk_all o ON c.DSYSRTKY=o.DSYSRTKY
            WHERE o.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY

            UNION ALL

            SELECT c.DSYSRTKY,
                   0.0,
                   0.0,
                   SUM(CAST(COALESCE(NULLIF(r.PMT_AMT,''),'0') AS DOUBLE))
            FROM {cohort_table} c
            JOIN car_claimsk_all r ON c.DSYSRTKY=r.DSYSRTKY
            WHERE r.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY
        )
        SELECT DSYSRTKY,
               SUM(inp) AS inpatient,
               SUM(outp) AS outpatient,
               SUM(car) AS carrier,
               SUM(inp+outp+car) AS total
        FROM per_patient
        GROUP BY DSYSRTKY
        USING SAMPLE {cont_sample_n}
    """).fetchdf()

    k_costs = con.execute(f"""
        WITH per_patient AS (
            SELECT c.DSYSRTKY,
                   SUM(CAST(COALESCE(NULLIF(i.PMT_AMT,''),'0') AS DOUBLE)) AS inp,
                   0.0 AS outp,
                   0.0 AS car
            FROM {comparison_table} c
            JOIN inp_claimsk_all i ON c.DSYSRTKY=i.DSYSRTKY
            WHERE i.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY

            UNION ALL

            SELECT c.DSYSRTKY,
                   0.0,
                   SUM(CAST(COALESCE(NULLIF(o.PMT_AMT,''),'0') AS DOUBLE)),
                   0.0
            FROM {comparison_table} c
            JOIN out_claimsk_all o ON c.DSYSRTKY=o.DSYSRTKY
            WHERE o.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY

            UNION ALL

            SELECT c.DSYSRTKY,
                   0.0,
                   0.0,
                   SUM(CAST(COALESCE(NULLIF(r.PMT_AMT,''),'0') AS DOUBLE))
            FROM {comparison_table} c
            JOIN car_claimsk_all r ON c.DSYSRTKY=r.DSYSRTKY
            WHERE r.THRU_DT >= c.INDEX_DATE
            GROUP BY c.DSYSRTKY
        )
        SELECT DSYSRTKY,
               SUM(inp) AS inpatient,
               SUM(outp) AS outpatient,
               SUM(car) AS carrier,
               SUM(inp+outp+car) AS total
        FROM per_patient
        GROUP BY DSYSRTKY
        USING SAMPLE {cont_sample_n}
    """).fetchdf()

    res["total_cost_test"] = welch_t_test(c_costs["total"].tolist(), k_costs["total"].tolist())
    res["inpatient_cost_test"] = welch_t_test(c_costs["inpatient"].tolist(), k_costs["inpatient"].tolist())
    res["outpatient_cost_test"] = welch_t_test(c_costs["outpatient"].tolist(), k_costs["outpatient"].tolist())
    res["carrier_cost_test"] = welch_t_test(c_costs["carrier"].tolist(), k_costs["carrier"].tolist())

    return res


def analyze_comorbidities(con, cohort_table: str, comparison_table: str, diagnoses: dict) -> dict:
    """Comorbidity prevalence restricted to cohort+comparison patients.

    This keeps the UNNEST manageable.
    """
    res = {}

    con.execute("DROP TABLE IF EXISTS _pt")
    con.execute(f"""
        CREATE TEMP TABLE _pt AS
        SELECT DSYSRTKY FROM {cohort_table}
        UNION
        SELECT DSYSRTKY FROM {comparison_table}
    """)

    # Build dx long table ONLY for _pt
    con.execute("DROP TABLE IF EXISTS _all_dx")
    con.execute("""
        CREATE TEMP TABLE _all_dx AS
        SELECT DSYSRTKY, dx
        FROM (
            SELECT i.DSYSRTKY,
                   UNNEST([
                       ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4, ICD_DGNS_CD5,
                       ICD_DGNS_CD6, ICD_DGNS_CD7, ICD_DGNS_CD8, ICD_DGNS_CD9, ICD_DGNS_CD10
                   ]) AS dx
            FROM inp_claimsk_all i
            JOIN _pt p ON i.DSYSRTKY = p.DSYSRTKY

            UNION ALL

            SELECT o.DSYSRTKY,
                   UNNEST([
                       ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4, ICD_DGNS_CD5,
                       ICD_DGNS_CD6, ICD_DGNS_CD7, ICD_DGNS_CD8, ICD_DGNS_CD9, ICD_DGNS_CD10
                   ]) AS dx
            FROM out_claimsk_all o
            JOIN _pt p ON o.DSYSRTKY = p.DSYSRTKY

            UNION ALL

            SELECT r.DSYSRTKY,
                   UNNEST([
                       PRNCPAL_DGNS_CD, ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3, ICD_DGNS_CD4, ICD_DGNS_CD5, ICD_DGNS_CD6
                   ]) AS dx
            FROM car_claimsk_all r
            JOIN _pt p ON r.DSYSRTKY = p.DSYSRTKY
        )
        WHERE dx IS NOT NULL AND dx != ''
    """)

    cohort_n = con.execute(f"SELECT COUNT(*) FROM {cohort_table}").fetchone()[0]
    comp_n = con.execute(f"SELECT COUNT(*) FROM {comparison_table}").fetchone()[0]

    rows = []
    for group_name in [k for k in diagnoses.keys() if k != 'notes']:
        try:
            codes, match_type = get_diagnosis_codes(diagnoses, group_name)
            where_dx = build_dx_match_expr(codes, match_type, "dx")

            c_cnt = con.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT c.DSYSRTKY
                    FROM {cohort_table} c
                    JOIN _all_dx d ON c.DSYSRTKY = d.DSYSRTKY
                    WHERE ({where_dx})
                )
            """).fetchone()[0]

            k_cnt = con.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT c.DSYSRTKY
                    FROM {comparison_table} c
                    JOIN _all_dx d ON c.DSYSRTKY = d.DSYSRTKY
                    WHERE ({where_dx})
                )
            """).fetchone()[0]

            pdiff = proportion_diff_ci(cohort_n, c_cnt, comp_n, k_cnt)

            rows.append({
                "condition": group_name,
                "cohort_patients": int(c_cnt),
                "cohort_pct": round(100.0 * c_cnt / cohort_n, 2) if cohort_n else 0.0,
                "comparison_patients": int(k_cnt),
                "comparison_pct": round(100.0 * k_cnt / comp_n, 2) if comp_n else 0.0,
                "diff_pct_points": round(100.0 * (pdiff["diff"] or 0.0), 2),
                "p_value": pdiff.get("p_value"),
            })
        except Exception as e:
            print(f"       [WARN] comorbidity '{group_name}' skipped: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cohort_patients"], ascending=False)
    res["prevalence"] = df
    return res


# =============================================================================
# Output
# =============================================================================

def generate_text_report(diagnosis_group: str, description: str, comparison_label: str,
                         all_results: dict, output_path: Path):

    lines = []
    lines.append("=" * 90)
    lines.append("CMS MEDICARE CLAIMS COHORT ANALYSIS REPORT (OPTIMIZED + FIXED)")
    lines.append("=" * 90)
    lines.append(f"Diagnosis Group: {diagnosis_group}")
    lines.append(f"Description: {description}")
    lines.append(f"Comparison: {comparison_label}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Incident rule: clean {LOOKBACK_DAYS}d + (optional) full-year FFS A/B in index+prior year")
    lines.append("=" * 90)

    s = all_results["cohort_summary"]["summary"].iloc[0]
    lines.append("\nSECTION 1: COHORT SUMMARY")
    lines.append("-" * 90)
    lines.append(f"Total Incident Cases: {int(s.total_patients):,}")
    lines.append(f"Index Date Range: {s.earliest_index} to {s.latest_index}")

    d = all_results["demographics"]
    lines.append("\nSECTION 2: DEMOGRAPHICS (Cohort vs Comparison)")
    lines.append("-" * 90)
    lines.append(
        f"Age mean (cohort vs comp): {d['age_stats'].iloc[0].mean_age} vs {d['comparison_age_stats'].iloc[0].mean_age} | p={format_p(d['age_comparison'].get('p_value'))}"
    )
    lines.append(
        f"Sex chi2 p={format_p(d['sex_comparison'].get('p_value'))} | Female proportion diff p={format_p(d['female_proportion_comparison'].get('p_value'))}"
    )
    lines.append(f"Race chi2 p={format_p(d['race_comparison'].get('p_value'))}")

    m = all_results["mortality"]
    c_m = m["cohort"].iloc[0]
    k_m = m["comparison"].iloc[0]
    lines.append("\nSECTION 3: MORTALITY (Cohort vs Comparison)")
    lines.append("-" * 90)
    lines.append(
        f"Overall mortality: {c_m.overall_mortality_pct:.2f}% vs {k_m.overall_mortality_pct:.2f}% | p={format_p(m['overall_mortality_comparison'].get('p_value'))}"
    )
    lines.append(
        f"1-year mortality: {c_m.mortality_1yr_pct:.2f}% vs {k_m.mortality_1yr_pct:.2f}% | p={format_p(m['mortality_1yr_comparison'].get('p_value'))}"
    )

    u = all_results["inpatient_utilization"]
    lines.append("\nSECTION 4: INPATIENT UTILIZATION (Cohort vs Comparison)")
    lines.append("-" * 90)
    lines.append(f"Admission rate p={format_p(u['admission_rate_comparison'].get('p_value'))}")
    lines.append(f"# admissions per patient p={format_p(u['num_admissions_test'].get('p_value'))}")
    lines.append(f"Avg LOS per patient p={format_p(u['los_test'].get('p_value'))}")

    cst = all_results["costs"]
    lines.append("\nSECTION 5: COSTS (Cohort vs Comparison)")
    lines.append("-" * 90)
    lines.append(f"Total cost per patient (sampled) p={format_p(cst['total_cost_test'].get('p_value'))}")
    lines.append(
        f"Inpatient cost p={format_p(cst['inpatient_cost_test'].get('p_value'))} | "
        f"Outpatient cost p={format_p(cst['outpatient_cost_test'].get('p_value'))} | "
        f"Carrier cost p={format_p(cst['carrier_cost_test'].get('p_value'))}"
    )

    com = all_results["comorbidities"]["prevalence"]
    lines.append("\nSECTION 6: COMORBIDITIES (Top 30 by cohort count)")
    lines.append("-" * 90)
    if com is None or com.empty:
        lines.append("No comorbidity results.")
    else:
        top = com.head(30)
        for _, r in top.iterrows():
            lines.append(
                f"{r['condition'][:34]:34} {int(r['cohort_patients']):>10,} ({r['cohort_pct']:>5.1f}%) "
                f"vs {int(r['comparison_patients']):>10,} ({r['comparison_pct']:>5.1f}%)  p={format_p(r['p_value'])}"
            )

    lines.append("\n" + "=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    output_path.write_text("\n".join(lines))
    print(f"[INFO] Text report saved to: {output_path}")


def export_csvs(all_results: dict, output_dir: Path):
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    n = 0
    for section, section_results in all_results.items():
        for key, obj in section_results.items():
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(csv_dir / f"{section}_{key}.csv", index=False)
                n += 1
            elif isinstance(obj, dict):
                pd.DataFrame([obj]).to_csv(csv_dir / f"{section}_{key}.csv", index=False)
                n += 1
    print(f"[INFO] Exported {n} CSV files to: {csv_dir}")


def save_metadata(args, output_dir: Path, cohort_size: int, comparison_size: int, elapsed: float):
    meta = {
        "diagnosis_group": args.diagnosis,
        "comparison_group": args.compare,
        "database": args.database,
        "diagnoses_file": args.diagnoses_file,
        "lookback_days": LOOKBACK_DAYS,
        "min_year_for_incident": MIN_YEAR_FOR_INCIDENT,
        "require_full_ffs_ab_coverage": (not args.no_coverage_filter),
        "cohort_size": cohort_size,
        "comparison_size": comparison_size,
        "cont_sample_n": args.cont_sample,
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "memory_limit": args.memory,
        "threads": args.threads,
        "temp_directory": args.temp_dir,
        "max_temp_directory_size": args.max_temp,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# =============================================================================
# Main
# =============================================================================

def main():
    load_dotenv()
    args = parse_arguments()

    start = datetime.now()

    try:
        diagnoses = load_diagnoses(args.diagnoses_file)
    except FileNotFoundError:
        print(f"[ERROR] Diagnoses file not found: {args.diagnoses_file}")
        sys.exit(1)

    if args.diagnosis not in diagnoses:
        available = [k for k in diagnoses.keys() if k != 'notes']
        print(f"[ERROR] Diagnosis group '{args.diagnosis}' not found. Available: {available}")
        sys.exit(1)

    description = (diagnoses.get(args.diagnosis) or {}).get('description', 'No description')

    # temp dir default: create a dedicated folder if they passed a volume path
    if args.temp_dir is None:
        # If you have a big SSD mounted, pass --temp-dir explicitly.
        # Otherwise DuckDB uses its default.
        temp_dir = None
    else:
        temp_dir = args.temp_dir

    require_cov = (not args.no_coverage_filter)

    # RO connection for cohort build; keep threads low to reduce peak memory
    con = connect_database(args.database, args.memory, args.threads, read_only=True,
                           temp_dir=temp_dir, max_temp=args.max_temp, low_thread_mode=True)
    if con is None:
        sys.exit(1)

    if not print_data_status(check_data_availability(con)):
        con.close()
        sys.exit(1)

    print("\n[INFO] Building primary cohort...")
    cohort_df = build_incident_cohort(con, args.diagnosis, diagnoses, require_coverage=require_cov)
    if cohort_df.empty:
        print("[ERROR] No incident cases found.")
        con.close()
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{args.diagnosis}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    con.close()

    # RW connection for temp tables + analysis; can use more threads
    con = connect_database(args.database, args.memory, args.threads, read_only=False,
                           temp_dir=temp_dir, max_temp=args.max_temp, low_thread_mode=False)
    if con is None:
        sys.exit(1)

    create_cohort_table(con, cohort_df, "cohort")

    comparison_label = "Non-diagnosis Medicare population"
    if args.compare:
        if args.compare not in diagnoses:
            available = [k for k in diagnoses.keys() if k != 'notes']
            print(f"[ERROR] Comparison group '{args.compare}' not found. Available: {available}")
            con.close()
            sys.exit(1)
        print("\n[INFO] Building comparison cohort as incident cohort for comparison diagnosis...")

        # build comparison with low-thread mode by temporarily reducing threads
        con.execute("PRAGMA threads=2;")
        comp_df = build_incident_cohort(con, args.compare, diagnoses, require_coverage=require_cov)
        con.execute(f"PRAGMA threads={args.threads};")

        comparison_label = f"Incident cohort: {args.compare}"
    else:
        comp_df = build_nondx_comparison(con, args.diagnosis, diagnoses, "cohort", require_coverage=require_cov)

    if comp_df.empty:
        print("[ERROR] Comparison cohort is empty.")
        con.close()
        sys.exit(1)

    create_cohort_table(con, comp_df, "comparison")

    all_results: dict = {}
    all_results["cohort_summary"] = analyze_cohort_summary(con, "cohort")
    all_results["demographics"] = analyze_demographics(con, "cohort", "comparison", args.cont_sample)
    all_results["mortality"] = analyze_mortality(con, "cohort", "comparison")
    all_results["inpatient_utilization"] = analyze_inpatient_utilization(con, "cohort", "comparison", args.cont_sample)
    all_results["costs"] = analyze_costs(con, "cohort", "comparison", args.cont_sample)
    all_results["comorbidities"] = analyze_comorbidities(con, "cohort", "comparison", diagnoses)

    generate_text_report(args.diagnosis, description, comparison_label, all_results, output_dir / "report.txt")
    export_csvs(all_results, output_dir)

    elapsed = (datetime.now() - start).total_seconds()
    save_metadata(args, output_dir, len(cohort_df), len(comp_df), elapsed)

    con.close()

    print("\n" + "=" * 70)
    print(f"Analysis complete! Total time: {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
