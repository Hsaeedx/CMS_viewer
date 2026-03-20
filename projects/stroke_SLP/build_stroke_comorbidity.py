"""
build_stroke_comorbidity.py

Generates and runs the stroke_comorbidity table using the existing
elixhauser_mapping and comorbid_weights from opscc/comorbidity.py.

No UNNEST -- keeps claims in wide format (1 row per claim) so there
is no 27x data explosion. Three single-pass scans with inline GROUP BY:
  inp_flags  -- inpatient  (PRNCPAL + ADMTG + ICD1-25 = 27 cols)
  car_flags  -- carrier    (LINE_ICD_DGNS_CD = 1 col)
  out_flags  -- outpatient (PRNCPAL + ICD1-10 = 11 cols)
Then final join of three small per-patient tables.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_opscc_dir = Path(os.getenv("project_paths", ".")) / "opscc"
sys.path.insert(0, str(_opscc_dir))

import duckdb
import time
from collections import defaultdict
from comorbidity import elixhauser_mapping, comorbid_weights

DB_PATH  = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
CMS_DIR  = Path(os.getenv("CMS_directory", ""))

# ── Column lists ──────────────────────────────────────────────────────────────

INP_COLS = (
    ['i.PRNCPAL_DGNS_CD', 'i.ADMTG_DGNS_CD'] +
    [f'i.ICD_DGNS_CD{n}' for n in range(1, 26)]
)  # 27 columns

OUT_COLS = (
    ['oc.PRNCPAL_DGNS_CD'] +
    [f'oc.ICD_DGNS_CD{n}' for n in range(1, 11)]
)  # 11 columns

CAR_COLS = ['cl.LINE_ICD_DGNS_CD']  # 1 column

# ── Stroke-specific extra flags (prefix -> flag_name) ─────────────────────────

STROKE_EXTRA = {
    'prior_stroke':['I60', 'I61', 'I63', 'I64', 'I69'],
    'afib':        ['I48'],
    'prior_tia':   ['G45'],
    'hypertension':['I10', 'I11', 'I12', 'I13', 'I15'],
    'dyslipid':    ['E78'],
    'smoking':     ['F17', 'Z87891', 'Z720'],
    'dementia':    ['F01', 'F02', 'F03', 'G30'],
}

# ── Build reverse mapping: flag -> [prefixes] ─────────────────────────────────

def reverse_mapping(mapping):
    rev = defaultdict(list)
    for prefix, flag in mapping.items():
        rev[flag].append(prefix)
    return rev

elix_flags = reverse_mapping(elixhauser_mapping)  # flag -> [prefixes]
all_flags  = {**elix_flags, **STROKE_EXTRA}
flag_names = list(elix_flags.keys()) + list(STROKE_EXTRA.keys())

# ── SQL generators ────────────────────────────────────────────────────────────

def like_expr(col, prefixes):
    """e.g.  col LIKE 'I50%' OR col LIKE 'I099%' ..."""
    return ' OR '.join(f"{col} LIKE '{p}%'" for p in prefixes)

def case_when(cols, prefixes, alias):
    """MAX(CASE WHEN col1 LIKE ... OR col2 LIKE ... THEN 1 ELSE 0 END) AS alias"""
    conditions = ' OR '.join(like_expr(col, prefixes) for col in cols)
    return f"        MAX(CASE WHEN {conditions} THEN 1 ELSE 0 END) AS {alias}"

def flag_selects(cols, flag_names, all_flags):
    return ',\n'.join(case_when(cols, all_flags[f], f) for f in flag_names)

def greatest_coalesce(flag):
    return f"    GREATEST(COALESCE(i.{flag},0), COALESCE(car.{flag},0), COALESCE(o.{flag},0)) AS {flag}"

def walraven_term(flag):
    w = comorbid_weights.get(flag, 0)
    if w == 0:
        return None
    sign = '+' if w > 0 else ''
    return f"        COALESCE(GREATEST(COALESCE(i.{flag},0),COALESCE(car.{flag},0),COALESCE(o.{flag},0)),0) * {sign}{w}"

# ── Build full SQL ────────────────────────────────────────────────────────────

def build_sql():
    date_filter_inp = (
        "TRY_STRPTIME(i.THRU_DT, '%Y%m%d') "
        "BETWEEN c.index_adm_date - INTERVAL 12 MONTH "
        "AND c.index_adm_date - INTERVAL 1 DAY"
    )
    date_filter_car = (
        "TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') "
        "BETWEEN c.index_adm_date - INTERVAL 12 MONTH "
        "AND c.index_adm_date - INTERVAL 1 DAY"
    )
    date_filter_out = (
        "TRY_STRPTIME(oc.THRU_DT, '%Y%m%d') "
        "BETWEEN c.index_adm_date - INTERVAL 12 MONTH "
        "AND c.index_adm_date - INTERVAL 1 DAY"
    )

    inp_selects = flag_selects(INP_COLS, flag_names, all_flags)
    car_selects = flag_selects(CAR_COLS, flag_names, all_flags)
    out_selects = flag_selects(OUT_COLS, flag_names, all_flags)

    final_cols = ',\n'.join(greatest_coalesce(f) for f in flag_names)

    wterms = [t for t in (walraven_term(f) for f in elix_flags) if t]
    walraven = ' +\n'.join(wterms)

    sql = f"""
DROP TABLE IF EXISTS stroke_comorbidity;

CREATE TABLE stroke_comorbidity AS
WITH

inp_flags AS (
    SELECT
        c.DSYSRTKY,
{inp_selects}
    FROM inp_claimsk_all i
    JOIN stroke_cohort c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE {date_filter_inp}
    GROUP BY c.DSYSRTKY
),

car_flags AS (
    SELECT
        c.DSYSRTKY,
{car_selects}
    FROM car_linek_all cl
    JOIN stroke_cohort c ON cl.DSYSRTKY = c.DSYSRTKY
    WHERE {date_filter_car}
      AND cl.LINE_ICD_DGNS_CD IS NOT NULL
    GROUP BY c.DSYSRTKY
),

out_flags AS (
    SELECT
        c.DSYSRTKY,
{out_selects}
    FROM out_claimsk_all oc
    JOIN stroke_cohort c ON oc.DSYSRTKY = c.DSYSRTKY
    WHERE {date_filter_out}
    GROUP BY c.DSYSRTKY
)

SELECT
    c.DSYSRTKY,
{final_cols},
    (
{walraven}
    ) AS van_walraven_score

FROM stroke_cohort c
LEFT JOIN inp_flags i   ON i.DSYSRTKY   = c.DSYSRTKY
LEFT JOIN car_flags car ON car.DSYSRTKY = c.DSYSRTKY
LEFT JOIN out_flags o   ON o.DSYSRTKY   = c.DSYSRTKY;
"""
    return sql


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    sql = build_sql()

    print('Connecting...')
    con = duckdb.connect(str(DB_PATH), read_only=False)
    temp_dir = (CMS_DIR / "duckdb_temp").as_posix()
    con.execute(
        "SET memory_limit='24GB'; "
        "SET threads=12; "
        f"SET temp_directory='{temp_dir}';"
    )

    print('Building stroke_comorbidity (no UNNEST, 3 parallel scans)...')
    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s')

    n = con.execute('SELECT COUNT(*) FROM stroke_comorbidity').fetchone()[0]
    print(f'stroke_comorbidity: {n:,} rows')

    sample = con.execute("""
        SELECT
            SUM(chf) AS n_chf,
            SUM(carit) AS n_carit,
            SUM(afib) AS n_afib,
            SUM(hypertension) AS n_htn,
            ROUND(AVG(van_walraven_score),2) AS mean_vw
        FROM stroke_comorbidity
    """).df()
    print(sample.to_string(index=False))

    con.close()


if __name__ == '__main__':
    main()
