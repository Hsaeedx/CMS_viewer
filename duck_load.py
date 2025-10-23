#!/usr/bin/env python3
"""
cms_duckdb_pipeline_2019_fixed.py
- Reads Parquet files (MBSF, INP, INP-LINK, OUT)
- Properly joins INP header -> INP link to get BENE_ID
- Finds HNC beneficiaries (ICD-10 C00–C14, C30–C32)
- Joins to MBSF for demographics/death date
- Saves a tidy cohort file (CSV + Parquet)

NOTE: Adjust column names if your drop uses slightly different headers.
Run these to inspect columns if anything errors:
  PRAGMA table_info('inp2019');
  PRAGMA table_info('inplink2019');
  PRAGMA table_info('out2019');
  PRAGMA table_info('mbsf2019');
"""

from pathlib import Path
import duckdb
import sys

# ----------------------------
# 1) CONFIG — EDIT THESE PATHS
# ----------------------------
YEAR = 2019  # which year to process
DATA_DIR = Path(f"/Volumes/EPIC SSDDDD/CMS/{YEAR}")  # folder with your .parquet files
OUT_DIR  = DATA_DIR                                   # where to write results
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parquet file names expected
MBSF   = DATA_DIR / "mbsf_2019.parquet"
INP    = DATA_DIR / "inp_2019.parquet"          # INP header
INP_LNK= DATA_DIR / "inplink_2019.parquet"      # INP link (BENE_ID mapping)
OUTPT  = DATA_DIR / "out_2019.parquet"          # OUT header

# ----------------------------
# 2) CONNECT + REGISTER VIEWS
# ----------------------------
con = duckdb.connect("cms.duckdb")

def register_view(name: str, parquet_path: Path):
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found", file=sys.stderr)
        sys.exit(1)
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{parquet_path.as_posix()}');")

register_view("mbsf2019",   MBSF)
register_view("inp2019",    INP)
register_view("inplink2019",INP_LNK)
register_view("out2019",    OUTPT)

# ----------------------------
# 3) BUILD CLAIMS WITH BENE_ID
#    In many 2019 drops:
#    - INP header has claim keys like CLAIMNO and/or DSYSRTKY but NOT BENE_ID
#    - INP link has BENE_ID + the same claim key (CLAIMNO or DSYSRTKY)
#    We join USING(CLAIMNO) when present; otherwise fall back to DSYSRTKY.
# ----------------------------

# Detect which common key to use between INP and INP_LINK
common_keys = [r[0] for r in con.execute(
    """
    SELECT name FROM pragma_table_info('inp2019')
    INTERSECT
    SELECT name FROM pragma_table_info('inplink2019');
    """
).fetchall()]

if "CLAIMNO" in common_keys:
    join_key = "CLAIMNO"
elif "DSYSRTKY" in common_keys:
    join_key = "DSYSRTKY"
else:
    raise SystemExit(f"No common join key found between INP and INP_LINK. Common columns: {common_keys}")

con.execute(f"""
CREATE OR REPLACE TEMP VIEW inp2019_join AS
SELECT
  l.BENE_ID,
  h.{join_key}              AS CLM_ID,
  h.PRNCPAL_DGNS_CD         AS dx,
  COALESCE(h.CLM_FROM_DT, h.ADMIT_DT, h.ADMSN_DT) AS clm_from_dt,
  COALESCE(h.CLM_THRU_DT, h.DSCHRGDT)            AS clm_thru_dt
FROM inp2019 h
JOIN inplink2019 l USING ({join_key});
""")

# OUTPATIENT usually already contains BENE_ID; keep columns aligned with INP join
# If your OUT header lacks BENE_ID, you'd need an out-link file and to join the same way.
con.execute(
    """
CREATE OR REPLACE TEMP VIEW out2019_aligned AS
SELECT
  BENE_ID,
  COALESCE(CLM_ID, CLAIMNO) AS CLM_ID,
  PRNCPAL_DGNS_CD           AS dx,
  COALESCE(CLM_FROM_DT, FROM_DT) AS clm_from_dt,
  COALESCE(CLM_THRU_DT,  THRU_DT) AS clm_thru_dt
FROM out2019;
"""
)

# ----------------------------
# 4) HNC ICD-10 FILTER (C00–C14, C30–C32)
# ----------------------------
con.execute(
    """
CREATE OR REPLACE TEMP VIEW hnc_claims_2019 AS
SELECT BENE_ID, CLM_ID, dx, clm_from_dt FROM inp2019_join
UNION ALL
SELECT BENE_ID, CLM_ID, dx, clm_from_dt FROM out2019_aligned;
"""
)

con.execute(
    """
CREATE OR REPLACE TEMP VIEW hnc_ids_2019 AS
SELECT DISTINCT BENE_ID
FROM hnc_claims_2019
WHERE substr(dx,1,3) BETWEEN 'C00' AND 'C14'
   OR substr(dx,1,3) BETWEEN 'C30' AND 'C32';
"""
)

n_claims = con.execute("SELECT COUNT(*) FROM hnc_claims_2019").fetchone()[0]
n_hnc_ids = con.execute("SELECT COUNT(DISTINCT BENE_ID) FROM hnc_ids_2019").fetchone()[0]
print(f"[INFO] 2019 total INP+OUT rows scanned: {n_claims:,}")
print(f"[INFO] 2019 unique HNC beneficiaries:  {n_hnc_ids:,}")

# ----------------------------
# 5) JOIN TO MBSF FOR DEMOGRAPHICS / DEATH
#    Adjust column names below to match your MBSF drop (STATE_CD vs STATE_CODE, etc.)
# ----------------------------
cohort_df = con.execute(
    """
SELECT
  m.BENE_ID,
  m.BENE_BIRTH_DT,
  COALESCE(m.BENE_SEX_IDENT_CD, m.GNDR_CD) AS SEX,
  COALESCE(m.RACE_CD, m.RACE)              AS RACE,
  COALESCE(m.STATE_CODE, m.STATE_CD)       AS STATE,
  m.DEATH_DT
FROM mbsf2019 m
JOIN hnc_ids_2019 h USING (BENE_ID);
"""
).df()

print(f"[INFO] Cohort rows (unique bene-level): {len(cohort_df):,}")

# ----------------------------
# 6) SAVE COHORT
# ----------------------------
cohort_csv = OUT_DIR / "hnc_beneficiaries_2019.csv"
cohort_parquet = OUT_DIR / "hnc_beneficiaries_2019.parquet"
cohort_df.to_csv(cohort_csv, index=False)
con.execute(f"""
COPY (SELECT * FROM read_csv_auto('{cohort_csv.as_posix()}', HEADER=TRUE))
TO '{cohort_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION 'snappy');
""")
print(f"[OK] Wrote: {cohort_csv}")
print(f"[OK] Wrote: {cohort_parquet}")

# ----------------------------
# 7) OPTIONAL: AUTONOMIC DYSFUNCTION FLAG
# ----------------------------
con.execute(
    """
CREATE OR REPLACE TEMP VIEW auto_dys_2019 AS
SELECT DISTINCT BENE_ID
FROM hnc_claims_2019
WHERE dx IN ('I951','I95.1','I959','I95.9')
   OR substr(dx,1,3) IN ('G90');
"""
)

auton_df = con.execute(
    """
SELECT
  c.BENE_ID,
  CASE WHEN a.BENE_ID IS NOT NULL THEN 1 ELSE 0 END AS AUTONOMIC_FLAG
FROM (SELECT DISTINCT BENE_ID FROM hnc_ids_2019) c
LEFT JOIN auto_dys_2019 a USING (BENE_ID);
"""
).df()

auton_csv = OUT_DIR / "hnc_autonomic_flag_2019.csv"
auton_df.to_csv(auton_csv, index=False)
print(f"[OK] Wrote: {auton_csv}")

print("\n[Done] Next steps:")
print(" - If OUT header lacks BENE_ID, create an OUT link parquet and join like INP.")
print(" - Add 2020/2021 by registering mbsf/inp/inplink/out views per year and UNION ALL.")
print(" - Cast dates in SQL as needed: try_strptime(DSCHRGDT, '%Y%m%d')::DATE.")
