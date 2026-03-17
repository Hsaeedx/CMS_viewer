-- Step 3a: HNC diagnosis claims within 24 months of death
-- Adapted from: opscc/queries/hnc_dx_raw.sql
-- Performance: single scan of inp_claimsk_all (UNNEST principal + admitting together)
-- Diagnosis codes at CLAIM level (not line level)
-- Output: io_hnc_dx_raw

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_hnc_dx_raw;

CREATE TABLE io_hnc_dx_raw AS

-- Inpatient: single scan, check principal + admitting via UNNEST
SELECT DISTINCT
    i.DSYSRTKY,
    TRY_STRPTIME(COALESCE(NULLIF(i.ADMSN_DT,''), i.THRU_DT), '%Y%m%d') AS dx_date,
    'inp' AS source,
    SUBSTRING(code, 1, 4) AS dx_prefix
FROM inp_claimsk_all i
JOIN io_decedents d ON i.DSYSRTKY = d.DSYSRTKY,
UNNEST([i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD]) AS t(code)
WHERE code IS NOT NULL AND code <> ''
  AND (
    SUBSTRING(code, 1, 3) BETWEEN 'C00' AND 'C14'
    OR SUBSTRING(code, 1, 3) IN ('C30','C31','C32','C43','C44','C49','C76','C77')
  )
  AND TRY_STRPTIME(COALESCE(NULLIF(i.ADMSN_DT,''), i.THRU_DT), '%Y%m%d')
      BETWEEN d.death_dt_parsed - INTERVAL 24 MONTH AND d.death_dt_parsed

UNION ALL

-- Outpatient: principal diagnosis
SELECT
    oc.DSYSRTKY,
    TRY_STRPTIME(oc.THRU_DT, '%Y%m%d') AS dx_date,
    'out' AS source,
    SUBSTRING(oc.PRNCPAL_DGNS_CD, 1, 4) AS dx_prefix
FROM out_claimsk_all oc
JOIN io_decedents d ON oc.DSYSRTKY = d.DSYSRTKY
WHERE (
    SUBSTRING(oc.PRNCPAL_DGNS_CD, 1, 3) BETWEEN 'C00' AND 'C14'
    OR SUBSTRING(oc.PRNCPAL_DGNS_CD, 1, 3) IN ('C30','C31','C32','C43','C44','C49','C76','C77')
)
AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
    BETWEEN d.death_dt_parsed - INTERVAL 24 MONTH AND d.death_dt_parsed

UNION ALL

-- Carrier: principal diagnosis
SELECT
    c.DSYSRTKY,
    TRY_STRPTIME(c.THRU_DT, '%Y%m%d') AS dx_date,
    'car' AS source,
    SUBSTRING(c.PRNCPAL_DGNS_CD, 1, 4) AS dx_prefix
FROM car_claimsk_all c
JOIN io_decedents d ON c.DSYSRTKY = d.DSYSRTKY
WHERE (
    SUBSTRING(c.PRNCPAL_DGNS_CD, 1, 3) BETWEEN 'C00' AND 'C14'
    OR SUBSTRING(c.PRNCPAL_DGNS_CD, 1, 3) IN ('C30','C31','C32','C43','C44','C49','C76','C77')
)
AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
    BETWEEN d.death_dt_parsed - INTERVAL 24 MONTH AND d.death_dt_parsed;
