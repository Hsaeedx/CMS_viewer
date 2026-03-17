-- Step 3b: Confirm HNC diagnosis with ≥2 claims on separate dates
-- Adapted from: opscc/queries/hnc_confirmed.sql
-- Change: require COUNT(DISTINCT dx_date) >= 2 on ANY separate dates
--   (no ≥30 day spacing requirement, no inpatient shortcut)
-- Source: io_hnc_dx_raw
-- Output: io_hnc_confirmed

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_hnc_confirmed;

CREATE TABLE io_hnc_confirmed AS

SELECT
    DSYSRTKY,
    MIN(dx_date) AS first_hnc_dx_date,
    COUNT(*) AS total_claims,
    COUNT(DISTINCT dx_date) AS distinct_dx_dates

FROM io_hnc_dx_raw

GROUP BY DSYSRTKY
HAVING COUNT(DISTINCT dx_date) >= 2;
