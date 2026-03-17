-- Step 5: IO claims (pembrolizumab J9271, nivolumab J9299)
-- HCPCS codes are at LINE level only — use car_linek_all and out_revenuek_all
-- Performance: join against io_subsite (41K HNC patients) instead of io_decedents (14.5M)
-- Output: io_claims_raw

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_claims_raw;

CREATE TABLE io_claims_raw AS

WITH hnc_decedents AS (
    -- Pre-filter to HNC-eligible patients only (~41K vs 14.5M)
    SELECT d.DSYSRTKY, d.death_dt_parsed
    FROM io_decedents d
    JOIN io_subsite s ON d.DSYSRTKY = s.DSYSRTKY
)

-- Carrier line: HCPCS at line level
SELECT
    l.DSYSRTKY,
    TRY_STRPTIME(l.THRU_DT, '%Y%m%d') AS io_date,
    l.HCPCS_CD
FROM car_linek_all l
JOIN hnc_decedents h ON l.DSYSRTKY = h.DSYSRTKY
WHERE l.HCPCS_CD IN ('J9271','J9299')
  AND TRY_STRPTIME(l.THRU_DT, '%Y%m%d')
      BETWEEN h.death_dt_parsed - INTERVAL 24 MONTH AND h.death_dt_parsed

UNION ALL

-- Outpatient revenue: HCPCS at revenue line level
SELECT
    r.DSYSRTKY,
    TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d') AS io_date,
    r.HCPCS_CD
FROM out_revenuek_all r
JOIN hnc_decedents h ON r.DSYSRTKY = h.DSYSRTKY
WHERE r.HCPCS_CD IN ('J9271','J9299')
  AND TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d')
      BETWEEN h.death_dt_parsed - INTERVAL 24 MONTH AND h.death_dt_parsed;
