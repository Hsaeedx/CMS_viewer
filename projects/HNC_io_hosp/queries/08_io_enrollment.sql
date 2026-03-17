-- Steps 9-11: Enrollment eligibility
--   9. Exclude ESRD (OREC = '2')
--  10. Require valid geography (STATE_CD, CNTY_CD)
--  11. Require continuous FFS enrollment for all 24 months before death
-- Adapted from: opscc/queries/opscc_cohort.sql
-- NOTE: 24-month lookback spans 2 calendar years, so mbsf_all must be used
--   (io_decedents only has the death-year MBSF row = 12 months).
--   Performance: JOIN io_episodes (HNC+IO patients, ~few K) instead of
--   io_decedents (14.5M) to filter mbsf_all at source.
-- Output: io_ffs_eligible

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_ffs_eligible;

CREATE TABLE io_ffs_eligible AS

WITH hnc_io AS (
    -- Narrow to HNC+IO patients only (~7K), pulling death/geography info from io_decedents
    SELECT d.DSYSRTKY, d.death_dt_parsed, d.OREC, d.STATE_CD, d.CNTY_CD
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
    JOIN io_decedents d ON e.DSYSRTKY = d.DSYSRTKY
),

enroll AS (
    -- Join mbsf_all against HNC+IO patients only (not all 14.5M decedents)
    SELECT
        m.DSYSRTKY,
        h.death_dt_parsed,
        make_date(CAST(m.RFRNC_YR AS INTEGER), mth, 1) AS month_start,
        CASE
            WHEN (
                CASE mth
                    WHEN 1  THEN m.HMOIND1   WHEN 2  THEN m.HMOIND2
                    WHEN 3  THEN m.HMOIND3   WHEN 4  THEN m.HMOIND4
                    WHEN 5  THEN m.HMOIND5   WHEN 6  THEN m.HMOIND6
                    WHEN 7  THEN m.HMOIND7   WHEN 8  THEN m.HMOIND8
                    WHEN 9  THEN m.HMOIND9   WHEN 10 THEN m.HMOIND10
                    WHEN 11 THEN m.HMOIND11  WHEN 12 THEN m.HMOIND12
                END
            ) = '0'
            AND (
                CASE mth
                    WHEN 1  THEN m.MDCR_STUS_CD_01 WHEN 2  THEN m.MDCR_STUS_CD_02
                    WHEN 3  THEN m.MDCR_STUS_CD_03 WHEN 4  THEN m.MDCR_STUS_CD_04
                    WHEN 5  THEN m.MDCR_STUS_CD_05 WHEN 6  THEN m.MDCR_STUS_CD_06
                    WHEN 7  THEN m.MDCR_STUS_CD_07 WHEN 8  THEN m.MDCR_STUS_CD_08
                    WHEN 9  THEN m.MDCR_STUS_CD_09 WHEN 10 THEN m.MDCR_STUS_CD_10
                    WHEN 11 THEN m.MDCR_STUS_CD_11 WHEN 12 THEN m.MDCR_STUS_CD_12
                END
            ) IN ('10','11','20','21')
            THEN 1 ELSE 0
        END AS ffs_month
    FROM mbsf_all m
    JOIN hnc_io h ON m.DSYSRTKY = h.DSYSRTKY,
    UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mth)
),

ffs_coverage AS (
    SELECT
        e.DSYSRTKY,
        SUM(e.ffs_month) AS ffs_months
    FROM enroll e
    WHERE e.month_start BETWEEN
        date_trunc('month', e.death_dt_parsed) - INTERVAL 23 MONTH
        AND date_trunc('month', e.death_dt_parsed)
    GROUP BY e.DSYSRTKY
    HAVING SUM(e.ffs_month) = 24
)

SELECT h.DSYSRTKY

FROM hnc_io h
JOIN ffs_coverage fc ON h.DSYSRTKY = fc.DSYSRTKY

-- Exclusion 9: ESRD
WHERE h.OREC <> '2'
-- Exclusion 10: Valid geography
  AND h.STATE_CD IS NOT NULL AND h.STATE_CD <> ''
  AND h.CNTY_CD  IS NOT NULL AND h.CNTY_CD  <> '';
