-- Step 11b: Outcome definitions for ITT cohort (io_cohort_itc)
-- Identical to 11_io_hospice.sql; io_cohort replaced with io_cohort_itc
-- Output: io_outcomes_itc

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_outcomes_itc;

CREATE TABLE io_outcomes_itc AS

WITH hospice_raw AS (
    SELECT
        h.DSYSRTKY,
        TRY_STRPTIME(h.HSPCSTRT, '%Y%m%d') AS hospice_start
    FROM io_hosp_claims h
    JOIN io_cohort_itc c ON h.DSYSRTKY = c.DSYSRTKY
    WHERE h.HSPCSTRT IS NOT NULL
      AND h.HSPCSTRT <> ''
      AND TRY_STRPTIME(h.HSPCSTRT, '%Y%m%d') < c.death_dt
),

hospice_summary AS (
    SELECT
        DSYSRTKY,
        1 AS hospice_enrolled,
        MIN(hospice_start) AS hospice_election_date
    FROM hospice_raw
    GROUP BY DSYSRTKY
),

inhospital_raw AS (
    SELECT DISTINCT
        i.DSYSRTKY,
        1 AS in_hospital_death
    FROM io_inp_claims i
    JOIN io_cohort_itc c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE i.STUS_CD = '20'
      AND i.DSCHRGDT IS NOT NULL
      AND i.DSCHRGDT <> ''
      AND ABS(datediff('day',
            TRY_STRPTIME(i.DSCHRGDT, '%Y%m%d'),
            c.death_dt
          )) <= 1
)

SELECT
    c.DSYSRTKY,
    COALESCE(hs.hospice_enrolled, 0) AS hospice_enrolled,
    hs.hospice_election_date,
    CASE
        WHEN hs.hospice_election_date IS NOT NULL
        THEN datediff('day', hs.hospice_election_date, c.death_dt)
        ELSE NULL
    END AS hospice_los_days,
    CASE
        WHEN hs.hospice_election_date IS NOT NULL
             AND datediff('day', hs.hospice_election_date, c.death_dt) <= 7
        THEN 1 ELSE 0
    END AS hospice_short_stay,
    CASE
        WHEN hs.hospice_election_date IS NOT NULL
        THEN datediff('day', c.last_io_date, hs.hospice_election_date)
        ELSE NULL
    END AS days_last_io_to_hospice,
    COALESCE(ih.in_hospital_death, 0) AS in_hospital_death

FROM io_cohort_itc c
LEFT JOIN hospice_summary hs ON c.DSYSRTKY = hs.DSYSRTKY
LEFT JOIN inhospital_raw  ih ON c.DSYSRTKY = ih.DSYSRTKY;
