-- Step 12b: Sensitivity analysis variables for ITT cohort (io_cohort_itt)
-- Identical logic to 12a_io_sensitivity.sql scoped to io_cohort_itt.
-- Requires io_cohort_itt to exist (step 09b).
-- Output: io_sensitivity_vars_itt

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_sensitivity_vars_itt;

CREATE TABLE io_sensitivity_vars_itt AS

WITH last_ep_claims AS (
    SELECT
        cr.DSYSRTKY,
        cr.io_date,
        LAG(cr.io_date) OVER (PARTITION BY cr.DSYSRTKY ORDER BY cr.io_date) AS prev_date
    FROM io_claims_raw cr
    JOIN io_episodes ep ON cr.DSYSRTKY = ep.DSYSRTKY
    WHERE cr.io_date BETWEEN ep.last_io_episode_start AND ep.last_io_episode_end
      AND cr.DSYSRTKY IN (SELECT DSYSRTKY FROM io_cohort_itt)
),

intervals AS (
    SELECT
        DSYSRTKY,
        datediff('day', prev_date, io_date) AS interdose_days
    FROM last_ep_claims
    WHERE prev_date IS NOT NULL
      AND datediff('day', prev_date, io_date) > 0
),

patient_median AS (
    SELECT
        DSYSRTKY,
        MEDIAN(interdose_days)  AS median_interdose_days,
        COUNT(*)                AS n_intervals
    FROM intervals
    GROUP BY DSYSRTKY
)

SELECT
    c.DSYSRTKY,
    pm.median_interdose_days,
    pm.n_intervals,
    CASE
        WHEN pm.median_interdose_days IS NULL THEN 'single-dose'
        WHEN pm.median_interdose_days <= 21   THEN 'q2w'
        WHEN pm.median_interdose_days <= 35   THEN 'q3w'
        WHEN pm.median_interdose_days <= 56   THEN 'q6w'
        ELSE 'other'
    END AS estimated_regimen,
    CASE
        WHEN pm.median_interdose_days IS NULL THEN 42
        ELSE CAST(pm.median_interdose_days AS INT) + 14
    END AS discontinued_threshold_days

FROM (SELECT DSYSRTKY FROM io_cohort_itt) c
LEFT JOIN patient_median pm ON c.DSYSRTKY = pm.DSYSRTKY;
