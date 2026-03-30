-- Step 12a: Sensitivity analysis variables — median interdose interval
-- For each cohort patient, computes the median time between IO doses
-- within their final episode, classifies estimated dosing regimen, and
-- derives a personalized discontinuation threshold.
--
-- Discontinuation threshold = median_interdose_days + 14 days (tolerance)
-- Single-dose patients (no intervals) default to 42 days (q6w + 2-week buffer)
--
-- Output: io_sensitivity_vars

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_sensitivity_vars;

CREATE TABLE io_sensitivity_vars AS

WITH last_ep_claims AS (
    -- All IO claims for each cohort patient scoped to their final episode
    SELECT
        cr.DSYSRTKY,
        cr.io_date,
        LAG(cr.io_date) OVER (PARTITION BY cr.DSYSRTKY ORDER BY cr.io_date) AS prev_date
    FROM io_claims_raw cr
    JOIN io_episodes ep ON cr.DSYSRTKY = ep.DSYSRTKY
    WHERE cr.io_date BETWEEN ep.last_io_episode_start AND ep.last_io_episode_end
      AND cr.DSYSRTKY IN (SELECT DSYSRTKY FROM io_cohort)
),

intervals AS (
    -- Consecutive interdose intervals within the final episode
    SELECT
        DSYSRTKY,
        datediff('day', prev_date, io_date) AS interdose_days
    FROM last_ep_claims
    WHERE prev_date IS NOT NULL   -- skip first dose (no prior dose in episode)
      AND datediff('day', prev_date, io_date) > 0  -- guard against same-day duplicates
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
    -- Discontinued threshold: median interval + 14-day tolerance
    -- Default 42 days for single-dose patients (conservative q6w assumption)
    CASE
        WHEN pm.median_interdose_days IS NULL THEN 42
        ELSE CAST(pm.median_interdose_days AS INT) + 14
    END AS discontinued_threshold_days

FROM (SELECT DSYSRTKY FROM io_cohort) c
LEFT JOIN patient_median pm ON c.DSYSRTKY = pm.DSYSRTKY;

