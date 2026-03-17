-- Step 6: Gap-based IO episode algorithm
-- Gap threshold: ≥120 days between claims = new episode
-- Identifies LAST episode (closest to death) as index palliative episode
-- Source: io_claims_raw
-- Output: io_episodes

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_episodes;

CREATE TABLE io_episodes AS

WITH io_ordered AS (
    SELECT
        DSYSRTKY,
        io_date,
        HCPCS_CD,
        LAG(io_date) OVER (PARTITION BY DSYSRTKY ORDER BY io_date) AS prev_io_date
    FROM io_claims_raw
    WHERE io_date IS NOT NULL
),

episode_boundaries AS (
    SELECT
        *,
        CASE
            WHEN prev_io_date IS NULL THEN 1
            WHEN datediff('day', prev_io_date, io_date) >= 120 THEN 1
            ELSE 0
        END AS new_episode_flag
    FROM io_ordered
),

episode_assignments AS (
    SELECT
        *,
        SUM(new_episode_flag) OVER (
            PARTITION BY DSYSRTKY
            ORDER BY io_date
            ROWS UNBOUNDED PRECEDING
        ) AS episode_num
    FROM episode_boundaries
),

episode_summary AS (
    SELECT
        DSYSRTKY,
        episode_num,
        MIN(io_date) AS episode_start,
        MAX(io_date) AS episode_end,
        COUNT(DISTINCT io_date) AS episode_doses,
        -- Determine agent: both if mixed, else single
        CASE
            WHEN COUNT(DISTINCT HCPCS_CD) > 1 THEN 'both'
            WHEN MAX(HCPCS_CD) = 'J9271' THEN 'pembrolizumab'
            WHEN MAX(HCPCS_CD) = 'J9299' THEN 'nivolumab'
            ELSE MAX(HCPCS_CD)
        END AS episode_agent
    FROM episode_assignments
    GROUP BY DSYSRTKY, episode_num
),

-- Total IO summary per patient (across all episodes)
patient_totals AS (
    SELECT
        DSYSRTKY,
        COUNT(DISTINCT episode_num) AS total_io_episodes,
        MIN(episode_start) AS first_io_date,
        MAX(episode_end) AS last_io_date,
        SUM(episode_doses) AS total_io_doses,
        -- Overall agent
        CASE
            WHEN COUNT(DISTINCT episode_agent) > 1
                 OR BOOL_OR(episode_agent = 'both') THEN 'both'
            ELSE MAX(episode_agent)
        END AS io_agent
    FROM episode_summary
    GROUP BY DSYSRTKY
),

-- Last episode only
last_ep AS (
    SELECT
        es.*,
        ROW_NUMBER() OVER (PARTITION BY es.DSYSRTKY ORDER BY es.episode_num DESC) AS rn
    FROM episode_summary es
)

SELECT
    le.DSYSRTKY,
    le.episode_start AS last_io_episode_start,
    le.episode_end AS last_io_episode_end,
    le.episode_doses AS last_episode_doses,
    le.episode_agent AS last_episode_agent,
    pt.total_io_episodes,
    pt.first_io_date,
    pt.last_io_date,
    pt.total_io_doses,
    pt.io_agent,
    datediff('day', pt.first_io_date, pt.last_io_date) AS io_duration_days

FROM last_ep le
JOIN patient_totals pt ON le.DSYSRTKY = pt.DSYSRTKY
WHERE le.rn = 1;
