-- opscc_slp.sql  —  Multi-interval SLP visit tracking
--
-- Scoped to PSM-matched patients (Comp A, B, or C).
-- Frames events around dx_date AND first_tx_date.
-- Reports cumulative-incidence flags at 14d, 30d, 90d, 180d, 1yr, 3yr post-tx.
-- Reports event-count distributions per window + pre-tx baseline events.
--
-- SLP codes (from stroke_SLP/codes.json):
--   HCPCS/CPT:
--     Evaluation  : 92521, 92522, 92523
--     Treatment   : 92507, 92526, 97129, 97130
--     Swallowing  : 92610, 92611, 92612, 92616, 92617
--   Revenue center: 0440-0449 (Speech-Language Pathology)

-- ── STEP 1: Matched cohort ──────────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_cohort AS
SELECT p.DSYSRTKY,
       p.first_hnc_date::DATE AS dx_date,
       p.first_tx_date,
       -- Treatment completion = last RT/chemo date; falls back to first_tx_date
       -- (TORS-alone patients have no RT/chemo). Anchor for delayed-toxicity metrics.
       COALESCE(list_max([co.last_rt_date, co.last_chemo_date]), p.first_tx_date) AS tx_completion_date
FROM opscc_propensity p
LEFT JOIN opscc_cohort co ON co.DSYSRTKY = p.DSYSRTKY
WHERE p.first_tx_date IS NOT NULL
  AND (p.psm_matched_A = TRUE OR p.psm_matched_B = TRUE OR p.psm_matched_C = TRUE);

-- ── STEP 2: Extract SLP events from all sources ────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_car AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS dt
FROM _slp_cohort c
JOIN car_linek_all cl USING (DSYSRTKY)
WHERE cl.HCPCS_CD IN ('92521','92522','92523',
                       '92507','92526','97129','97130',
                       '92610','92611','92612','92616','92617');

CREATE OR REPLACE TEMP TABLE _slp_rev AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS dt
FROM _slp_cohort c
JOIN out_revenuek_all r USING (DSYSRTKY)
WHERE r.HCPCS_CD IN ('92521','92522','92523',
                      '92507','92526','97129','97130',
                      '92610','92611','92612','92616','92617')
   OR LEFT(r.REV_CNTR, 3) = '044';

CREATE OR REPLACE TEMP TABLE _slp_snf AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS dt
FROM _slp_cohort c
JOIN snf_revenuek_all r USING (DSYSRTKY)
WHERE LEFT(r.REV_CNTR, 3) = '044';

CREATE OR REPLACE TEMP TABLE _slp_hha AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS dt
FROM _slp_cohort c
JOIN hha_revenuek_all r USING (DSYSRTKY)
WHERE LEFT(r.REV_CNTR, 3) = '044';

-- ── STEP 3: Union all events ───────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_events AS
SELECT DSYSRTKY, dx_date, first_tx_date, dt FROM _slp_car WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt FROM _slp_rev WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt FROM _slp_snf WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt FROM _slp_hha WHERE dt IS NOT NULL;

-- ── STEP 4: Per-patient summary ────────────────────────────────────────────

DROP TABLE IF EXISTS opscc_slp;

CREATE TABLE opscc_slp AS
WITH per_pt AS (
    SELECT
        c.DSYSRTKY,
        c.dx_date,
        c.first_tx_date,
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date
             THEN NULL
             ELSE DATE_DIFF('day', c.dx_date, c.first_tx_date)
        END AS days_dx_to_tx,

        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            (COUNT(DISTINCT CASE WHEN e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END) > 0)
        END AS pre_event_any,
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            COUNT(DISTINCT CASE WHEN e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END)
        END AS pre_event_count,

        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  14 DAY THEN e.dt END) > 0) AS event_by_14d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  30 DAY THEN e.dt END) > 0) AS event_by_30d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  90 DAY THEN e.dt END) > 0) AS event_by_90d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 180 DAY THEN e.dt END) > 0) AS event_by_180d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 365 DAY THEN e.dt END) > 0) AS event_by_365d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) > 0) AS event_by_1095d,

        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date AND e.dt <  c.first_tx_date + INTERVAL  14 DAY THEN e.dt END) AS cnt_0_14d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL  14 DAY AND e.dt <  c.first_tx_date + INTERVAL  30 DAY THEN e.dt END) AS cnt_14_30d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL  30 DAY AND e.dt <  c.first_tx_date + INTERVAL  90 DAY THEN e.dt END) AS cnt_30_90d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL  90 DAY AND e.dt <  c.first_tx_date + INTERVAL 180 DAY THEN e.dt END) AS cnt_90_180d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL 180 DAY AND e.dt <  c.first_tx_date + INTERVAL 365 DAY THEN e.dt END) AS cnt_180_365d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL 365 DAY AND e.dt <  c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) AS cnt_365_1095d,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) AS cnt_gt_1095d,

        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date THEN e.dt END)         AS total_post_events,
        MIN(CASE WHEN e.dt >= c.first_tx_date THEN DATE_DIFF('day', c.first_tx_date, e.dt) END) AS first_post_day_from_tx,
        MAX(CASE WHEN e.dt >= c.first_tx_date THEN DATE_DIFF('day', c.first_tx_date, e.dt) END) AS last_post_day_from_tx,

        -- ── DELAYED SLP (anchored on treatment completion, 90d washout) ──
        c.tx_completion_date,
        DATE_DIFF('day', c.first_tx_date, c.tx_completion_date) AS days_tx_to_completion,
        -- Incident-eligibility: any SLP visit up through completion + 90d washout.
        (COUNT(DISTINCT CASE WHEN e.dt <= c.tx_completion_date + INTERVAL 90 DAY THEN e.dt END) > 0) AS slp_through_completion90,
        MIN(CASE WHEN e.dt > c.tx_completion_date + INTERVAL 90 DAY
                 THEN DATE_DIFF('day', c.tx_completion_date, e.dt) END) AS first_delayed_slp_day,
        (COUNT(DISTINCT CASE WHEN e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  180 DAY THEN e.dt END) > 0) AS delayed_slp_by_180d,
        (COUNT(DISTINCT CASE WHEN e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  365 DAY THEN e.dt END) > 0) AS delayed_slp_by_365d,
        (COUNT(DISTINCT CASE WHEN e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  730 DAY THEN e.dt END) > 0) AS delayed_slp_by_730d,
        (COUNT(DISTINCT CASE WHEN e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL 1095 DAY THEN e.dt END) > 0) AS delayed_slp_by_1095d
    FROM _slp_cohort c
    LEFT JOIN _slp_events e ON e.DSYSRTKY = c.DSYSRTKY
    GROUP BY c.DSYSRTKY, c.dx_date, c.first_tx_date, c.tx_completion_date
)
SELECT * FROM per_pt;

-- ── Summary print ──────────────────────────────────────────────────────────

SELECT
    p.tx_group,
    COUNT(*) AS n,
    ROUND(100.0 * SUM(s.event_by_14d::INT)   / COUNT(*), 1) AS pct_by_14d,
    ROUND(100.0 * SUM(s.event_by_30d::INT)   / COUNT(*), 1) AS pct_by_30d,
    ROUND(100.0 * SUM(s.event_by_90d::INT)   / COUNT(*), 1) AS pct_by_90d,
    ROUND(100.0 * SUM(s.event_by_180d::INT)  / COUNT(*), 1) AS pct_by_180d,
    ROUND(100.0 * SUM(s.event_by_365d::INT)  / COUNT(*), 1) AS pct_by_1yr,
    ROUND(100.0 * SUM(s.event_by_1095d::INT) / COUNT(*), 1) AS pct_by_3yr
FROM opscc_propensity p
JOIN opscc_slp s USING (DSYSRTKY)
WHERE p.psm_matched_A OR p.psm_matched_B OR p.psm_matched_C
GROUP BY p.tx_group ORDER BY p.tx_group;
