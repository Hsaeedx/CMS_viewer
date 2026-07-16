-- opscc_outcomes.sql  (two-step: extract cohort claims first, then look up codes)
--
-- Step 1: Pull each cohort patient's claims into small temp tables (one scan per
--         source table). This is the only time the large parquet tables are touched.
-- Step 2: All outcome code lookups run on the small temp tables — very fast.
--
-- Pre-existing rule: any matching code in [first_tx_date - 365d, first_tx_date)
--   -> all fields for that outcome are NULL.
--
-- Outcomes:
--   dysphagia    : ICD10CM R131* (R1310-R1319 — prefix match LEFT(code,4)='R131')
--
-- G-tube events are handled separately in opscc_gtube_dependence.sql
-- using a multi-interval cumulative-incidence framework.
--
-- ICD-10-CM diagnosis codes use LEFT(code,4) prefix matching because claims bill
-- the most specific subcode (e.g. R1311, R1312) not the parent (R131).

-- ── STEP 1: Cohort ────────────────────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _cohort AS
SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
       -- Treatment completion = last RT/chemo date; falls back to first_tx_date
       -- (TORS-alone patients have no RT/chemo). Anchor for delayed-toxicity metrics.
       COALESCE(list_max([co.last_rt_date, co.last_chemo_date]), p.first_tx_date) AS tx_completion_date
FROM opscc_propensity p
LEFT JOIN opscc_cohort co ON co.DSYSRTKY = p.DSYSRTKY
WHERE p.first_tx_date IS NOT NULL;

-- ── STEP 2: Extract cohort patients' claims (one scan per large table) ────────

-- Inpatient: dx codes, date-filtered to lookback + post-tx window
CREATE OR REPLACE TEMP TABLE _inp AS
SELECT
    c.DSYSRTKY, c.first_tx_date, c.tx_completion_date,
    TRY_STRPTIME(i.THRU_DT, '%Y%m%d')  AS event_dt,
    i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
    i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,
    i.ICD_DGNS_CD4,  i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,
    i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,  i.ICD_DGNS_CD9,
    i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
    i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15
FROM _cohort c
JOIN inp_claimsk_all i ON i.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(i.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY;

-- Outpatient: dx codes only, date-filtered
CREATE OR REPLACE TEMP TABLE _out AS
SELECT
    c.DSYSRTKY, c.first_tx_date, c.tx_completion_date,
    TRY_STRPTIME(o.THRU_DT, '%Y%m%d')  AS event_dt,
    o.PRNCPAL_DGNS_CD,
    o.ICD_DGNS_CD1,  o.ICD_DGNS_CD2,  o.ICD_DGNS_CD3,
    o.ICD_DGNS_CD4,  o.ICD_DGNS_CD5,  o.ICD_DGNS_CD6,
    o.ICD_DGNS_CD7,  o.ICD_DGNS_CD8,  o.ICD_DGNS_CD9,
    o.ICD_DGNS_CD10
FROM _cohort c
JOIN out_claimsk_all o ON o.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(o.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY;

-- Carrier line: dysphagia dx only
CREATE OR REPLACE TEMP TABLE _car AS
SELECT
    c.DSYSRTKY, c.first_tx_date, c.tx_completion_date,
    TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS event_dt,
    cl.LINE_ICD_DGNS_CD
FROM _cohort c
JOIN car_linek_all cl ON cl.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY
  AND LEFT(cl.LINE_ICD_DGNS_CD, 4) = 'R131';

-- ── STEP 3: Outcome lookups on the small temp tables ─────────────────────────

DROP TABLE IF EXISTS opscc_outcomes;

CREATE TABLE opscc_outcomes AS

WITH

-- Inpatient dx codes -> dysphagia (prefix match)
inp_dx AS (
    SELECT DSYSRTKY, first_tx_date, tx_completion_date, event_dt, 'dysphagia' AS outcome
    FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5,  ICD_DGNS_CD6,
            ICD_DGNS_CD7,  ICD_DGNS_CD8,  ICD_DGNS_CD9,
            ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE LEFT(t.code, 4) = 'R131'
),

-- Outpatient dx codes -> dysphagia (prefix match)
out_dx AS (
    SELECT DSYSRTKY, first_tx_date, tx_completion_date, event_dt, 'dysphagia' AS outcome
    FROM _out,
    UNNEST([PRNCPAL_DGNS_CD,
            ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4, ICD_DGNS_CD5, ICD_DGNS_CD6,
            ICD_DGNS_CD7, ICD_DGNS_CD8, ICD_DGNS_CD9,
            ICD_DGNS_CD10]) AS t(code)
    WHERE LEFT(t.code, 4) = 'R131'
),

-- Carrier dx codes -> dysphagia (already pre-filtered at extraction)
car_dx AS (
    SELECT DSYSRTKY, first_tx_date, tx_completion_date, event_dt, 'dysphagia' AS outcome
    FROM _car
),

-- All events combined
all_events AS (
    SELECT * FROM inp_dx
    UNION ALL SELECT * FROM out_dx
    UNION ALL SELECT * FROM car_dx
),

-- Per patient per outcome: pre-existing flag + first post-treatment date
outcome_summary AS (
    SELECT
        DSYSRTKY,
        outcome,
        MIN(first_tx_date)                                         AS first_tx_date,
        BOOL_OR(event_dt < first_tx_date)                         AS pre_existing,
        MIN(CASE WHEN event_dt >= first_tx_date THEN event_dt END) AS first_post_date,
        -- Delayed-toxicity (completion anchor, 90d washout)
        -- Incident-eligibility: any dysphagia between treatment start and completion+90d.
        BOOL_OR(event_dt >= first_tx_date AND event_dt <= tx_completion_date + INTERVAL 90 DAY) AS through_completion90,
        -- First dysphagia strictly >90d after completion (skips earlier acute dysphagia).
        MIN(CASE WHEN event_dt > tx_completion_date + INTERVAL 90 DAY THEN event_dt END)        AS first_delayed_date
    FROM all_events
    WHERE outcome IS NOT NULL
    GROUP BY DSYSRTKY, outcome
)

-- Final wide table
SELECT
    c.DSYSRTKY,
    c.tx_group,
    c.first_tx_date,

    -- Dysphagia
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_post_date IS NOT NULL) END AS has_dysphagia,
    CASE WHEN dys.pre_existing THEN NULL ELSE dys.first_post_date               END AS first_dysphagia_date,
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_post_date - c.first_tx_date) END AS days_to_dysphagia,

    -- Delayed dysphagia (completion anchor, 90d washout) — marker for delayed swallowing failure
    c.tx_completion_date,
    DATE_DIFF('day', c.first_tx_date, c.tx_completion_date) AS days_tx_to_completion,
    CASE WHEN dys.pre_existing THEN NULL ELSE COALESCE(dys.through_completion90, FALSE) END AS dysphagia_through_completion90,
    CASE WHEN dys.pre_existing THEN NULL ELSE dys.first_delayed_date END AS first_delayed_dysphagia_date,
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_delayed_date IS NOT NULL AND dys.first_delayed_date <= c.tx_completion_date + INTERVAL  180 DAY) END AS delayed_dysphagia_by_180d,
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_delayed_date IS NOT NULL AND dys.first_delayed_date <= c.tx_completion_date + INTERVAL  365 DAY) END AS delayed_dysphagia_by_365d,
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_delayed_date IS NOT NULL AND dys.first_delayed_date <= c.tx_completion_date + INTERVAL  730 DAY) END AS delayed_dysphagia_by_730d,
    CASE WHEN dys.pre_existing THEN NULL ELSE (dys.first_delayed_date IS NOT NULL AND dys.first_delayed_date <= c.tx_completion_date + INTERVAL 1095 DAY) END AS delayed_dysphagia_by_1095d

FROM _cohort c
LEFT JOIN outcome_summary dys ON dys.DSYSRTKY = c.DSYSRTKY AND dys.outcome = 'dysphagia';

-- Summary after build
SELECT
    tx_group,
    COUNT(*)                                                          AS n_patients,
    COUNT(has_dysphagia)                                              AS n_dysphagia_eligible,
    SUM(has_dysphagia::INT)                                           AS n_dysphagia,
    ROUND(100.0 * SUM(has_dysphagia::INT) / COUNT(has_dysphagia), 1)  AS pct_dysphagia
FROM opscc_outcomes
GROUP BY tx_group
ORDER BY tx_group;
