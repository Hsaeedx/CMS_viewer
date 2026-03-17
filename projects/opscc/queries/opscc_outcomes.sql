-- opscc_outcomes.sql  (two-step: extract cohort claims first, then look up codes)
--
-- Step 1: Pull each cohort patient's claims into small temp tables (one scan per
--         source table). This is the only time the large parquet tables are touched.
-- Step 2: All outcome code lookups run on the small temp tables — very fast.
--
-- Pre-existing rule: any matching code in [first_tx_date - 365d, first_tx_date)
--   -> all fields for that outcome are NULL.
--
-- Outcomes (outcomes.json):
--   dysphagia    : ICD10CM R131* (R1310-R1319 — prefix match LEFT(code,4)='R131')
--   gtube        : ICD10CM Z931* (prefix) | CPT 43246, 49440 | HCPCS B4087, B4088
--                  ICD10PCS 0DH63UZ, 0DH60UZ (exact)
--   trach        : ICD10CM Z930* (prefix) | CPT 31600, 31603, 31610, 31612 (exact)
--
-- ICD-10-CM diagnosis codes use LEFT(code,4) prefix matching because claims bill
-- the most specific subcode (e.g. R1311, R1312) not the parent (R131).
-- CPT, HCPCS, and ICD-10-PCS codes are exact matches.

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

-- ── STEP 1: Cohort ────────────────────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _cohort AS
SELECT DSYSRTKY, tx_group, first_tx_date
FROM opscc_propensity
WHERE first_tx_date IS NOT NULL;

-- ── STEP 2: Extract cohort patients' claims (one scan per large table) ────────

-- Inpatient: dx + procedure codes, date-filtered to lookback + post-tx window
CREATE OR REPLACE TEMP TABLE _inp AS
SELECT
    c.DSYSRTKY, c.first_tx_date,
    TRY_STRPTIME(i.THRU_DT, '%Y%m%d')  AS event_dt,
    i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
    i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,
    i.ICD_DGNS_CD4,  i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,
    i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,  i.ICD_DGNS_CD9,
    i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
    i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15,
    i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,
    i.ICD_PRCDR_CD4,  i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,
    i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,  i.ICD_PRCDR_CD9,
    i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
    i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15
FROM _cohort c
JOIN inp_claimsk_all i ON i.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(i.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY;

-- Outpatient: dx codes only, date-filtered
CREATE OR REPLACE TEMP TABLE _out AS
SELECT
    c.DSYSRTKY, c.first_tx_date,
    TRY_STRPTIME(o.THRU_DT, '%Y%m%d')  AS event_dt,
    o.PRNCPAL_DGNS_CD,
    o.ICD_DGNS_CD1,  o.ICD_DGNS_CD2,  o.ICD_DGNS_CD3,
    o.ICD_DGNS_CD4,  o.ICD_DGNS_CD5,  o.ICD_DGNS_CD6,
    o.ICD_DGNS_CD7,  o.ICD_DGNS_CD8,  o.ICD_DGNS_CD9,
    o.ICD_DGNS_CD10
FROM _cohort c
JOIN out_claimsk_all o ON o.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(o.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY;

-- Carrier line: already one code per row — filter by relevant codes at extraction
CREATE OR REPLACE TEMP TABLE _car AS
SELECT
    c.DSYSRTKY, c.first_tx_date,
    TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS event_dt,
    cl.LINE_ICD_DGNS_CD,
    cl.HCPCS_CD
FROM _cohort c
JOIN car_linek_all cl ON cl.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY
  AND (LEFT(cl.LINE_ICD_DGNS_CD, 4) IN ('R131', 'Z931', 'Z930')
    OR cl.HCPCS_CD                  IN ('43246', '49440', 'B4087', 'B4088',
                                        '31600', '31603', '31610', '31612'));

-- Outpatient revenue center: one HCPCS per row — filter at extraction
CREATE OR REPLACE TEMP TABLE _rev AS
SELECT
    c.DSYSRTKY, c.first_tx_date,
    TRY_STRPTIME(r.THRU_DT, '%Y%m%d')  AS event_dt,
    r.HCPCS_CD
FROM _cohort c
JOIN out_revenuek_all r ON r.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(r.THRU_DT, '%Y%m%d') >= c.first_tx_date - INTERVAL 365 DAY
  AND r.HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088',
                      '31600', '31603', '31610', '31612');

-- ── STEP 3: Outcome lookups on the small temp tables ─────────────────────────

DROP TABLE IF EXISTS opscc_outcomes;

CREATE TABLE opscc_outcomes AS

WITH

-- Inpatient dx codes -> outcomes  (prefix match on first 4 chars)
inp_dx AS (
    SELECT DSYSRTKY, first_tx_date, event_dt,
           CASE LEFT(t.code, 4)
               WHEN 'R131' THEN 'dysphagia'
               WHEN 'Z931' THEN 'gtube'
               WHEN 'Z930' THEN 'trach'
           END AS outcome
    FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5,  ICD_DGNS_CD6,
            ICD_DGNS_CD7,  ICD_DGNS_CD8,  ICD_DGNS_CD9,
            ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE LEFT(t.code, 4) IN ('R131', 'Z931', 'Z930')
),

-- Inpatient ICD10-PCS procedure codes -> gtube
inp_pcs AS (
    SELECT DSYSRTKY, first_tx_date, event_dt, 'gtube' AS outcome
    FROM _inp,
    UNNEST([ICD_PRCDR_CD1,  ICD_PRCDR_CD2,  ICD_PRCDR_CD3,
            ICD_PRCDR_CD4,  ICD_PRCDR_CD5,  ICD_PRCDR_CD6,
            ICD_PRCDR_CD7,  ICD_PRCDR_CD8,  ICD_PRCDR_CD9,
            ICD_PRCDR_CD10, ICD_PRCDR_CD11, ICD_PRCDR_CD12,
            ICD_PRCDR_CD13, ICD_PRCDR_CD14, ICD_PRCDR_CD15]) AS t(code)
    WHERE t.code IN ('0DH63UZ', '0DH60UZ')
),

-- Outpatient dx codes -> outcomes  (prefix match)
out_dx AS (
    SELECT DSYSRTKY, first_tx_date, event_dt,
           CASE LEFT(t.code, 4)
               WHEN 'R131' THEN 'dysphagia'
               WHEN 'Z931' THEN 'gtube'
               WHEN 'Z930' THEN 'trach'
           END AS outcome
    FROM _out,
    UNNEST([PRNCPAL_DGNS_CD,
            ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4, ICD_DGNS_CD5, ICD_DGNS_CD6,
            ICD_DGNS_CD7, ICD_DGNS_CD8, ICD_DGNS_CD9,
            ICD_DGNS_CD10]) AS t(code)
    WHERE LEFT(t.code, 4) IN ('R131', 'Z931', 'Z930')
),

-- Carrier dx codes -> outcomes  (prefix match, already pre-filtered at extraction)
car_dx AS (
    SELECT DSYSRTKY, first_tx_date, event_dt,
           CASE LEFT(LINE_ICD_DGNS_CD, 4)
               WHEN 'R131' THEN 'dysphagia'
               WHEN 'Z931' THEN 'gtube'
               WHEN 'Z930' THEN 'trach'
           END AS outcome
    FROM _car
    WHERE LEFT(LINE_ICD_DGNS_CD, 4) IN ('R131', 'Z931', 'Z930')
),

-- Carrier HCPCS codes -> outcomes  (already filtered)
car_cpt AS (
    SELECT DSYSRTKY, first_tx_date, event_dt,
           CASE
               WHEN HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088') THEN 'gtube'
               WHEN HCPCS_CD IN ('31600', '31603', '31610', '31612') THEN 'trach'
           END AS outcome
    FROM _car
    WHERE HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088',
                        '31600', '31603', '31610', '31612')
),

-- Revenue center HCPCS codes -> outcomes  (already filtered)
rev_cpt AS (
    SELECT DSYSRTKY, first_tx_date, event_dt,
           CASE
               WHEN HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088') THEN 'gtube'
               WHEN HCPCS_CD IN ('31600', '31603', '31610', '31612') THEN 'trach'
           END AS outcome
    FROM _rev
),

-- All events combined
all_events AS (
    SELECT * FROM inp_dx
    UNION ALL SELECT * FROM inp_pcs
    UNION ALL SELECT * FROM out_dx
    UNION ALL SELECT * FROM car_dx
    UNION ALL SELECT * FROM car_cpt
    UNION ALL SELECT * FROM rev_cpt
),

-- Per patient per outcome: pre-existing flag + first post-treatment date
outcome_summary AS (
    SELECT
        DSYSRTKY,
        outcome,
        MIN(first_tx_date)                                         AS first_tx_date,
        BOOL_OR(event_dt < first_tx_date)                         AS pre_existing,
        MIN(CASE WHEN event_dt >= first_tx_date THEN event_dt END) AS first_post_date
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

    -- Gastrostomy tube
    CASE WHEN gtu.pre_existing THEN NULL ELSE (gtu.first_post_date IS NOT NULL) END AS has_gtube,
    CASE WHEN gtu.pre_existing THEN NULL ELSE gtu.first_post_date               END AS first_gtube_date,
    CASE WHEN gtu.pre_existing THEN NULL ELSE (gtu.first_post_date - c.first_tx_date) END AS days_to_gtube,

    -- Tracheostomy
    CASE WHEN tra.pre_existing THEN NULL ELSE (tra.first_post_date IS NOT NULL) END AS has_tracheostomy,
    CASE WHEN tra.pre_existing THEN NULL ELSE tra.first_post_date               END AS first_trach_date,
    CASE WHEN tra.pre_existing THEN NULL ELSE (tra.first_post_date - c.first_tx_date) END AS days_to_trach

FROM _cohort c
LEFT JOIN outcome_summary dys ON dys.DSYSRTKY = c.DSYSRTKY AND dys.outcome = 'dysphagia'
LEFT JOIN outcome_summary gtu ON gtu.DSYSRTKY = c.DSYSRTKY AND gtu.outcome = 'gtube'
LEFT JOIN outcome_summary tra ON tra.DSYSRTKY = c.DSYSRTKY AND tra.outcome = 'trach';

-- Summary after build
SELECT
    tx_group,
    COUNT(*)                                                              AS n_patients,
    COUNT(has_dysphagia)                                                  AS n_dysphagia_eligible,
    SUM(has_dysphagia::INT)                                               AS n_dysphagia,
    ROUND(100.0 * SUM(has_dysphagia::INT) / COUNT(has_dysphagia), 1)     AS pct_dysphagia,
    COUNT(has_gtube)                                                      AS n_gtube_eligible,
    SUM(has_gtube::INT)                                                   AS n_gtube,
    ROUND(100.0 * SUM(has_gtube::INT) / COUNT(has_gtube), 1)             AS pct_gtube,
    COUNT(has_tracheostomy)                                               AS n_trach_eligible,
    SUM(has_tracheostomy::INT)                                            AS n_trach,
    ROUND(100.0 * SUM(has_tracheostomy::INT) / COUNT(has_tracheostomy), 1) AS pct_trach
FROM opscc_outcomes
GROUP BY tx_group
ORDER BY tx_group;
