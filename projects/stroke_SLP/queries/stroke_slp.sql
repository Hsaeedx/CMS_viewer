-- stroke_slp.sql
-- Identifies post-discharge SLP contact for each stroke_cohort patient.
--
-- Primary exposure (original): any SLP service within 30 days of index discharge date.
--
-- Time-varying Cox exposure: all post-discharge SLP settings, within 90 days.
--   Includes: carrier HCPCS codes, outpatient facility rev center 044x,
--             HHA rev center 044x, SNF rev center 044x.
--   Hospice patients are already excluded from stroke_cohort (dschg_status 50/51).
--
-- SLP signals:
--   A) Carrier claims: specific SLP HCPCS codes (extended to 90 days)
--   B) Outpatient facility: revenue center 0440-0449 (extended to 90 days)
--   C) HHA claims: revenue center 0440-0449 (extended to 90 days)
--   D) SNF claims: revenue center 0440-0449 (extended to 90 days)
--
-- SLP HCPCS codes:
--   92507        Speech-language treatment (individual)
--   92521-92523  Speech/language evaluation
--   92526        Oral function therapy
--   92610        Oral/pharyngeal swallowing function evaluation
--   92611-92612  Flexible endoscopic swallowing evaluation (FEES)
--   92616-92617  FEES with therapeutic intervention
--   97129-97130  Therapeutic interventions (cognitive/communication)
--
-- Revenue center 0440-0449 = speech-language pathology (facility)
--
-- Output table: stroke_slp

SET memory_limit='24GB';
SET threads=12;
-- temp_directory set by run_pipeline.py via SET temp_directory

-- ── Step 1: Carrier claims with SLP HCPCS codes ───────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_car AS
SELECT
    cl.DSYSRTKY,
    TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS svc_date,
    cl.HCPCS_CD,
    DATEDIFF('day', c.index_dschg_date,
             TRY_STRPTIME(cl.THRU_DT, '%Y%m%d')) AS days_from_dschg,
    CASE
        WHEN cl.HCPCS_CD IN ('92521','92522','92523')         THEN 'eval'
        WHEN cl.HCPCS_CD IN ('92507','92526','97129','97130') THEN 'treatment'
        WHEN cl.HCPCS_CD IN ('92610','92611','92612',
                              '92616','92617')                 THEN 'swallow'
        ELSE 'other_slp'
    END AS slp_type
FROM stroke_cohort c
JOIN car_linek_all cl ON cl.DSYSRTKY = c.DSYSRTKY
WHERE cl.HCPCS_CD IN ('92507','92521','92522','92523','92526',
                       '92610','92611','92612','92616','92617',
                       '97129','97130')
  AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d')
          BETWEEN c.index_dschg_date
              AND c.index_dschg_date + INTERVAL 90 DAY;

-- ── Step 2: Outpatient revenue center 0440–0449 ───────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_out AS
SELECT
    r.DSYSRTKY,
    TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS svc_date,
    r.HCPCS_CD,
    DATEDIFF('day', c.index_dschg_date,
             TRY_STRPTIME(r.THRU_DT, '%Y%m%d')) AS days_from_dschg,
    'rev_center_out' AS slp_type
FROM stroke_cohort c
JOIN out_revenuek_all r ON r.DSYSRTKY = c.DSYSRTKY
WHERE r.REV_CNTR BETWEEN '0440' AND '0449'
  AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
          BETWEEN c.index_dschg_date
              AND c.index_dschg_date + INTERVAL 90 DAY;

-- ── Step 3: HHA revenue center 0440–0449 ─────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_hha AS
SELECT
    r.DSYSRTKY,
    TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS svc_date,
    r.HCPCS_CD,
    DATEDIFF('day', c.index_dschg_date,
             TRY_STRPTIME(r.THRU_DT, '%Y%m%d')) AS days_from_dschg,
    'rev_center_hha' AS slp_type
FROM stroke_cohort c
JOIN (
    SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2016
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2017
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2018
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2019
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2020
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2021
    UNION ALL SELECT DSYSRTKY, THRU_DT, REV_CNTR, HCPCS_CD FROM HHA_revenuek_2022
) r ON r.DSYSRTKY = c.DSYSRTKY
WHERE r.REV_CNTR BETWEEN '0440' AND '0449'
  AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
          BETWEEN c.index_dschg_date
              AND c.index_dschg_date + INTERVAL 90 DAY;

-- ── Step 4: SNF revenue center 0440–0449 ─────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_snf AS
SELECT
    r.DSYSRTKY,
    TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS svc_date,
    r.HCPCS_CD,
    DATEDIFF('day', c.index_dschg_date,
             TRY_STRPTIME(r.THRU_DT, '%Y%m%d')) AS days_from_dschg,
    'rev_center_snf' AS slp_type
FROM stroke_cohort c
JOIN snf_revenuek_all r ON r.DSYSRTKY = c.DSYSRTKY
WHERE r.REV_CNTR BETWEEN '0440' AND '0449'
  AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
          BETWEEN c.index_dschg_date
              AND c.index_dschg_date + INTERVAL 90 DAY;

-- ── Step 5: Combine all sources ───────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _slp_all AS
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_car
UNION ALL
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_out
UNION ALL
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_hha
UNION ALL
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_snf;

-- ── Step 5b: Clinic-only SLP (carrier + outpatient facility) ─────────────────
-- Used for slp_timing_group assignment and inclusion criterion.
-- Cohort is restricted to home-discharged patients whose first SLP contact
-- was in an outpatient/clinic setting (not HHA). This ensures both comparison
-- groups are ambulatory enough to access clinic-based care.

CREATE OR REPLACE TEMP TABLE _slp_outpt_only AS
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_car
UNION ALL
SELECT DSYSRTKY, svc_date, days_from_dschg, slp_type FROM _slp_out;

-- ── Final: Build stroke_slp ───────────────────────────────────────────────────

DROP TABLE IF EXISTS stroke_slp;

CREATE TABLE stroke_slp AS
SELECT
    c.DSYSRTKY,

    -- Primary exposure: any SLP within 30 days of discharge
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30  THEN 1 ELSE 0 END) AS slp_any_30d,

    -- Setting flags
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type = 'rev_center_hha'  THEN 1 ELSE 0 END) AS slp_hha,
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type = 'rev_center_snf'  THEN 1 ELSE 0 END) AS slp_snf,
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type IN ('rev_center_out','eval','treatment','swallow','other_slp')
              AND s.slp_type != 'rev_center_hha'
              AND s.slp_type != 'rev_center_snf' THEN 1 ELSE 0 END) AS slp_outpt,

    -- Service type flags (within 30 days of discharge)
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type = 'eval'             THEN 1 ELSE 0 END) AS slp_eval,
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type = 'swallow'          THEN 1 ELSE 0 END) AS slp_swallow,
    MAX(CASE WHEN s.days_from_dschg BETWEEN 0 AND 30
              AND s.slp_type = 'treatment'        THEN 1 ELSE 0 END) AS slp_tx,

    -- First SLP contact post-discharge (any source)
    MIN(CASE WHEN s.days_from_dschg >= 0 THEN s.svc_date     END) AS first_slp_date,
    MIN(CASE WHEN s.days_from_dschg >= 0 THEN s.days_from_dschg END) AS days_to_slp,

    -- ── All-setting SLP (time-varying Cox timing exposure) ────────────────────
    -- All post-discharge settings: carrier, outpatient facility, SNF, HHA.
    -- Intragroup comparisons are ensured by PSM exact matching on dschg_group.
    MAX(CASE WHEN o.days_from_dschg BETWEEN 0 AND 90 THEN 1 ELSE 0 END) AS slp_outpt_any_90d,
    MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END)    AS days_to_slp_outpt,

    -- Timing group flags (mutually exclusive: first clinic SLP contact)
    CASE
        WHEN MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END) BETWEEN 0  AND 14 THEN 1
        ELSE 0
    END AS slp_outpt_0_14d,
    CASE
        WHEN MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END) BETWEEN 15 AND 30 THEN 1
        ELSE 0
    END AS slp_outpt_15_30d,
    CASE
        WHEN MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END) BETWEEN 31 AND 90 THEN 1
        ELSE 0
    END AS slp_outpt_31_90d,

    -- Inclusion criterion: first SLP contact of any type must be clinic-based (not HHA/SNF).
    -- TRUE  = first SLP was outpatient/carrier (eligible)
    -- FALSE = first SLP was HHA or no clinic SLP exists within 90d (excluded in propensity step)
    CASE
        WHEN MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END) IS NOT NULL
         AND (
               -- No HHA/SNF SLP at all
               MIN(CASE WHEN s.days_from_dschg >= 0
                         AND s.slp_type IN ('rev_center_hha', 'rev_center_snf')
                        THEN s.days_from_dschg END) IS NULL
               OR
               -- Clinic SLP is on same day or earlier than first HHA/SNF SLP
               MIN(CASE WHEN o.days_from_dschg >= 0 THEN o.days_from_dschg END)
                   <= MIN(CASE WHEN s.days_from_dschg >= 0
                               AND s.slp_type IN ('rev_center_hha', 'rev_center_snf')
                              THEN s.days_from_dschg END)
             )
        THEN TRUE ELSE FALSE
    END AS first_slp_is_clinic

FROM stroke_cohort c
LEFT JOIN _slp_all      s ON s.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _slp_outpt_only o ON o.DSYSRTKY = c.DSYSRTKY
GROUP BY c.DSYSRTKY;

-- ── Summary ───────────────────────────────────────────────────────────────────
SELECT
    SUM(slp_any_30d)      AS n_slp_30d,
    SUM(slp_hha)          AS n_slp_hha,
    SUM(slp_snf)          AS n_slp_snf,
    SUM(slp_outpt)        AS n_slp_outpt,
    SUM(slp_eval)         AS n_slp_eval,
    SUM(slp_swallow)      AS n_slp_swallow,
    SUM(slp_tx)           AS n_slp_tx,
    -- Clinic-only timing groups (time-varying Cox)
    SUM(slp_outpt_any_90d)    AS n_slp_clinic_90d,
    SUM(slp_outpt_0_14d)      AS n_slp_0_14d,
    SUM(slp_outpt_15_30d)     AS n_slp_15_30d,
    SUM(slp_outpt_31_90d)     AS n_slp_31_90d,
    SUM(first_slp_is_clinic::INT) AS n_first_slp_clinic,
    COUNT(*)                  AS n_total,
    ROUND(100.0 * SUM(slp_any_30d)           / COUNT(*), 1) AS pct_slp_30d,
    ROUND(100.0 * SUM(slp_outpt_any_90d)     / COUNT(*), 1) AS pct_slp_clinic_90d,
    ROUND(100.0 * SUM(first_slp_is_clinic::INT) / COUNT(*), 1) AS pct_first_slp_clinic
FROM stroke_slp;
