-- stroke_propensity.sql
-- Assembles one row per patient with SLP timing exposure, demographics,
-- stroke characteristics, and comorbidity covariates for PSM.
--
-- Cohort: Home-discharged stroke patients whose FIRST SLP contact post-discharge
--   was in an outpatient/clinic setting (carrier or outpatient facility).
--   Patients whose first SLP was through HHA are excluded (see stroke_slp.first_slp_is_clinic).
--
-- SLP exposure (slp_timing_group): based on first CLINIC SLP contact
--   '0-14d'  = first clinic SLP within  0-14 days of discharge
--   '15-30d' = first clinic SLP within 15-30 days of discharge
--   '31-90d' = first clinic SLP within 31-90 days of discharge (reference)
--
-- Two pairwise PSM comparisons (run by stroke_psm.py):
--   Comparison A: 0-14d  vs 31-90d  → psm_matched_A / psm_match_id_A
--   Comparison B: 15-30d vs 31-90d  → psm_matched_B / psm_match_id_B
--
-- PSM covariates:
--   age_at_adm, sex, race
--   stroke_type (SAH / ICH / Ischemic / Unspecified)
--   dysphagia_poa, aspiration_poa
--   adm_source, index_los (severity proxies)
--   adm_year (secular trends)
--   van_walraven_score + individual comorbidity flags
--   dschg_group (Home vs Home+HHA) — covariate, not exact match
--
-- Output table: stroke_propensity (PSM columns populated by stroke_psm.py)

SET memory_limit='24GB';
SET threads=12;
-- temp_directory set by run_pipeline.py via SET temp_directory

DROP TABLE IF EXISTS stroke_propensity;

CREATE TABLE stroke_propensity AS

SELECT
    c.DSYSRTKY,
    c.index_adm_date,
    c.index_dschg_date,
    c.index_los,
    c.stroke_type,
    c.dschg_status,
    CASE c.dschg_status
        WHEN '01' THEN 'Home'
        WHEN '07' THEN 'Home'
        WHEN '06' THEN 'Home+HHA'
        WHEN '03' THEN 'SNF'
        WHEN '64' THEN 'SNF'
        WHEN '62' THEN 'IRF'
        WHEN '63' THEN 'LTACH'
        ELSE       'Other'
    END AS dschg_group,
    c.drg_cd,
    c.adm_source,
    c.index_pmt,
    c.index_chrg,
    c.dysphagia_poa,
    c.aspiration_poa,
    c.mech_vent,
    c.peg_placed,
    c.trach_placed,
    c.age_at_adm,
    CASE
        WHEN c.age_at_adm < 70              THEN '<70'
        WHEN c.age_at_adm BETWEEN 70 AND 74 THEN '70-74'
        WHEN c.age_at_adm BETWEEN 75 AND 79 THEN '75-79'
        WHEN c.age_at_adm BETWEEN 80 AND 84 THEN '80-84'
        ELSE '85+'
    END AS age_group,
    c.sex,
    c.race,
    YEAR(c.index_adm_date) AS adm_year,

    -- SLP timing (clinic-only: carrier + outpatient facility)
    -- Cohort restricted to patients whose first SLP was in a clinic setting.
    s.days_to_slp_outpt,
    s.first_slp_is_clinic,
    COALESCE(s.slp_outpt_0_14d,  0) AS slp_outpt_0_14d,
    COALESCE(s.slp_outpt_15_30d, 0) AS slp_outpt_15_30d,
    COALESCE(s.slp_outpt_31_90d, 0) AS slp_outpt_31_90d,

    -- Primary exposure: 4-level timing group
    CASE
        WHEN s.days_to_slp_outpt BETWEEN  0 AND 14 THEN '0-14d'
        WHEN s.days_to_slp_outpt BETWEEN 15 AND 30 THEN '15-30d'
        WHEN s.days_to_slp_outpt BETWEEN 31 AND 90 THEN '31-90d'
        ELSE 'No SLP'
    END AS slp_timing_group,

    -- Comorbidity
    e.van_walraven_score,
    e.chf,
    e.carit,
    e.valv,
    e.pcd,
    e.pvd,
    e.hypunc,
    e.hypc,
    e.para,
    e.ond,
    e.cpd,
    e.diabunc,
    e.diabc,
    e.hypothy,
    e.rf,
    e.ld,
    e.pud,
    e.aids,
    e.lymph,
    e.metacanc,
    e.solidtum,
    e.rheumd,
    e.coag,
    e.obes,
    e.wloss,
    e.fed,
    e.blane,
    e.dane,
    e.alcohol,
    e.drug,
    e.psycho,
    e.depre,
    -- Stroke-specific
    e.prior_stroke,
    e.afib,
    e.prior_tia,
    e.hypertension,
    e.dyslipid,
    e.smoking,
    -- PSM match flags — populated by stroke_psm.py
    -- Comparison A: 0-14d vs 31-90d
    FALSE     AS psm_matched_A,
    NULL::VARCHAR AS psm_match_id_A,
    NULL::DOUBLE  AS prop_score_A,
    -- Comparison B: 15-30d vs 31-90d
    FALSE     AS psm_matched_B,
    NULL::VARCHAR AS psm_match_id_B,
    NULL::DOUBLE  AS prop_score_B

FROM stroke_cohort c
LEFT JOIN stroke_slp         s ON s.DSYSRTKY = c.DSYSRTKY
LEFT JOIN stroke_comorbidity e ON e.DSYSRTKY = c.DSYSRTKY
WHERE s.first_slp_is_clinic = TRUE;  -- include only patients whose first SLP was outpatient/clinic

-- ── Summary: covariate balance check (pre-PSM) ────────────────────────────────
SELECT
    slp_timing_group,
    COUNT(*)                                              AS n,
    ROUND(AVG(age_at_adm), 1)                            AS mean_age,
    ROUND(100.0 * SUM(CASE WHEN sex='Male'       THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_male,
    ROUND(100.0 * SUM(CASE WHEN race='White'     THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_white,
    ROUND(AVG(van_walraven_score), 2)                    AS mean_vw,
    ROUND(100.0 * SUM(dysphagia_poa)  / COUNT(*), 1)    AS pct_dysphagia_poa,
    ROUND(100.0 * SUM(aspiration_poa) / COUNT(*), 1)    AS pct_aspiration_poa,
    ROUND(100.0 * SUM(mech_vent)      / COUNT(*), 1)    AS pct_mech_vent,
    ROUND(100.0 * SUM(peg_placed)     / COUNT(*), 1)    AS pct_peg_placed,
    ROUND(100.0 * SUM(trach_placed)   / COUNT(*), 1)    AS pct_trach_placed,
    ROUND(100.0 * SUM(CASE WHEN stroke_type='Ischemic'    THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_ischemic,
    ROUND(100.0 * SUM(CASE WHEN stroke_type='ICH'         THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_ich,
    ROUND(100.0 * SUM(CASE WHEN stroke_type='SAH'         THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_sah,
    ROUND(AVG(index_los), 1)                             AS mean_los,
    ROUND(AVG(afib), 3)                                  AS pct_afib,
    ROUND(AVG(hypertension), 3)                          AS pct_htn
FROM stroke_propensity
GROUP BY slp_timing_group
ORDER BY slp_timing_group;
