-- stroke_propensity.sql
-- Assembles one row per patient with SLP exposure flag, demographics,
-- stroke characteristics, and comorbidity covariates for PSM.
--
-- SLP exposure (slp_group):
--   'SLP'    = slp_any_90d = 1
--   'No SLP' = slp_any_90d = 0
--
-- PSM covariates (do NOT include discharge disposition — it is a mediator):
--   age_at_adm, sex, race
--   stroke_type (SAH / ICH / Ischemic / Unspecified)
--   dysphagia_poa, aspiration_poa
--   adm_source, index_los (severity proxies)
--   adm_year (secular trends)
--   van_walraven_score + individual comorbidity flags
--
-- Output table: stroke_propensity (with psm_matched flag to be added by Python)

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS stroke_propensity;

CREATE TABLE stroke_propensity AS

SELECT
    c.DSYSRTKY,
    c.index_adm_date,
    c.index_dschg_date,
    c.index_los,
    c.stroke_type,
    c.dschg_status,
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

    -- SLP exposure (all sources, 30 days post-discharge — original analysis)
    COALESCE(s.slp_any_30d, 0) AS slp_any_30d,
    COALESCE(s.slp_hha,     0) AS slp_hha,
    COALESCE(s.slp_snf,     0) AS slp_snf,
    COALESCE(s.slp_outpt,   0) AS slp_outpt,
    COALESCE(s.slp_eval,    0) AS slp_eval,
    COALESCE(s.slp_swallow, 0) AS slp_swallow,
    COALESCE(s.slp_tx,      0) AS slp_tx,
    s.first_slp_date,
    s.days_to_slp,

    -- Outpatient-only SLP (landmark timing analysis — carrier + outpatient, NOT SNF/HHA)
    COALESCE(s.slp_outpt_any_90d, 0) AS slp_outpt_any_90d,
    s.days_to_slp_outpt,
    COALESCE(s.slp_outpt_0_14d,  0) AS slp_outpt_0_14d,
    COALESCE(s.slp_outpt_15_30d, 0) AS slp_outpt_15_30d,
    COALESCE(s.slp_outpt_31_90d, 0) AS slp_outpt_31_90d,

    -- Primary exposure group label (original analysis)
    CASE WHEN COALESCE(s.slp_any_30d, 0) = 1 THEN 'SLP' ELSE 'No SLP' END AS slp_group,

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
    e.dementia,

    -- PSM match flag and propensity score — to be populated by Python PSM script
    FALSE AS psm_matched,
    NULL::VARCHAR AS psm_match_id,
    NULL::DOUBLE AS prop_score

FROM stroke_cohort c
LEFT JOIN stroke_slp         s ON s.DSYSRTKY = c.DSYSRTKY
LEFT JOIN stroke_comorbidity e ON e.DSYSRTKY = c.DSYSRTKY;

-- ── Summary: covariate balance check (pre-PSM) ────────────────────────────────
SELECT
    slp_group,
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
GROUP BY slp_group
ORDER BY slp_group;
