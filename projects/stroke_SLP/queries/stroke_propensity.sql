-- stroke_propensity.sql
-- Assembles one row per patient with SLP timing exposure, demographics,
-- stroke characteristics, and comorbidity covariates for PSM.
--
-- Cohort: Home-discharged stroke patients whose FIRST SLP contact post-discharge
--   was in an outpatient/clinic setting (carrier or outpatient facility).
--   Patients whose first SLP was through HHA are excluded (see stroke_slp.first_slp_is_clinic).
--
-- SLP exposure (slp_timing_group): based on first CLINIC SLP contact
--   'Wk0'   = days  1-7  (retained with flag; excluded from analysis — discharge-transitional)
--   'Early' = first clinic SLP days  8-35 post-discharge (treated)
--   'Late'  = first clinic SLP days 36-90 post-discharge (reference)
--
-- Single PSM comparison (run by stroke_psm.py):
--   Comparison A: Early (8-35d) vs Late (36-90d)  → psm_matched_A / psm_match_id_A
--
-- PSM covariates:
--   age_at_adm, sex, race
--   stroke_type (SAH / ICH / Ischemic / Unspecified)
--   dysphagia_poa, aspiration_poa
--   adm_source, index_los (severity proxies)
--   adm_year (secular trends)
--   van_walraven_score + individual comorbidity flags
--   dschg_group (Home vs Home+HHA) — covariate, not exact match
--   rucc_group (Metro/Nonmetro/Rural) — geographic access covariate
--   dual_eligible (Medicare+Medicaid) — SES proxy
--
-- Output table: stroke_propensity (PSM columns populated by stroke_psm.py)

SET memory_limit='24GB';
SET threads=12;
-- temp_directory set by run_pipeline.py via SET temp_directory

-- ── Readmission timing: first inpatient readmission within 90d of discharge ───
-- Used to flag patients whose first inpatient readmission preceded their SLP visit.
-- readmit_before_slp = TRUE patients are retained in the primary analysis.
-- The flag is used only for sensitivity analyses because excluding these rows
-- creates differential healthy-survivor bias, especially in the Late group.
CREATE OR REPLACE TEMP TABLE _readmit_timing AS
SELECT
    c.DSYSRTKY,
    MIN(DATEDIFF('day', c.index_dschg_date,
                 TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d'))) AS days_to_first_readmit
FROM stroke_cohort c
JOIN inp_claimsk_all i ON i.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') > c.index_dschg_date
  AND DATEDIFF('day', c.index_dschg_date,
               TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')) BETWEEN 1 AND 90
GROUP BY c.DSYSRTKY;

-- ── Geography (RUCC) and SES (dual eligibility) from MBSF ────────────────────
-- rucc_group: county-level rural/urban classification (RUCC 2023, July county FIPS)
-- dual_eligible: Medicare+Medicaid dual enrollment in the admission month
CREATE OR REPLACE TEMP TABLE _geo_ses AS
SELECT
    c.DSYSRTKY,
    CASE
        WHEN rl.rucc BETWEEN 1 AND 3 THEN 'Metro'
        WHEN rl.rucc BETWEEN 4 AND 6 THEN 'Nonmetro'
        WHEN rl.rucc BETWEEN 7 AND 9 THEN 'Rural'
        ELSE 'Unknown'
    END AS rucc_group,
    CASE
        WHEN CASE MONTH(c.index_adm_date)
                 WHEN 1  THEN m.DUAL_01 WHEN 2  THEN m.DUAL_02
                 WHEN 3  THEN m.DUAL_03 WHEN 4  THEN m.DUAL_04
                 WHEN 5  THEN m.DUAL_05 WHEN 6  THEN m.DUAL_06
                 WHEN 7  THEN m.DUAL_07 WHEN 8  THEN m.DUAL_08
                 WHEN 9  THEN m.DUAL_09 WHEN 10 THEN m.DUAL_10
                 WHEN 11 THEN m.DUAL_11 WHEN 12 THEN m.DUAL_12
             END IN ('01','02','03','04','05','06','08','09') THEN 1
        WHEN CASE MONTH(c.index_adm_date)
                 WHEN 1  THEN m.DUAL_01 WHEN 2  THEN m.DUAL_02
                 WHEN 3  THEN m.DUAL_03 WHEN 4  THEN m.DUAL_04
                 WHEN 5  THEN m.DUAL_05 WHEN 6  THEN m.DUAL_06
                 WHEN 7  THEN m.DUAL_07 WHEN 8  THEN m.DUAL_08
                 WHEN 9  THEN m.DUAL_09 WHEN 10 THEN m.DUAL_10
                 WHEN 11 THEN m.DUAL_11 WHEN 12 THEN m.DUAL_12
             END = 'NA' THEN 0
        ELSE NULL  -- '00'=not enrolled that month, '99'=unknown
    END AS dual_eligible
FROM stroke_cohort c
JOIN mbsf_all m
    ON  m.DSYSRTKY = c.DSYSRTKY
    AND m.RFRNC_YR = CAST(YEAR(c.index_adm_date) AS VARCHAR)
LEFT JOIN rucc_lookup rl ON rl.FIPS = m.STATE_CNTY_FIPS_CD_07;

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
    LEAST(1, COALESCE(s.slp_outpt_0_14d, 0) + COALESCE(s.slp_outpt_15_30d, 0) + COALESCE(s.slp_outpt_31_90d, 0)) AS slp_outpt_any_90d,

    -- Primary exposure: week-anchored timing (Wk0 retained with flag; Early/Late = analytic groups)
    CASE
        WHEN s.days_to_slp_outpt BETWEEN  1 AND  7 THEN 'Wk0'    -- discharge-transitional, excluded from primary
        WHEN s.days_to_slp_outpt BETWEEN  8 AND 35 THEN 'Early'
        WHEN s.days_to_slp_outpt BETWEEN 36 AND 90 THEN 'Late'
        ELSE 'No SLP'
    END AS slp_timing_group,

    -- Readmission flag: TRUE if first inpatient readmission preceded first SLP visit
    r.days_to_first_readmit,
    CASE WHEN r.days_to_first_readmit IS NOT NULL
          AND r.days_to_first_readmit < s.days_to_slp_outpt
         THEN TRUE ELSE FALSE END AS readmit_before_slp,

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
    e.dementia,
    e.prior_stroke,
    e.afib,
    e.prior_tia,
    e.hypertension,
    e.dyslipid,
    e.smoking,
    -- Geography and SES
    COALESCE(g.rucc_group,    'Unknown') AS rucc_group,
    g.dual_eligible,

    -- PSM match flags — populated by stroke_psm.py
    -- Comparison A: Early (8-35d) vs Late (36-90d)
    FALSE     AS psm_matched_A,
    NULL::VARCHAR AS psm_match_id_A,
    NULL::DOUBLE  AS prop_score_A

FROM stroke_cohort c
LEFT JOIN stroke_slp         s ON s.DSYSRTKY = c.DSYSRTKY
LEFT JOIN stroke_comorbidity e ON e.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _readmit_timing    r ON r.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _geo_ses           g ON g.DSYSRTKY = c.DSYSRTKY
WHERE s.first_slp_is_clinic = TRUE
  AND s.days_to_slp_outpt BETWEEN 1 AND 90  -- exclude day-0; Wk0 retained with flag
  AND COALESCE(c.dysphagia_poa,  0) = 0   -- exclude pre-existing dysphagia
  AND COALESCE(c.aspiration_poa, 0) = 0   -- exclude pre-existing aspiration
  AND COALESCE(c.peg_placed,     0) = 0   -- exclude index PEG placement
  AND COALESCE(c.trach_placed,   0) = 0;  -- exclude index tracheostomy

-- ── Summary: covariate balance check (pre-PSM) ────────────────────────────────
-- Primary analytic cohort: slp_timing_group IN ('Early','Late')
-- Sensitivity cohort: add readmit_before_slp = FALSE
SELECT
    slp_timing_group,
    SUM(CASE WHEN readmit_before_slp THEN 1 ELSE 0 END)  AS n_readmit_excl,
    COUNT(*) - SUM(CASE WHEN readmit_before_slp THEN 1 ELSE 0 END) AS n_analytic,
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
