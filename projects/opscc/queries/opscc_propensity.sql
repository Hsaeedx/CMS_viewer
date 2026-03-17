-- Build opscc_propensity: one row per patient with treatment group,
-- demographics (age at dx, race, sex), and Elixhauser comorbidity score.
-- Excludes patients with metastatic dx within ±90 days of first_hnc_date.
-- Treatment window: 12 months from first_hnc_date.

DROP TABLE IF EXISTS opscc_propensity;

CREATE TABLE opscc_propensity AS

WITH demo AS (
    -- One mbsf row per patient: join on year of diagnosis for correct age
    SELECT
        m.DSYSRTKY,
        CAST(m.AGE AS INTEGER)  AS age_at_dx,
        CASE m.SEX
            WHEN '1' THEN 'Male'
            WHEN '2' THEN 'Female'
            ELSE 'Unknown'
        END AS sex,
        CASE m.RACE
            WHEN '1' THEN 'White'
            WHEN '2' THEN 'Black'
            WHEN '4' THEN 'Asian/PI'
            WHEN '5' THEN 'Hispanic'
            WHEN '6' THEN 'AIAN'
            WHEN '3' THEN 'Other'
            ELSE 'Unknown'
        END AS race
    FROM mbsf_all m
    JOIN opscc_cohort o ON m.DSYSRTKY = o.DSYSRTKY
      AND CAST(m.RFRNC_YR AS INTEGER) = YEAR(o.first_hnc_date)
)

SELECT
    o.DSYSRTKY,
    o.first_hnc_date,

    -- Treatment group (12-month window)
    CASE
        WHEN (o.first_tors_date  IS NOT NULL AND o.first_tors_date  <= o.first_hnc_date + INTERVAL 12 MONTH)
         AND (o.first_ctcrt_date IS NOT NULL AND o.first_ctcrt_date <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN 'Both'
        WHEN (o.first_tors_date  IS NOT NULL AND o.first_tors_date  <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN 'TORS only'
        WHEN (o.first_ctcrt_date IS NOT NULL AND o.first_ctcrt_date <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN 'CT/CRT only'
        ELSE 'Neither'
    END AS tx_group,

    -- First treatment date (whichever came first within window)
    CASE
        WHEN (o.first_tors_date  IS NOT NULL AND o.first_tors_date  <= o.first_hnc_date + INTERVAL 12 MONTH)
         AND (o.first_ctcrt_date IS NOT NULL AND o.first_ctcrt_date <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN LEAST(o.first_tors_date, o.first_ctcrt_date)
        WHEN (o.first_tors_date  IS NOT NULL AND o.first_tors_date  <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN o.first_tors_date
        WHEN (o.first_ctcrt_date IS NOT NULL AND o.first_ctcrt_date <= o.first_hnc_date + INTERVAL 12 MONTH)
            THEN o.first_ctcrt_date
        ELSE NULL
    END AS first_tx_date,

    -- Demographics
    d.age_at_dx,
    CASE
        WHEN d.age_at_dx < 65              THEN '<65'
        WHEN d.age_at_dx BETWEEN 65 AND 69 THEN '65-69'
        WHEN d.age_at_dx BETWEEN 70 AND 74 THEN '70-74'
        WHEN d.age_at_dx BETWEEN 75 AND 79 THEN '75-79'
        WHEN d.age_at_dx BETWEEN 80 AND 84 THEN '80-84'
        ELSE '85+'
    END AS age_group,
    d.sex,
    d.race,

    -- Elixhauser comorbidity
    e.van_walraven_score,

    -- Individual comorbidity flags (for propensity model covariates)
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
    e.depre

FROM opscc_cohort o
JOIN demo d ON o.DSYSRTKY = d.DSYSRTKY
LEFT JOIN opscc_comorbidity e ON o.DSYSRTKY = e.DSYSRTKY
WHERE o.has_metastatic_dx = FALSE;
