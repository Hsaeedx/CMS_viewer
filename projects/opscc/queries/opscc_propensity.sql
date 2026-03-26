-- Build opscc_propensity: one row per patient with treatment group,
-- demographics (age at dx, race, sex), and Elixhauser comorbidity score.
-- Excludes patients with metastatic dx within ±90 days of first_hnc_date.
-- Treatment window: 12 months from first_hnc_date.
--
-- Treatment groups (mutually exclusive):
--   TORS alone  : TORS within 12mo; no chemo; no RT
--   RT alone    : RT within 12mo;   no chemo; no TORS
--   TORS + RT   : TORS + RT within 12mo; no chemo
--   CT/CRT      : chemo + RT within 12mo; no TORS
--   Other       : all remaining combinations (excluded from PSM analyses)

DROP TABLE IF EXISTS opscc_propensity;

CREATE TABLE opscc_propensity AS

WITH demo AS (
    -- One mbsf row per patient: join on year of diagnosis for correct age/race/sex/state
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
        END AS race,
        CASE m.STATE_CD
            WHEN '07' THEN 'Northeast' WHEN '09' THEN 'Northeast'
            WHEN '20' THEN 'Northeast' WHEN '22' THEN 'Northeast'
            WHEN '30' THEN 'Northeast' WHEN '31' THEN 'Northeast'
            WHEN '33' THEN 'Northeast' WHEN '39' THEN 'Northeast'
            WHEN '41' THEN 'Northeast' WHEN '47' THEN 'Northeast'
            WHEN '01' THEN 'South' WHEN '04' THEN 'South'
            WHEN '08' THEN 'South' WHEN '10' THEN 'South'
            WHEN '11' THEN 'South' WHEN '18' THEN 'South'
            WHEN '19' THEN 'South' WHEN '21' THEN 'South'
            WHEN '25' THEN 'South' WHEN '34' THEN 'South'
            WHEN '37' THEN 'South' WHEN '42' THEN 'South'
            WHEN '44' THEN 'South' WHEN '45' THEN 'South'
            WHEN '49' THEN 'South' WHEN '51' THEN 'South'
            WHEN '14' THEN 'Midwest' WHEN '15' THEN 'Midwest'
            WHEN '16' THEN 'Midwest' WHEN '17' THEN 'Midwest'
            WHEN '23' THEN 'Midwest' WHEN '24' THEN 'Midwest'
            WHEN '26' THEN 'Midwest' WHEN '28' THEN 'Midwest'
            WHEN '35' THEN 'Midwest' WHEN '36' THEN 'Midwest'
            WHEN '43' THEN 'Midwest' WHEN '52' THEN 'Midwest'
            WHEN '02' THEN 'West' WHEN '03' THEN 'West'
            WHEN '05' THEN 'West' WHEN '06' THEN 'West'
            WHEN '12' THEN 'West' WHEN '13' THEN 'West'
            WHEN '27' THEN 'West' WHEN '29' THEN 'West'
            WHEN '32' THEN 'West' WHEN '38' THEN 'West'
            WHEN '46' THEN 'West' WHEN '50' THEN 'West'
            WHEN '53' THEN 'West'
            ELSE 'Other'
        END AS census_region
    FROM mbsf_all m
    JOIN opscc_cohort o ON m.DSYSRTKY = o.DSYSRTKY
      AND CAST(m.RFRNC_YR AS INTEGER) = YEAR(o.first_hnc_date)
),

subsite AS (
    -- Modal primary site code for each patient (C01/C09/C10/C14)
    SELECT DSYSRTKY, MODE(dx_prefix) AS subsite
    FROM hnc_dx_raw
    WHERE dx_prefix IN ('C01', 'C09', 'C10', 'C14')
    GROUP BY DSYSRTKY
),

tx AS (
    -- Derive boolean treatment flags within the 12-month window
    SELECT
        DSYSRTKY,
        (first_tors_date  IS NOT NULL AND first_tors_date  <= first_hnc_date + INTERVAL 12 MONTH) AS has_tors,
        (first_chemo_date IS NOT NULL AND first_chemo_date <= first_hnc_date + INTERVAL 12 MONTH) AS has_chemo,
        (first_rt_date    IS NOT NULL AND first_rt_date    <= first_hnc_date + INTERVAL 12 MONTH) AS has_rt,
        first_tors_date,
        first_chemo_date,
        first_rt_date
    FROM opscc_cohort
)

SELECT
    o.DSYSRTKY,
    o.first_hnc_date,

    -- Treatment group
    CASE
        WHEN t.has_tors  AND NOT t.has_chemo AND NOT t.has_rt  THEN 'TORS alone'
        WHEN t.has_rt    AND NOT t.has_chemo AND NOT t.has_tors THEN 'RT alone'
        WHEN t.has_tors  AND t.has_rt        AND NOT t.has_chemo THEN 'TORS + RT'
        WHEN t.has_chemo AND t.has_rt        AND NOT t.has_tors  THEN 'CT/CRT'
        ELSE 'Other'
    END AS tx_group,

    -- First treatment date (time origin for survival / outcomes)
    CASE
        WHEN t.has_tors  AND NOT t.has_chemo AND NOT t.has_rt  THEN t.first_tors_date
        WHEN t.has_rt    AND NOT t.has_chemo AND NOT t.has_tors THEN t.first_rt_date
        WHEN t.has_tors  AND t.has_rt        AND NOT t.has_chemo THEN t.first_tors_date
        WHEN t.has_chemo AND t.has_rt        AND NOT t.has_tors
            THEN LEAST(t.first_chemo_date, t.first_rt_date)
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
    d.census_region,
    YEAR(o.first_hnc_date)              AS dx_year,
    COALESCE(s.subsite, 'C14')          AS subsite,

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
JOIN tx t ON o.DSYSRTKY = t.DSYSRTKY
JOIN demo d ON o.DSYSRTKY = d.DSYSRTKY
LEFT JOIN subsite s ON o.DSYSRTKY = s.DSYSRTKY
LEFT JOIN opscc_comorbidity e ON o.DSYSRTKY = e.DSYSRTKY
WHERE o.has_metastatic_dx = FALSE
  AND o.first_hnc_date < DATE '2023-07-01';
