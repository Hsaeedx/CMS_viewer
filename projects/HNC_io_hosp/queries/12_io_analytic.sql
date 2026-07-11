-- Final analytic dataset: one row per patient
-- Joins io_cohort + io_comorbidity + io_outcomes
-- Derives all analysis-ready variables
-- Output: io_analytic

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_analytic;

CREATE TABLE io_analytic AS

WITH fips_cte AS (
    -- Pull FIPS code for the month of death from MBSF (death-year row only)
    SELECT m.DSYSRTKY,
        CASE MONTH(TRY_STRPTIME(m.DEATH_DT,'%Y%m%d'))
            WHEN 1  THEN m.STATE_CNTY_FIPS_CD_01
            WHEN 2  THEN m.STATE_CNTY_FIPS_CD_02
            WHEN 3  THEN m.STATE_CNTY_FIPS_CD_03
            WHEN 4  THEN m.STATE_CNTY_FIPS_CD_04
            WHEN 5  THEN m.STATE_CNTY_FIPS_CD_05
            WHEN 6  THEN m.STATE_CNTY_FIPS_CD_06
            WHEN 7  THEN m.STATE_CNTY_FIPS_CD_07
            WHEN 8  THEN m.STATE_CNTY_FIPS_CD_08
            WHEN 9  THEN m.STATE_CNTY_FIPS_CD_09
            WHEN 10 THEN m.STATE_CNTY_FIPS_CD_10
            WHEN 11 THEN m.STATE_CNTY_FIPS_CD_11
            WHEN 12 THEN m.STATE_CNTY_FIPS_CD_12
        END AS fips_cd
    FROM mbsf_all m
    JOIN io_cohort c ON m.DSYSRTKY = c.DSYSRTKY
    WHERE CAST(m.RFRNC_YR AS INT) = YEAR(c.death_dt)
),

chemo_io_flag AS (
    -- Platinum within ±21 days of any IO claim = chemo-IO combination
    SELECT DISTINCT io.DSYSRTKY, 1 AS chemo_io
    FROM io_claims_raw io
    JOIN (
        SELECT l.DSYSRTKY, TRY_STRPTIME(l.THRU_DT,'%Y%m%d') AS chemo_date
        FROM io_car_lines l
        JOIN io_cohort c ON l.DSYSRTKY = c.DSYSRTKY
        WHERE l.HCPCS_CD IN ('J9060','J9045')

        UNION ALL

        SELECT r.DSYSRTKY, TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT),'%Y%m%d') AS chemo_date
        FROM io_out_revenue r
        JOIN io_cohort c ON r.DSYSRTKY = c.DSYSRTKY
        WHERE r.HCPCS_CD IN ('J9060','J9045')
    ) chemo ON io.DSYSRTKY = chemo.DSYSRTKY
    WHERE ABS(datediff('day', chemo.chemo_date, io.io_date)) <= 21
),

-- ── Earle-benchmark end-of-life care flags (30-day window before death) ───────
ed_outpatient_dates AS (
    -- Distinct outpatient ED visit dates (rev codes 0450-0459)
    SELECT DISTINCT
        r.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d') AS ed_dt
    FROM io_out_revenue r
    JOIN io_cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE r.REV_CNTR BETWEEN '0450' AND '0459'
      AND TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),
ed_admitted_dates AS (
    -- Inpatient stays admitted via ED (TYPE_ADM = '1' = Emergency)
    -- SRC_ADMS = '7' retired by CMS in 2015; TYPE_ADM = '1' is the correct post-2015 signal
    SELECT DISTINCT
        i.DSYSRTKY,
        TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') AS ed_dt
    FROM io_inp_claims i
    JOIN io_cohort c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE i.TYPE_ADM = '1'
      AND TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),
ed_all_dates AS (
    SELECT DSYSRTKY, ed_dt FROM ed_outpatient_dates
    UNION                                  -- UNION (not UNION ALL) dedupes same-day ED->admit
    SELECT DSYSRTKY, ed_dt FROM ed_admitted_dates
),
ed_counts AS (
    SELECT DSYSRTKY, COUNT(DISTINCT ed_dt) AS n_ed_last_30d
    FROM ed_all_dates
    GROUP BY 1
),
icu_last_30d_cte AS (
    -- Any inpatient revenue line with ICU (020x) or CCU (021x) rev code whose
    -- line THRU_DT falls in the last 30d of life
    SELECT DISTINCT r.DSYSRTKY
    FROM io_inp_revenue r
    JOIN io_cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE (r.REV_CNTR BETWEEN '0200' AND '0209'
        OR r.REV_CNTR BETWEEN '0210' AND '0219')
      AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),
adm_last_30d_cte AS (
    -- Any inpatient admission with ADMSN_DT in the last 30d
    SELECT DISTINCT i.DSYSRTKY
    FROM io_inp_claims i
    JOIN io_cohort c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
)

SELECT
    -- Identity
    c.DSYSRTKY,

    -- Demographics
    c.age_at_death,
    CASE
        WHEN c.age_at_death BETWEEN 66 AND 69 THEN '66-69'
        WHEN c.age_at_death BETWEEN 70 AND 74 THEN '70-74'
        WHEN c.age_at_death BETWEEN 75 AND 79 THEN '75-79'
        WHEN c.age_at_death BETWEEN 80 AND 84 THEN '80-84'
        ELSE '85+'
    END AS age_cat,
    CASE c.SEX WHEN '1' THEN 'Male' WHEN '2' THEN 'Female' ELSE 'Unknown' END AS sex,
    CASE c.RACE
        WHEN '1' THEN 'White'
        WHEN '2' THEN 'Black'
        WHEN '5' THEN 'Hispanic'
        WHEN '4' THEN 'Asian/PI'
        WHEN '6' THEN 'Native American'
        WHEN '3' THEN 'Other'
        ELSE 'Unknown'
    END AS race,
    CASE
        WHEN c.dual_cd_death_month IN ('01','02','03','04','05','06','07','08') THEN 1
        ELSE 0
    END AS dual_eligible,
    c.STATE_CD,
    c.CNTY_CD,
    f.fips_cd,
    r.rucc,
    CASE
        WHEN r.rucc BETWEEN 1 AND 3 THEN 'Metro'
        WHEN r.rucc BETWEEN 4 AND 9 THEN 'Non-metro'
        ELSE NULL
    END AS urban_rural,
    CASE c.STATE_CD
        -- Northeast
        WHEN '07' THEN 'Northeast'  -- CT
        WHEN '20' THEN 'Northeast'  -- ME
        WHEN '22' THEN 'Northeast'  -- MA
        WHEN '30' THEN 'Northeast'  -- NH
        WHEN '31' THEN 'Northeast'  -- NJ
        WHEN '33' THEN 'Northeast'  -- NY
        WHEN '39' THEN 'Northeast'  -- PA
        WHEN '73' THEN 'Northeast'  -- PA (eff. 10/2005)
        WHEN '41' THEN 'Northeast'  -- RI
        WHEN '47' THEN 'Northeast'  -- VT
        -- Midwest
        WHEN '14' THEN 'Midwest'    -- IL
        WHEN '15' THEN 'Midwest'    -- IN
        WHEN '16' THEN 'Midwest'    -- IA
        WHEN '17' THEN 'Midwest'    -- KS
        WHEN '70' THEN 'Midwest'    -- KS (eff. 10/2005)
        WHEN '23' THEN 'Midwest'    -- MI
        WHEN '24' THEN 'Midwest'    -- MN
        WHEN '26' THEN 'Midwest'    -- MO
        WHEN '28' THEN 'Midwest'    -- NE
        WHEN '35' THEN 'Midwest'    -- ND
        WHEN '36' THEN 'Midwest'    -- OH
        WHEN '72' THEN 'Midwest'    -- OH (eff. 10/2005)
        WHEN '43' THEN 'Midwest'    -- SD
        WHEN '52' THEN 'Midwest'    -- WI
        -- South
        WHEN '01' THEN 'South'      -- AL
        WHEN '04' THEN 'South'      -- AR
        WHEN '08' THEN 'South'      -- DE
        WHEN '09' THEN 'South'      -- DC
        WHEN '10' THEN 'South'      -- FL
        WHEN '68' THEN 'South'      -- FL (eff. 10/2005)
        WHEN '69' THEN 'South'      -- FL (eff. 10/2005)
        WHEN '11' THEN 'South'      -- GA
        WHEN '18' THEN 'South'      -- KY
        WHEN '19' THEN 'South'      -- LA
        WHEN '71' THEN 'South'      -- LA (eff. 10/2005)
        WHEN '21' THEN 'South'      -- MD
        WHEN '80' THEN 'South'      -- MD (eff. 8/2000)
        WHEN '25' THEN 'South'      -- MS
        WHEN '34' THEN 'South'      -- NC
        WHEN '37' THEN 'South'      -- OK
        WHEN '42' THEN 'South'      -- SC
        WHEN '44' THEN 'South'      -- TN
        WHEN '45' THEN 'South'      -- TX
        WHEN '67' THEN 'South'      -- TX (eff. 10/2005)
        WHEN '74' THEN 'South'      -- TX (eff. 10/2005)
        WHEN '49' THEN 'South'      -- VA
        WHEN '51' THEN 'South'      -- WV
        -- West
        WHEN '02' THEN 'West'       -- AK
        WHEN '03' THEN 'West'       -- AZ
        WHEN '05' THEN 'West'       -- CA
        WHEN '55' THEN 'West'       -- CA (duplicate)
        WHEN '06' THEN 'West'       -- CO
        WHEN '12' THEN 'West'       -- HI
        WHEN '13' THEN 'West'       -- ID
        WHEN '27' THEN 'West'       -- MT
        WHEN '29' THEN 'West'       -- NV
        WHEN '32' THEN 'West'       -- NM
        WHEN '38' THEN 'West'       -- OR
        WHEN '46' THEN 'West'       -- UT
        WHEN '50' THEN 'West'       -- WA
        WHEN '53' THEN 'West'       -- WY
        ELSE NULL
    END AS census_region,

    -- Dates
    c.death_dt,
    YEAR(c.death_dt) AS death_year,
    c.first_hnc_dx_date,
    c.last_io_episode_start,
    c.last_io_episode_end,
    c.last_io_date,
    c.first_io_date,

    -- IO treatment
    c.io_agent,
    c.last_episode_agent,
    c.total_io_episodes,
    c.total_io_doses,
    c.last_episode_doses,
    c.io_duration_days,
    COALESCE(cf.chemo_io, 0) AS chemo_io_flag,
    CASE
        WHEN COALESCE(cf.chemo_io, 0) = 1 THEN 'chemo-IO'
        ELSE 'IO monotherapy'
    END AS io_regimen,

    -- Curative therapy
    c.curative_therapy_date,
    c.primary_curative_type,
    c.had_surgery,
    c.had_radiation,

    -- Subsite
    c.predominant_subsite,
    c.subsite_category,

    -- Derived time variables
    c.days_dx_to_last_io_episode,
    datediff('day', c.last_io_date, c.death_dt) AS days_last_io_to_death,
    CASE
        WHEN datediff('day', c.last_io_date, c.death_dt) <= 3   THEN '<=3 days'
        WHEN datediff('day', c.last_io_date, c.death_dt) <= 14  THEN '4-14 days'
        WHEN datediff('day', c.last_io_date, c.death_dt) <= 30  THEN '15-30 days'
        WHEN datediff('day', c.last_io_date, c.death_dt) <= 90  THEN '31-90 days'
        ELSE '>90 days'
    END AS days_last_io_to_death_cat,

    -- IO timing flags
    (datediff('day', c.last_io_date, c.death_dt) <= 14)::INT AS io_within_14d_of_death,
    (datediff('day', c.last_io_date, c.death_dt) <= 30)::INT AS io_within_30d_of_death,

    -- Earle-benchmark aggressive care flags (30-day window before death)
    COALESCE(ec.n_ed_last_30d, 0)                         AS n_ed_last_30d,
    (COALESCE(ec.n_ed_last_30d, 0) >= 1)::INT             AS any_ed_last_30d,
    (COALESCE(ec.n_ed_last_30d, 0) >= 2)::INT             AS ge_2_ed_last_30d,
    (icu.DSYSRTKY IS NOT NULL)::INT                       AS icu_last_30d,
    (adm.DSYSRTKY IS NOT NULL)::INT                       AS admission_last_30d,

    -- Sensitivity analysis: regimen-adjusted timing
    sv.median_interdose_days,
    sv.estimated_regimen,
    sv.discontinued_threshold_days,
    (datediff('day', c.last_io_date, c.death_dt) <= COALESCE(sv.discontinued_threshold_days, 42))::INT AS io_within_regimen_window,
    -- Days from next expected dose to death (negative = died before next dose was due)
    (datediff('day', c.last_io_date, c.death_dt) - COALESCE(sv.median_interdose_days, 42)) AS days_past_expected_dose_to_death,

    -- Outcomes
    o.hospice_enrolled,
    o.hospice_election_date,
    o.hospice_los_days,
    o.hospice_short_stay,
    o.days_last_io_to_hospice,
    o.in_hospital_death,

    -- Timely hospice enrollment (>3 days before death) - PI-preferred primary outcome
    -- for the multivariable model (mitigates "any enrollment" not capturing lateness)
    (COALESCE(o.hospice_los_days, 0) > 3)::INT AS timely_enrollment,

    -- Time-to-event structure (t0 = last ICI administration; competing risks)
    --   event_type = 1 : hospice election came first (before death)
    --   event_type = 2 : died without hospice election (competing event)
    --   event_type = 0 : censored (shouldn't occur - everyone in cohort died)
    -- time_to_event: days from last ICI date to whichever event occurred first
    CASE
        WHEN o.hospice_enrolled = 1
             AND o.days_last_io_to_hospice IS NOT NULL
             AND o.days_last_io_to_hospice > 0
             AND o.days_last_io_to_hospice <= datediff('day', c.last_io_date, c.death_dt)
            THEN 1
        ELSE 2
    END AS event_type,
    CASE
        WHEN o.hospice_enrolled = 1
             AND o.days_last_io_to_hospice IS NOT NULL
             AND o.days_last_io_to_hospice > 0
             AND o.days_last_io_to_hospice <= datediff('day', c.last_io_date, c.death_dt)
            THEN o.days_last_io_to_hospice
        ELSE datediff('day', c.last_io_date, c.death_dt)
    END AS time_to_event,


    -- Comorbidity
    cm.van_walraven_score,
    CASE
        WHEN cm.van_walraven_score < 0  THEN '<0'
        WHEN cm.van_walraven_score = 0  THEN '0'
        WHEN cm.van_walraven_score <= 4 THEN '1-4'
        ELSE '5+'
    END AS elixhauser_cat,
    -- Individual flags
    cm.chf, cm.carit, cm.valv, cm.pcd, cm.pvd,
    cm.hypunc, cm.hypc, cm.para, cm.ond, cm.cpd,
    cm.diabunc, cm.diabc, cm.hypothy, cm.rf, cm.ld,
    cm.pud, cm.aids, cm.lymph, cm.metacanc, cm.solidtum,
    cm.rheumd, cm.coag, cm.obes, cm.wloss, cm.fed,
    cm.blane, cm.dane, cm.alcohol, cm.drug, cm.psycho, cm.depre

FROM io_cohort c
JOIN io_outcomes o  ON c.DSYSRTKY = o.DSYSRTKY
JOIN io_comorbidity cm ON c.DSYSRTKY = cm.DSYSRTKY
LEFT JOIN chemo_io_flag cf ON c.DSYSRTKY = cf.DSYSRTKY
LEFT JOIN fips_cte f ON c.DSYSRTKY = f.DSYSRTKY
LEFT JOIN rucc_lookup r ON f.fips_cd = r.FIPS
LEFT JOIN io_sensitivity_vars sv ON c.DSYSRTKY = sv.DSYSRTKY
LEFT JOIN ed_counts ec       ON c.DSYSRTKY = ec.DSYSRTKY
LEFT JOIN icu_last_30d_cte icu ON c.DSYSRTKY = icu.DSYSRTKY
LEFT JOIN adm_last_30d_cte adm ON c.DSYSRTKY = adm.DSYSRTKY;
