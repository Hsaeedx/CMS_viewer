-- Final cohort assembly
-- Sequential joins applying all inclusion/exclusion criteria
-- Criterion: ≥180 days from first HNC dx date to last IO episode start
-- Output: io_cohort (one row per patient, all key dates + MBSF demographics)

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_cohort;

CREATE TABLE io_cohort AS

SELECT
    d.DSYSRTKY,
    -- Death
    d.death_dt_parsed AS death_dt,
    d.DEATH_DT AS death_dt_raw,
    -- HNC diagnosis
    hnc.first_hnc_dx_date,
    -- Subsite
    sub.predominant_subsite,
    sub.subsite_category,
    -- IO episodes
    ep.last_io_episode_start,
    ep.last_io_episode_end,
    ep.last_episode_doses,
    ep.last_episode_agent,
    ep.total_io_episodes,
    ep.first_io_date,
    ep.last_io_date,
    ep.total_io_doses,
    ep.io_agent,
    ep.io_duration_days,
    -- Days from HNC dx to last IO episode (inclusion criterion: ≥180)
    datediff('day', hnc.first_hnc_dx_date, ep.last_io_episode_start) AS days_dx_to_last_io_episode,
    -- Curative therapy
    cur.curative_therapy_date,
    cur.primary_curative_type,
    cur.had_surgery,
    cur.had_radiation,
    -- Demographics (from MBSF)
    d.age_at_death,
    d.SEX,
    d.RACE,
    d.STATE_CD,
    d.CNTY_CD,
    d.OREC,
    d.rfrnc_yr,
    -- Dual eligibility in month of death
    CASE MONTH(d.death_dt_parsed)
        WHEN 1  THEN d.DUAL_01  WHEN 2  THEN d.DUAL_02
        WHEN 3  THEN d.DUAL_03  WHEN 4  THEN d.DUAL_04
        WHEN 5  THEN d.DUAL_05  WHEN 6  THEN d.DUAL_06
        WHEN 7  THEN d.DUAL_07  WHEN 8  THEN d.DUAL_08
        WHEN 9  THEN d.DUAL_09  WHEN 10 THEN d.DUAL_10
        WHEN 11 THEN d.DUAL_11  WHEN 12 THEN d.DUAL_12
    END AS dual_cd_death_month

FROM io_decedents d
-- Step 3b: confirmed HNC diagnosis
JOIN io_hnc_confirmed hnc ON d.DSYSRTKY = hnc.DSYSRTKY
-- Step 4: eligible subsite
JOIN io_subsite sub ON d.DSYSRTKY = sub.DSYSRTKY
-- Steps 9-11: FFS enrollment / geography / ESRD
JOIN io_ffs_eligible ffs ON d.DSYSRTKY = ffs.DSYSRTKY
-- Step 5-6: IO claims with episode identified
JOIN io_episodes ep ON d.DSYSRTKY = ep.DSYSRTKY
-- Step 7: curative therapy before last IO episode
JOIN io_curative cur ON d.DSYSRTKY = cur.DSYSRTKY

-- Step 8: ≥180 days from HNC dx to last IO episode start
WHERE datediff('day', hnc.first_hnc_dx_date, ep.last_io_episode_start) >= 180;
