-- Steps 1-2: Medicare FFS Decedents, age ≥66 at death
-- Study period: 2017-01-01 to 2023-06-30
-- Source: mbsf_all
-- Output: io_decedents (one row per decedent with all MBSF columns needed downstream)

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_decedents;

CREATE TABLE io_decedents AS

SELECT
    m.DSYSRTKY,
    m.DEATH_DT,
    TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') AS death_dt_parsed,
    CAST(m.RFRNC_YR AS INTEGER) AS rfrnc_yr,
    CAST(m.AGE AS INTEGER) AS age_mbsf,
    -- Age at death: AGE is as of Jan 1 of RFRNC_YR
    CAST(m.AGE AS INTEGER) + (YEAR(TRY_STRPTIME(m.DEATH_DT, '%Y%m%d')) - CAST(m.RFRNC_YR AS INTEGER)) AS age_at_death,
    m.SEX,
    m.RACE,
    m.STATE_CD,
    m.CNTY_CD,
    m.OREC,
    -- Dual eligibility monthly indicators
    m.DUAL_01, m.DUAL_02, m.DUAL_03, m.DUAL_04,
    m.DUAL_05, m.DUAL_06, m.DUAL_07, m.DUAL_08,
    m.DUAL_09, m.DUAL_10, m.DUAL_11, m.DUAL_12,
    -- HMO indicators (FFS = '0')
    m.HMOIND1,  m.HMOIND2,  m.HMOIND3,  m.HMOIND4,
    m.HMOIND5,  m.HMOIND6,  m.HMOIND7,  m.HMOIND8,
    m.HMOIND9,  m.HMOIND10, m.HMOIND11, m.HMOIND12,
    -- Medicare status monthly
    m.MDCR_STUS_CD_01, m.MDCR_STUS_CD_02, m.MDCR_STUS_CD_03, m.MDCR_STUS_CD_04,
    m.MDCR_STUS_CD_05, m.MDCR_STUS_CD_06, m.MDCR_STUS_CD_07, m.MDCR_STUS_CD_08,
    m.MDCR_STUS_CD_09, m.MDCR_STUS_CD_10, m.MDCR_STUS_CD_11, m.MDCR_STUS_CD_12,
    m.A_MO_CNT,
    m.B_MO_CNT,
    m.HMO_MO

FROM mbsf_all m

WHERE m.DEATH_DT IS NOT NULL
  AND m.DEATH_DT <> ''
  AND TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') BETWEEN '2017-01-01' AND '2023-06-30'
  -- Age ≥66 at death
  AND (
      CAST(m.AGE AS INTEGER)
      + (YEAR(TRY_STRPTIME(m.DEATH_DT, '%Y%m%d')) - CAST(m.RFRNC_YR AS INTEGER))
  ) >= 66;
