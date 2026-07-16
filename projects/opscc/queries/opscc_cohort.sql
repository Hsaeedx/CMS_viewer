-- opscc_cohort.sql
-- Applies enrollment and cancer exclusion criteria to opscc_universe.
--
-- Inclusion: Enrolled in Medicare FFS (Part A+B, non-HMO) in the month of
--            first_hnc_date. FFS month = BUYIN = '3' AND HMOIND IN ('0','4')
--            Post-diagnosis FFS dropout is handled by censoring in opscc_ffs_dates.
--
-- Exclusion: Any non-oropharyngeal malignancy (ICD-10 C-codes, excluding
--            C01/C09/C10/C14) appearing in any claim in the 12 months before
--            first_hnc_date. Screens all available claims regardless of
--            enrollment continuity during the lookback.

DROP TABLE IF EXISTS opscc_cohort;

CREATE TABLE opscc_cohort AS

WITH dx_month_enroll AS (
    -- Require FFS enrollment in the month of diagnosis
    SELECT
        m.DSYSRTKY,
        CASE mth
            WHEN 1  THEN m.BUYIN1  WHEN 2  THEN m.BUYIN2
            WHEN 3  THEN m.BUYIN3  WHEN 4  THEN m.BUYIN4
            WHEN 5  THEN m.BUYIN5  WHEN 6  THEN m.BUYIN6
            WHEN 7  THEN m.BUYIN7  WHEN 8  THEN m.BUYIN8
            WHEN 9  THEN m.BUYIN9  WHEN 10 THEN m.BUYIN10
            WHEN 11 THEN m.BUYIN11 WHEN 12 THEN m.BUYIN12
        END AS buyin,
        CASE mth
            WHEN 1  THEN m.HMOIND1  WHEN 2  THEN m.HMOIND2
            WHEN 3  THEN m.HMOIND3  WHEN 4  THEN m.HMOIND4
            WHEN 5  THEN m.HMOIND5  WHEN 6  THEN m.HMOIND6
            WHEN 7  THEN m.HMOIND7  WHEN 8  THEN m.HMOIND8
            WHEN 9  THEN m.HMOIND9  WHEN 10 THEN m.HMOIND10
            WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
        END AS hmoind
    FROM mbsf_all m
    JOIN opscc_universe o ON m.DSYSRTKY = o.DSYSRTKY
      AND TRY_CAST(m.RFRNC_YR AS INTEGER) = YEAR(o.first_hnc_date),
    UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mth)
    WHERE mth = MONTH(o.first_hnc_date)
),

enrolled AS (
    -- Keep patients who were in FFS in their diagnosis month
    SELECT o.DSYSRTKY, o.first_hnc_date
    FROM opscc_universe o
    JOIN dx_month_enroll e ON o.DSYSRTKY = e.DSYSRTKY
    WHERE e.buyin = '3'
      AND e.hmoind IN ('0', '4')
),

prior_cancer AS (
    -- Any non-oropharyngeal malignancy in the 12-month lookback
    -- Screens all available claims regardless of enrollment continuity
    SELECT DISTINCT dsysrtky FROM (

        SELECT i.DSYSRTKY, t.code
        FROM inp_claimsk_all i
        JOIN enrolled e ON i.DSYSRTKY = e.DSYSRTKY,
        UNNEST([
            i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
            i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
            i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
            i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
            i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15, i.ICD_DGNS_CD16,
            i.ICD_DGNS_CD17, i.ICD_DGNS_CD18, i.ICD_DGNS_CD19, i.ICD_DGNS_CD20,
            i.ICD_DGNS_CD21, i.ICD_DGNS_CD22, i.ICD_DGNS_CD23, i.ICD_DGNS_CD24,
            i.ICD_DGNS_CD25
        ]) AS t(code)
        WHERE TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
                  BETWEEN e.first_hnc_date - INTERVAL 12 MONTH
                      AND e.first_hnc_date

        UNION ALL

        SELECT c.DSYSRTKY, c.LINE_ICD_DGNS_CD AS code
        FROM car_linek_all c
        JOIN enrolled e ON c.DSYSRTKY = e.DSYSRTKY
        WHERE TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
                  BETWEEN e.first_hnc_date - INTERVAL 12 MONTH
                      AND e.first_hnc_date

        UNION ALL

        SELECT oc.DSYSRTKY, t.code
        FROM out_claimsk_all oc
        JOIN enrolled e ON oc.DSYSRTKY = e.DSYSRTKY,
        UNNEST([
            oc.PRNCPAL_DGNS_CD,
            oc.ICD_DGNS_CD1,  oc.ICD_DGNS_CD2,  oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4,
            oc.ICD_DGNS_CD5,  oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
            oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10, oc.ICD_DGNS_CD11, oc.ICD_DGNS_CD12,
            oc.ICD_DGNS_CD13, oc.ICD_DGNS_CD14, oc.ICD_DGNS_CD15, oc.ICD_DGNS_CD16,
            oc.ICD_DGNS_CD17, oc.ICD_DGNS_CD18, oc.ICD_DGNS_CD19, oc.ICD_DGNS_CD20,
            oc.ICD_DGNS_CD21, oc.ICD_DGNS_CD22, oc.ICD_DGNS_CD23, oc.ICD_DGNS_CD24,
            oc.ICD_DGNS_CD25
        ]) AS t(code)
        WHERE TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
                  BETWEEN e.first_hnc_date - INTERVAL 12 MONTH
                      AND e.first_hnc_date
    )
    WHERE code LIKE 'C%'
      AND LEFT(code, 3) NOT IN ('C01', 'C09', 'C10', 'C14')   -- OPSCC primaries
      AND LEFT(code, 3) NOT IN ('C76', 'C77', 'C78', 'C79', 'C80')  -- secondary/metastatic/unknown-primary codes: these appear as staging from OPSCC itself, not separate prior cancers
)

SELECT
    e.DSYSRTKY,
    e.first_hnc_date
FROM enrolled e
WHERE e.DSYSRTKY NOT IN (SELECT DSYSRTKY FROM prior_cancer);

SELECT
    COUNT(*)  AS n_cohort
FROM opscc_cohort;
