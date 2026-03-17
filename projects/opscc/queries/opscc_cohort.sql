DROP TABLE IF EXISTS opscc_cohort;

CREATE TABLE opscc_cohort AS

WITH enroll AS (

    SELECT
        m.DSYSRTKY,
        make_date(CAST(m.RFRNC_YR AS INTEGER), mth, 1) AS month_start,

        CASE
            WHEN
                (
                    CASE mth
                        WHEN 1 THEN m.HMOIND1 WHEN 2 THEN m.HMOIND2
                        WHEN 3 THEN m.HMOIND3 WHEN 4 THEN m.HMOIND4
                        WHEN 5 THEN m.HMOIND5 WHEN 6 THEN m.HMOIND6
                        WHEN 7 THEN m.HMOIND7 WHEN 8 THEN m.HMOIND8
                        WHEN 9 THEN m.HMOIND9 WHEN 10 THEN m.HMOIND10
                        WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
                    END
                ) = '0'
            AND
                (
                    CASE mth
                        WHEN 1 THEN m.MDCR_STUS_CD_01 WHEN 2 THEN m.MDCR_STUS_CD_02
                        WHEN 3 THEN m.MDCR_STUS_CD_03 WHEN 4 THEN m.MDCR_STUS_CD_04
                        WHEN 5 THEN m.MDCR_STUS_CD_05 WHEN 6 THEN m.MDCR_STUS_CD_06
                        WHEN 7 THEN m.MDCR_STUS_CD_07 WHEN 8 THEN m.MDCR_STUS_CD_08
                        WHEN 9 THEN m.MDCR_STUS_CD_09 WHEN 10 THEN m.MDCR_STUS_CD_10
                        WHEN 11 THEN m.MDCR_STUS_CD_11 WHEN 12 THEN m.MDCR_STUS_CD_12
                    END
                ) IN ('10','11','20','21')
            THEN 1 ELSE 0
        END AS ffs_month

    FROM mbsf_all m,
    UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mth)
)

SELECT
    o.DSYSRTKY,
    o.first_hnc_date

FROM opscc_universe o
JOIN enroll e
  ON o.DSYSRTKY = e.DSYSRTKY
  AND e.month_start BETWEEN
      date_trunc('month', o.first_hnc_date)
      AND date_trunc('month', o.first_hnc_date) + INTERVAL 5 MONTH

GROUP BY o.DSYSRTKY, o.first_hnc_date
HAVING SUM(e.ffs_month) = 6;