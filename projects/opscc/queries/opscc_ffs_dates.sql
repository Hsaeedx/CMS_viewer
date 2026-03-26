-- opscc_ffs_dates.sql
-- Computes the last continuous FFS (Medicare Part A+B, non-HMO) date for
-- each patient in opscc_propensity, starting the month after first_tx_date.
--
-- A patient is in FFS for a given month if:
--   BUYIN = '3' (Part A + B buy-in)  AND  HMOIND IN ('0', '4') (not managed care)
-- The first month that fails this criterion ends FFS follow-up.
--
-- last_ffs_date:    last day of the final consecutive FFS month
--                   (or last enrolled month if all months were FFS)
-- ffs_censor_date:  LEAST(last_ffs_date, death_date) — effective outcomes
--                   censoring date used by outcomes_analysis.py and make_figures.py

DROP TABLE IF EXISTS opscc_ffs_dates;

CREATE TABLE opscc_ffs_dates AS

WITH months AS (
    SELECT
        m.DSYSRTKY,
        m.DEATH_DT,
        make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)                    AS mo_start,
        make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
            + INTERVAL 1 MONTH - INTERVAL 1 DAY                            AS mo_end,
        CASE t.mo
            WHEN 1  THEN m.BUYIN1  WHEN 2  THEN m.BUYIN2
            WHEN 3  THEN m.BUYIN3  WHEN 4  THEN m.BUYIN4
            WHEN 5  THEN m.BUYIN5  WHEN 6  THEN m.BUYIN6
            WHEN 7  THEN m.BUYIN7  WHEN 8  THEN m.BUYIN8
            WHEN 9  THEN m.BUYIN9  WHEN 10 THEN m.BUYIN10
            WHEN 11 THEN m.BUYIN11 WHEN 12 THEN m.BUYIN12
        END AS buyin,
        CASE t.mo
            WHEN 1  THEN m.HMOIND1  WHEN 2  THEN m.HMOIND2
            WHEN 3  THEN m.HMOIND3  WHEN 4  THEN m.HMOIND4
            WHEN 5  THEN m.HMOIND5  WHEN 6  THEN m.HMOIND6
            WHEN 7  THEN m.HMOIND7  WHEN 8  THEN m.HMOIND8
            WHEN 9  THEN m.HMOIND9  WHEN 10 THEN m.HMOIND10
            WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
        END AS hmoind
    FROM mbsf_all m
    JOIN opscc_propensity p ON m.DSYSRTKY = p.DSYSRTKY,
    UNNEST([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) AS t(mo)
    WHERE p.first_tx_date IS NOT NULL
      AND make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
              >= date_trunc('month', p.first_tx_date) + INTERVAL 1 MONTH
),

agg AS (
    SELECT
        DSYSRTKY,
        COALESCE(
            MIN(CASE WHEN NOT (buyin = '3' AND hmoind IN ('0', '4'))
                     THEN mo_start END) - INTERVAL 1 DAY,
            MAX(mo_end)
        )                                                                   AS last_ffs_date,
        MAX(CASE WHEN DEATH_DT IS NOT NULL AND DEATH_DT != ''
                 THEN TRY_STRPTIME(DEATH_DT, '%Y%m%d') END)                AS death_date
    FROM months
    GROUP BY DSYSRTKY
)

SELECT
    DSYSRTKY,
    last_ffs_date,
    death_date,
    CASE
        WHEN death_date IS NOT NULL AND death_date < last_ffs_date
        THEN death_date
        ELSE last_ffs_date
    END                                                                     AS ffs_censor_date
FROM agg;

SELECT COUNT(*) AS n_patients FROM opscc_ffs_dates;
