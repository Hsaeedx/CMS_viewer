-- opscc_survival.sql
-- Builds opscc_survival: one row per patient in opscc_propensity with a
-- defined first_tx_date, joined to MBSF for death date and December-31 censoring.
--
-- death_date:   MAX(DEATH_DT) across all MBSF enrollment years
-- censor_date:  December 31 of the patient's last enrollment year
-- event:        1 = died, 0 = censored
-- t_days:       days from first_tx_date to death or censor
--
-- PSM match flags (psm_matched_A / psm_matched_B) live on opscc_propensity.
-- Python analysis scripts join this table with opscc_propensity on DSYSRTKY
-- to filter to the matched cohort for each comparison.

DROP TABLE IF EXISTS opscc_survival;

CREATE TABLE opscc_survival AS

WITH mbsf_sum AS (
    SELECT
        m.DSYSRTKY,
        MAX(CAST(m.RFRNC_YR AS INTEGER))                                    AS last_yr,
        MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                 THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END)              AS death_date
    FROM mbsf_all m
    JOIN opscc_propensity p ON m.DSYSRTKY = p.DSYSRTKY
    WHERE p.first_tx_date IS NOT NULL
    GROUP BY m.DSYSRTKY
)

SELECT
    p.DSYSRTKY,
    p.tx_group,
    p.first_tx_date,
    p.age_at_dx,
    p.van_walraven_score,
    s.death_date,
    make_date(s.last_yr, 12, 31)                                            AS censor_date,
    CASE WHEN s.death_date IS NOT NULL THEN 1 ELSE 0 END                    AS event,
    DATEDIFF('day', p.first_tx_date,
             COALESCE(s.death_date, make_date(s.last_yr, 12, 31)))          AS t_days
FROM opscc_propensity p
JOIN mbsf_sum s ON p.DSYSRTKY = s.DSYSRTKY
WHERE p.first_tx_date IS NOT NULL;

SELECT tx_group,
       COUNT(*)     AS n,
       SUM(event)   AS n_deaths,
       ROUND(100.0 * SUM(event) / COUNT(*), 1) AS pct_deaths
FROM opscc_survival
GROUP BY tx_group
ORDER BY tx_group;
