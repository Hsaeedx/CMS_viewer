-- Chemo-ICI definition expansion scan.
-- Current definition: platinum (cisplatin J9060 or carboplatin J9045) within
--   ±21 days of ANY ICI administration.
-- Question: how many additional primary-cohort patients would be classified
--   as receiving chemo-ICI if we expanded the chemo set to also include
--   docetaxel (J9171), paclitaxel (J9267), cetuximab (J9055), 5-FU (J9190)?
-- Denominator: io_cohort (primary cohort, N=2,527).

SET memory_limit='24GB';
SET threads=12;

WITH chemo_claims AS (
    -- Pull all 6 chemo-of-interest agents from carrier + outpatient revenue,
    -- restricted to the primary cohort.
    SELECT l.DSYSRTKY,
           TRY_STRPTIME(l.THRU_DT,'%Y%m%d') AS chemo_dt,
           l.HCPCS_CD
    FROM io_car_lines l
    JOIN io_cohort c ON l.DSYSRTKY = c.DSYSRTKY
    WHERE l.HCPCS_CD IN ('J9060','J9045','J9171','J9267','J9055','J9190')

    UNION ALL

    SELECT r.DSYSRTKY,
           TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT),'%Y%m%d') AS chemo_dt,
           r.HCPCS_CD
    FROM io_out_revenue r
    JOIN io_cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE r.HCPCS_CD IN ('J9060','J9045','J9171','J9267','J9055','J9190')
),

-- For each (patient × chemo claim × ICI claim) within ±21d, record which agent
chemo_ici_matches AS (
    SELECT DISTINCT
        ch.DSYSRTKY,
        ch.HCPCS_CD,
        CASE
            WHEN ch.HCPCS_CD IN ('J9060','J9045') THEN 'platinum'
            WHEN ch.HCPCS_CD = 'J9055'            THEN 'cetuximab'
            WHEN ch.HCPCS_CD = 'J9171'            THEN 'docetaxel'
            WHEN ch.HCPCS_CD = 'J9267'            THEN 'paclitaxel'
            WHEN ch.HCPCS_CD = 'J9190'            THEN '5-FU'
        END AS agent_grp
    FROM chemo_claims ch
    JOIN io_claims_raw ici ON ch.DSYSRTKY = ici.DSYSRTKY
    WHERE ABS(datediff('day', ch.chemo_dt, ici.io_date)) <= 21
),

-- Per-patient flags: did they have each agent within ±21d of ICI?
pt_flags AS (
    SELECT
        c.DSYSRTKY,
        BOOL_OR(m.agent_grp = 'platinum')   AS w_plat,
        BOOL_OR(m.agent_grp = 'cetuximab')  AS w_cetux,
        BOOL_OR(m.agent_grp = 'docetaxel')  AS w_doce,
        BOOL_OR(m.agent_grp = 'paclitaxel') AS w_pacli,
        BOOL_OR(m.agent_grp = '5-FU')       AS w_5fu
    FROM io_cohort c
    LEFT JOIN chemo_ici_matches m ON c.DSYSRTKY = m.DSYSRTKY
    GROUP BY c.DSYSRTKY
)

SELECT
    COUNT(*) AS n_cohort,
    -- Current definition: platinum-only
    COUNT(*) FILTER (WHERE COALESCE(w_plat,FALSE))                              AS n_current_chemo_ici,
    -- Each agent's contribution (within ±21d of ICI)
    COUNT(*) FILTER (WHERE COALESCE(w_cetux,FALSE))                             AS n_with_cetux,
    COUNT(*) FILTER (WHERE COALESCE(w_doce,FALSE))                              AS n_with_doce,
    COUNT(*) FILTER (WHERE COALESCE(w_pacli,FALSE))                             AS n_with_pacli,
    COUNT(*) FILTER (WHERE COALESCE(w_5fu,FALSE))                               AS n_with_5fu,
    -- Patients added by each individual agent (i.e., had the agent but NOT platinum)
    COUNT(*) FILTER (WHERE COALESCE(w_cetux,FALSE)  AND NOT COALESCE(w_plat,FALSE)) AS n_cetux_no_plat,
    COUNT(*) FILTER (WHERE COALESCE(w_doce,FALSE)   AND NOT COALESCE(w_plat,FALSE)) AS n_doce_no_plat,
    COUNT(*) FILTER (WHERE COALESCE(w_pacli,FALSE)  AND NOT COALESCE(w_plat,FALSE)) AS n_pacli_no_plat,
    COUNT(*) FILTER (WHERE COALESCE(w_5fu,FALSE)    AND NOT COALESCE(w_plat,FALSE)) AS n_5fu_no_plat,
    -- Expanded definition: any of the 6 agents within ±21d of ICI
    COUNT(*) FILTER (WHERE COALESCE(w_plat,FALSE) OR COALESCE(w_cetux,FALSE)
                       OR COALESCE(w_doce,FALSE) OR COALESCE(w_pacli,FALSE)
                       OR COALESCE(w_5fu,FALSE))                                 AS n_expanded_chemo_ici
FROM pt_flags;
