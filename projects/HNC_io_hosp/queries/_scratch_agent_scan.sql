-- Scan: upper-bound estimate of patients added if we expanded the cohort to include
-- docetaxel (J9171), paclitaxel (J9267), cetuximab (J9055), 5-FU (J9190).
-- Denominator = HNC decedents with confirmed HNC dx + eligible subsite (no FFS filter,
-- because io_ffs_eligible is pre-filtered to ICI recipients via io_episodes).

SET memory_limit='24GB';
SET threads=12;

WITH eligible AS (
    SELECT d.DSYSRTKY, d.death_dt_parsed AS death_dt
    FROM io_decedents d
    JOIN io_hnc_confirmed h ON d.DSYSRTKY = h.DSYSRTKY
    JOIN io_subsite       s ON d.DSYSRTKY = s.DSYSRTKY
),

agent_claims AS (
    SELECT l.DSYSRTKY, TRY_STRPTIME(l.THRU_DT, '%Y%m%d') AS claim_dt, l.HCPCS_CD
    FROM CAR_linek_all l
    WHERE l.HCPCS_CD IN ('J9271','J9299','J9171','J9267','J9055','J9190')

    UNION ALL

    SELECT r.DSYSRTKY,
           TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d') AS claim_dt,
           r.HCPCS_CD
    FROM out_revenuek_all r
    WHERE r.HCPCS_CD IN ('J9271','J9299','J9171','J9267','J9055','J9190')
),

qualifying AS (
    SELECT e.DSYSRTKY,
           CASE
               WHEN a.HCPCS_CD IN ('J9271','J9299') THEN 'ICI'
               WHEN a.HCPCS_CD = 'J9055'            THEN 'cetuximab'
               WHEN a.HCPCS_CD = 'J9171'            THEN 'docetaxel'
               WHEN a.HCPCS_CD = 'J9267'            THEN 'paclitaxel'
               WHEN a.HCPCS_CD = 'J9190'            THEN '5-FU'
           END AS agent_group
    FROM agent_claims a
    JOIN eligible e ON a.DSYSRTKY = e.DSYSRTKY
    WHERE a.claim_dt BETWEEN (e.death_dt - INTERVAL 24 MONTH) AND e.death_dt
),

pt_flags AS (
    SELECT DSYSRTKY,
           BOOL_OR(agent_group = 'ICI')        AS had_ici,
           BOOL_OR(agent_group = 'cetuximab')  AS had_cetux,
           BOOL_OR(agent_group = 'docetaxel')  AS had_doce,
           BOOL_OR(agent_group = 'paclitaxel') AS had_pacli,
           BOOL_OR(agent_group = '5-FU')       AS had_5fu
    FROM qualifying
    GROUP BY DSYSRTKY
)

SELECT
    -- Patients with ICI (matches existing eligibility pool — pre-FFS filter)
    COUNT(*) FILTER (WHERE had_ici)                                          AS n_ici_in_pool,
    -- NEW patients gained if we add the 4 agents (no ICI claim, but ≥1 of the 4)
    COUNT(*) FILTER (WHERE NOT had_ici AND (had_cetux OR had_doce OR had_pacli OR had_5fu))
                                                                              AS n_newly_added,
    -- Per-agent (newly added — exclusive to that agent class, no ICI overlap)
    COUNT(*) FILTER (WHERE NOT had_ici AND had_cetux) AS n_cetux_no_ici,
    COUNT(*) FILTER (WHERE NOT had_ici AND had_doce)  AS n_doce_no_ici,
    COUNT(*) FILTER (WHERE NOT had_ici AND had_pacli) AS n_pacli_no_ici,
    COUNT(*) FILTER (WHERE NOT had_ici AND had_5fu)   AS n_5fu_no_ici,
    -- Overlap: ICI patients who also had at least one of the new agents
    COUNT(*) FILTER (WHERE had_ici AND (had_cetux OR had_doce OR had_pacli OR had_5fu))
                                                                              AS n_ici_with_other,
    -- Any of the 4 new agents, regardless of ICI status
    COUNT(*) FILTER (WHERE had_cetux) AS n_any_cetux,
    COUNT(*) FILTER (WHERE had_doce)  AS n_any_doce,
    COUNT(*) FILTER (WHERE had_pacli) AS n_any_pacli,
    COUNT(*) FILTER (WHERE had_5fu)   AS n_any_5fu
FROM pt_flags;
