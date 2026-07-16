-- opscc_gtube_dependence.sql  —  Multi-interval G-tube event tracking
--
-- Scoped to PSM-matched patients (Comp A, B, or C).
--
-- PRIMARY signal: placement procedure events (CPT 43246/49440, HCPCS B4087/B4088,
--                 ICD-10-PCS 0DH60UZ/0DH63UZ). These fire only when a new tube
--                 is actually placed. Reported as `placement_by_{T}` flags.
--
-- SECONDARY/sensitivity signal: any G-tube-related event (placement + exchange
--                               CPTs 43760/43762/43763/49450 + Z93.1 status dx).
--                               Reported as `event_by_{T}` flags. Z93.1 inflates
--                               cumulative incidence in arms with high baseline
--                               pre-existing tube prevalence (RT alone, CRT) and
--                               should be interpreted with caution.

-- ── STEP 1: Matched cohort ──────────────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _gd_cohort AS
SELECT p.DSYSRTKY,
       p.first_hnc_date::DATE AS dx_date,
       p.first_tx_date,
       -- Treatment completion = last RT/chemo date; falls back to first_tx_date
       -- (TORS-alone patients have no RT/chemo). Anchor for delayed-toxicity metrics.
       COALESCE(list_max([co.last_rt_date, co.last_chemo_date]), p.first_tx_date) AS tx_completion_date
FROM opscc_propensity p
LEFT JOIN opscc_cohort co ON co.DSYSRTKY = p.DSYSRTKY
WHERE p.first_tx_date IS NOT NULL
  AND (p.psm_matched_A = TRUE OR p.psm_matched_B = TRUE OR p.psm_matched_C = TRUE);

-- ── STEP 2: Carrier line — placement + exchange HCPCS/CPT ──────────────────

CREATE OR REPLACE TEMP TABLE _gd_car AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS dt,
       CASE WHEN cl.HCPCS_CD IN ('43246','49440','B4087','B4088') THEN 'placement'
            WHEN cl.HCPCS_CD IN ('43760','43762','43763','49450') THEN 'exchange'
       END AS kind
FROM _gd_cohort c
JOIN car_linek_all cl USING (DSYSRTKY)
WHERE cl.HCPCS_CD IN (
    '43246','49440','B4087','B4088',
    '43760','43762','43763','49450'
);

-- ── STEP 3: Outpatient revenue — placement + exchange HCPCS/CPT ────────────

CREATE OR REPLACE TEMP TABLE _gd_rev AS
SELECT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS dt,
       CASE WHEN r.HCPCS_CD IN ('43246','49440','B4087','B4088') THEN 'placement'
            WHEN r.HCPCS_CD IN ('43760','43762','43763','49450') THEN 'exchange'
       END AS kind
FROM _gd_cohort c
JOIN out_revenuek_all r USING (DSYSRTKY)
WHERE r.HCPCS_CD IN (
    '43246','49440','B4087','B4088',
    '43760','43762','43763','49450'
);

-- ── STEP 4: Outpatient claims — Z93.1 dx + ICD-PCS placement ───────────────

CREATE OR REPLACE TEMP TABLE _gd_out AS
SELECT DISTINCT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(oc.THRU_DT, '%Y%m%d') AS dt,
       CASE WHEN oc.ICD_PRCDR_CD1 IN ('0DH60UZ','0DH63UZ')
              OR oc.ICD_PRCDR_CD2 IN ('0DH60UZ','0DH63UZ')
              OR oc.ICD_PRCDR_CD3 IN ('0DH60UZ','0DH63UZ')
              OR oc.ICD_PRCDR_CD4 IN ('0DH60UZ','0DH63UZ')
              OR oc.ICD_PRCDR_CD5 IN ('0DH60UZ','0DH63UZ')
            THEN 'placement'
            ELSE 'z931'
       END AS kind
FROM _gd_cohort c
JOIN out_claimsk_all oc USING (DSYSRTKY)
WHERE
    oc.PRNCPAL_DGNS_CD = 'Z931'
 OR oc.ICD_DGNS_CD1='Z931'  OR oc.ICD_DGNS_CD2='Z931'  OR oc.ICD_DGNS_CD3='Z931'
 OR oc.ICD_DGNS_CD4='Z931'  OR oc.ICD_DGNS_CD5='Z931'  OR oc.ICD_DGNS_CD6='Z931'
 OR oc.ICD_DGNS_CD7='Z931'  OR oc.ICD_DGNS_CD8='Z931'  OR oc.ICD_DGNS_CD9='Z931'
 OR oc.ICD_DGNS_CD10='Z931' OR oc.ICD_DGNS_CD11='Z931' OR oc.ICD_DGNS_CD12='Z931'
 OR oc.ICD_DGNS_CD13='Z931' OR oc.ICD_DGNS_CD14='Z931' OR oc.ICD_DGNS_CD15='Z931'
 OR oc.ICD_DGNS_CD16='Z931' OR oc.ICD_DGNS_CD17='Z931' OR oc.ICD_DGNS_CD18='Z931'
 OR oc.ICD_DGNS_CD19='Z931' OR oc.ICD_DGNS_CD20='Z931' OR oc.ICD_DGNS_CD21='Z931'
 OR oc.ICD_DGNS_CD22='Z931' OR oc.ICD_DGNS_CD23='Z931' OR oc.ICD_DGNS_CD24='Z931'
 OR oc.ICD_DGNS_CD25='Z931'
 OR oc.ICD_PRCDR_CD1  IN ('0DH60UZ','0DH63UZ')
 OR oc.ICD_PRCDR_CD2  IN ('0DH60UZ','0DH63UZ')
 OR oc.ICD_PRCDR_CD3  IN ('0DH60UZ','0DH63UZ')
 OR oc.ICD_PRCDR_CD4  IN ('0DH60UZ','0DH63UZ')
 OR oc.ICD_PRCDR_CD5  IN ('0DH60UZ','0DH63UZ');

-- ── STEP 5: Inpatient claims — Z93.1 dx + ICD-PCS placement ────────────────

CREATE OR REPLACE TEMP TABLE _gd_inp AS
SELECT DISTINCT c.DSYSRTKY, c.dx_date, c.first_tx_date,
       TRY_STRPTIME(i.THRU_DT, '%Y%m%d') AS dt,
       CASE WHEN i.ICD_PRCDR_CD1 IN ('0DH60UZ','0DH63UZ')
              OR i.ICD_PRCDR_CD2 IN ('0DH60UZ','0DH63UZ')
              OR i.ICD_PRCDR_CD3 IN ('0DH60UZ','0DH63UZ')
              OR i.ICD_PRCDR_CD4 IN ('0DH60UZ','0DH63UZ')
              OR i.ICD_PRCDR_CD5 IN ('0DH60UZ','0DH63UZ')
            THEN 'placement'
            ELSE 'z931'
       END AS kind
FROM _gd_cohort c
JOIN inp_claimsk_all i USING (DSYSRTKY)
WHERE
    i.PRNCPAL_DGNS_CD = 'Z931' OR i.ADMTG_DGNS_CD = 'Z931'
 OR i.ICD_DGNS_CD1='Z931'  OR i.ICD_DGNS_CD2='Z931'  OR i.ICD_DGNS_CD3='Z931'
 OR i.ICD_DGNS_CD4='Z931'  OR i.ICD_DGNS_CD5='Z931'  OR i.ICD_DGNS_CD6='Z931'
 OR i.ICD_DGNS_CD7='Z931'  OR i.ICD_DGNS_CD8='Z931'  OR i.ICD_DGNS_CD9='Z931'
 OR i.ICD_DGNS_CD10='Z931' OR i.ICD_DGNS_CD11='Z931' OR i.ICD_DGNS_CD12='Z931'
 OR i.ICD_DGNS_CD13='Z931' OR i.ICD_DGNS_CD14='Z931' OR i.ICD_DGNS_CD15='Z931'
 OR i.ICD_DGNS_CD16='Z931' OR i.ICD_DGNS_CD17='Z931' OR i.ICD_DGNS_CD18='Z931'
 OR i.ICD_DGNS_CD19='Z931' OR i.ICD_DGNS_CD20='Z931' OR i.ICD_DGNS_CD21='Z931'
 OR i.ICD_DGNS_CD22='Z931' OR i.ICD_DGNS_CD23='Z931' OR i.ICD_DGNS_CD24='Z931'
 OR i.ICD_DGNS_CD25='Z931'
 OR i.ICD_PRCDR_CD1  IN ('0DH60UZ','0DH63UZ')
 OR i.ICD_PRCDR_CD2  IN ('0DH60UZ','0DH63UZ')
 OR i.ICD_PRCDR_CD3  IN ('0DH60UZ','0DH63UZ')
 OR i.ICD_PRCDR_CD4  IN ('0DH60UZ','0DH63UZ')
 OR i.ICD_PRCDR_CD5  IN ('0DH60UZ','0DH63UZ');

-- ── STEP 6: Union events; preserve kind so placement-only flags can be built

CREATE OR REPLACE TEMP TABLE _gd_events AS
SELECT DSYSRTKY, dx_date, first_tx_date, dt, kind FROM _gd_car WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt, kind FROM _gd_rev WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt, kind FROM _gd_out WHERE dt IS NOT NULL
UNION
SELECT DSYSRTKY, dx_date, first_tx_date, dt, kind FROM _gd_inp WHERE dt IS NOT NULL;

-- ── STEP 7: Per-patient summary + cumulative-incidence flags ───────────────

DROP TABLE IF EXISTS opscc_gtube_dependence;

CREATE TABLE opscc_gtube_dependence AS
WITH per_pt AS (
    SELECT
        c.DSYSRTKY,
        c.dx_date,
        c.first_tx_date,
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date
             THEN NULL
             ELSE DATE_DIFF('day', c.dx_date, c.first_tx_date)
        END AS days_dx_to_tx,

        -- Pre-tx phase: [dx_date, first_tx_date) — placement events only
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END) > 0)
        END AS pre_placement_any,
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END)
        END AS pre_placement_count,

        -- Pre-tx phase: any G-tube-related event (placement + exchange + z931)
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            (COUNT(DISTINCT CASE WHEN e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END) > 0)
        END AS pre_event_any,
        CASE WHEN c.dx_date IS NULL OR c.first_tx_date < c.dx_date THEN NULL ELSE
            COUNT(DISTINCT CASE WHEN e.dt >= c.dx_date AND e.dt < c.first_tx_date THEN e.dt END)
        END AS pre_event_count,

        -- PRIMARY: placement-only cumulative incidence
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  14 DAY THEN e.dt END) > 0) AS placement_by_14d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  30 DAY THEN e.dt END) > 0) AS placement_by_30d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  90 DAY THEN e.dt END) > 0) AS placement_by_90d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 180 DAY THEN e.dt END) > 0) AS placement_by_180d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 365 DAY THEN e.dt END) > 0) AS placement_by_365d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) > 0) AS placement_by_1095d,

        -- SENSITIVITY: any G-tube event (placement + exchange + Z93.1)
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  14 DAY THEN e.dt END) > 0) AS event_by_14d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  30 DAY THEN e.dt END) > 0) AS event_by_30d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL  90 DAY THEN e.dt END) > 0) AS event_by_90d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 180 DAY THEN e.dt END) > 0) AS event_by_180d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 365 DAY THEN e.dt END) > 0) AS event_by_365d,
        (COUNT(DISTINCT CASE WHEN e.dt BETWEEN c.first_tx_date AND c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) > 0) AS event_by_1095d,

        -- Placement event counts per window (non-cumulative)
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date AND e.dt <  c.first_tx_date + INTERVAL  14 DAY THEN e.dt END) AS placement_cnt_0_14d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL  14 DAY AND e.dt <  c.first_tx_date + INTERVAL  30 DAY THEN e.dt END) AS placement_cnt_14_30d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL  30 DAY AND e.dt <  c.first_tx_date + INTERVAL  90 DAY THEN e.dt END) AS placement_cnt_30_90d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL  90 DAY AND e.dt <  c.first_tx_date + INTERVAL 180 DAY THEN e.dt END) AS placement_cnt_90_180d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL 180 DAY AND e.dt <  c.first_tx_date + INTERVAL 365 DAY THEN e.dt END) AS placement_cnt_180_365d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL 365 DAY AND e.dt <  c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) AS placement_cnt_365_1095d,
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date + INTERVAL 1095 DAY THEN e.dt END) AS placement_cnt_gt_1095d,

        -- Total events of any kind
        COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date THEN e.dt END) AS total_post_placements,
        COUNT(DISTINCT CASE WHEN e.dt >= c.first_tx_date THEN e.dt END)                          AS total_post_events,
        MIN(CASE WHEN e.kind = 'placement' AND e.dt >= c.first_tx_date THEN DATE_DIFF('day', c.first_tx_date, e.dt) END) AS first_post_placement_day,
        MIN(CASE WHEN e.dt >= c.first_tx_date THEN DATE_DIFF('day', c.first_tx_date, e.dt) END) AS first_post_day_from_tx,
        MAX(CASE WHEN e.dt >= c.first_tx_date THEN DATE_DIFF('day', c.first_tx_date, e.dt) END) AS last_post_day_from_tx,

        -- ── DELAYED PLACEMENT (anchored on treatment completion, 90d washout) ──
        -- Marker for delayed swallowing failure / G-tube dependence.
        c.tx_completion_date,
        DATE_DIFF('day', c.first_tx_date, c.tx_completion_date) AS days_tx_to_completion,
        -- Incident-eligibility: any placement up through completion + 90d washout.
        -- Patients flagged here did NOT finish treatment tube-free.
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt <= c.tx_completion_date + INTERVAL 90 DAY THEN e.dt END) > 0) AS placement_through_completion90,
        -- First placement strictly >90d after completion (days from completion).
        -- Skips any earlier acute tube, so "all-comers" delayed counts are correct.
        MIN(CASE WHEN e.kind = 'placement' AND e.dt > c.tx_completion_date + INTERVAL 90 DAY
                 THEN DATE_DIFF('day', c.tx_completion_date, e.dt) END) AS first_delayed_placement_day,
        -- Cumulative delayed-placement flags: placement in (completion+90d, completion+Xd]
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  180 DAY THEN e.dt END) > 0) AS delayed_placement_by_180d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  365 DAY THEN e.dt END) > 0) AS delayed_placement_by_365d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL  730 DAY THEN e.dt END) > 0) AS delayed_placement_by_730d,
        (COUNT(DISTINCT CASE WHEN e.kind = 'placement' AND e.dt >  c.tx_completion_date + INTERVAL  90 DAY AND e.dt <= c.tx_completion_date + INTERVAL 1095 DAY THEN e.dt END) > 0) AS delayed_placement_by_1095d
    FROM _gd_cohort c
    LEFT JOIN _gd_events e ON e.DSYSRTKY = c.DSYSRTKY
    GROUP BY c.DSYSRTKY, c.dx_date, c.first_tx_date, c.tx_completion_date
)
SELECT * FROM per_pt;

-- ── Summary print: placement-only cumulative incidence by arm ──────────────

SELECT
    p.tx_group,
    COUNT(*) AS n,
    ROUND(100.0 * SUM(g.placement_by_14d::INT)   / COUNT(*), 1) AS pct_placement_14d,
    ROUND(100.0 * SUM(g.placement_by_30d::INT)   / COUNT(*), 1) AS pct_placement_30d,
    ROUND(100.0 * SUM(g.placement_by_90d::INT)   / COUNT(*), 1) AS pct_placement_90d,
    ROUND(100.0 * SUM(g.placement_by_180d::INT)  / COUNT(*), 1) AS pct_placement_180d,
    ROUND(100.0 * SUM(g.placement_by_365d::INT)  / COUNT(*), 1) AS pct_placement_1yr,
    ROUND(100.0 * SUM(g.placement_by_1095d::INT) / COUNT(*), 1) AS pct_placement_3yr
FROM opscc_propensity p
JOIN opscc_gtube_dependence g USING (DSYSRTKY)
WHERE p.psm_matched_A OR p.psm_matched_B OR p.psm_matched_C
GROUP BY p.tx_group ORDER BY p.tx_group;
