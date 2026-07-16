-- Annotate opscc_cohort with first TORS date, first chemo date, first RT date,
-- metastatic flag (C78/C79 = true distant mets), and nodal flag (C77 = regional N+ staging).
--
-- Treatment groups (applied in propensity.sql):
--   TORS alone  | RT alone  | TORS + RT  | CRT
--
-- Metastatic flag: C78/C79 within ±90d of first_hnc_date → excluded from both comparisons.
-- Nodal flag:      C77 within ±90d of first_hnc_date     → excluded from Comp A only.
--
-- PERFORMANCE: each large table is scanned exactly once.
-- All signals (TORS/chemo/RT/met/nodal) extracted in a single GROUP BY per table.

ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS first_tors_date   DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS first_chemo_date  DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS first_rt_date     DATE;
-- Last RT / chemo date within the 12-month treatment window — marks the end of the
-- (chemo)radiation course. Used to anchor "delayed toxicity after completion" metrics.
-- Window-capped to avoid grabbing later salvage/recurrence therapy.
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS last_chemo_date   DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS last_rt_date      DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS has_metastatic_dx BOOLEAN;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS has_nodal_dx      BOOLEAN;

WITH

-- ─── 1. INPATIENT ─────────────────────────────────────────────────────────────
-- Extracts: TORS (robotic PCS), chemo (3E0% PCS), RT (D9%/D7_3%/DW_1% PCS),
--           metastatic dx (C78/C79), nodal dx (C77) — all in one scan.
inp_scan AS (
    SELECT
        i.DSYSRTKY,

        MIN(CASE WHEN len(list_filter(
            [i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
             i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
             i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
             i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
             i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
             i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
             i.ICD_PRCDR_CD25],
            x -> x IN ('8E09XCZ','8E097CZ','8E090CZ','8E098CZ')
        )) > 0
        THEN TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
        END) AS tors_date,

        MIN(CASE WHEN len(list_filter(
            [i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
             i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
             i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
             i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
             i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
             i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
             i.ICD_PRCDR_CD25],
            x -> x LIKE '3E0%'
        )) > 0
        THEN TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
        END) AS chemo_date,

        MIN(CASE WHEN len(list_filter(
            [i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
             i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
             i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
             i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
             i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
             i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
             i.ICD_PRCDR_CD25],
            x -> x LIKE 'D9%' OR x LIKE 'D7_3%' OR x LIKE 'DW_1%'
        )) > 0
        THEN TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
        END) AS rt_date,

        -- Last chemo / RT within the 12-month treatment window (course completion)
        MAX(CASE WHEN len(list_filter(
            [i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
             i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
             i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
             i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
             i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
             i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
             i.ICD_PRCDR_CD25],
            x -> x LIKE '3E0%'
        )) > 0
        AND TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
              BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
        THEN TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
        END) AS chemo_date_last,

        MAX(CASE WHEN len(list_filter(
            [i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
             i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
             i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
             i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
             i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
             i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
             i.ICD_PRCDR_CD25],
            x -> x LIKE 'D9%' OR x LIKE 'D7_3%' OR x LIKE 'DW_1%'
        )) > 0
        AND TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
              BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
        THEN TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d')
        END) AS rt_date_last,

        MAX(CASE WHEN len(list_filter(
            [i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
             i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
             i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
             i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
             i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15, i.ICD_DGNS_CD16,
             i.ICD_DGNS_CD17, i.ICD_DGNS_CD18, i.ICD_DGNS_CD19, i.ICD_DGNS_CD20,
             i.ICD_DGNS_CD21, i.ICD_DGNS_CD22, i.ICD_DGNS_CD23, i.ICD_DGNS_CD24,
             i.ICD_DGNS_CD25],
            x -> x LIKE 'C78%' OR x LIKE 'C79%'
        )) > 0
        AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
        THEN 1 ELSE 0 END) AS is_metastatic,

        MAX(CASE WHEN len(list_filter(
            [i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
             i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
             i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
             i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
             i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15, i.ICD_DGNS_CD16,
             i.ICD_DGNS_CD17, i.ICD_DGNS_CD18, i.ICD_DGNS_CD19, i.ICD_DGNS_CD20,
             i.ICD_DGNS_CD21, i.ICD_DGNS_CD22, i.ICD_DGNS_CD23, i.ICD_DGNS_CD24,
             i.ICD_DGNS_CD25],
            x -> x LIKE 'C77%'
        )) > 0
        AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
        THEN 1 ELSE 0 END) AS is_nodal

    FROM inp_claimsk_all i
    JOIN opscc_cohort o ON i.DSYSRTKY = o.DSYSRTKY
    GROUP BY i.DSYSRTKY
),

-- ─── 2. CARRIER ───────────────────────────────────────────────────────────────
-- Extracts: TORS/chemo/RT dates (HCPCS), metastatic/nodal dx (LINE_ICD_DGNS_CD).
-- Single column per row — no UNNEST needed.
car_scan AS (
    SELECT
        c.DSYSRTKY,

        MIN(CASE WHEN c.HCPCS_CD IN ('1007190','42842','42844','42845')
            THEN TRY_STRPTIME(c.THRU_DT, '%Y%m%d') END) AS tors_date,

        MIN(CASE WHEN c.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')
            THEN TRY_STRPTIME(c.THRU_DT, '%Y%m%d') END) AS chemo_date,

        -- 77427 (radiation management) included in carrier only — proof of active delivery
        MIN(CASE WHEN c.HCPCS_CD IN (
                '77402','77407','77412',
                '77385','77386',
                'G6015','G6016',
                '77373',
                '77520','77522','77523','77525',
                '77771','77772','77773',
                '77427')
            THEN TRY_STRPTIME(c.THRU_DT, '%Y%m%d') END) AS rt_date,

        -- Last chemo / RT within the 12-month treatment window (course completion)
        MAX(CASE WHEN c.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')
            AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
            THEN TRY_STRPTIME(c.THRU_DT, '%Y%m%d') END) AS chemo_date_last,

        MAX(CASE WHEN c.HCPCS_CD IN (
                '77402','77407','77412',
                '77385','77386',
                'G6015','G6016',
                '77373',
                '77520','77522','77523','77525',
                '77771','77772','77773',
                '77427')
            AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
            THEN TRY_STRPTIME(c.THRU_DT, '%Y%m%d') END) AS rt_date_last,

        MAX(CASE WHEN (c.LINE_ICD_DGNS_CD LIKE 'C78%' OR c.LINE_ICD_DGNS_CD LIKE 'C79%')
            AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
            THEN 1 ELSE 0 END) AS is_metastatic,

        MAX(CASE WHEN c.LINE_ICD_DGNS_CD LIKE 'C77%'
            AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
            THEN 1 ELSE 0 END) AS is_nodal

    FROM car_linek_all c
    JOIN opscc_cohort o ON c.DSYSRTKY = o.DSYSRTKY
    GROUP BY c.DSYSRTKY
),

-- ─── 3. OUTPATIENT REVENUE ────────────────────────────────────────────────────
-- Extracts: TORS/chemo/RT dates (HCPCS + REV_CNTR for RT).
-- No diagnosis codes on revenue lines — outpatient dx handled by out_claimsk_all below.
out_scan AS (
    SELECT
        r.DSYSRTKY,

        MIN(CASE WHEN r.HCPCS_CD IN ('1007190','42842','42844','42845')
            THEN TRY_STRPTIME(r.THRU_DT, '%Y%m%d') END) AS tors_date,

        MIN(CASE WHEN r.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')
            THEN TRY_STRPTIME(r.THRU_DT, '%Y%m%d') END) AS chemo_date,

        -- RT: specific HCPCS delivery codes OR revenue center 0330-0333
        -- (revenue center captures RT billed without a HCPCS delivery code, common in HOPD)
        MIN(CASE WHEN r.HCPCS_CD IN (
                '77402','77407','77412',
                '77385','77386',
                'G6015','G6016',
                '77373',
                '77520','77522','77523','77525',
                '77771','77772','77773')
             OR r.REV_CNTR IN ('0330','0331','0332','0333')
            THEN TRY_STRPTIME(r.THRU_DT, '%Y%m%d') END) AS rt_date,

        -- Last chemo / RT within the 12-month treatment window (course completion)
        MAX(CASE WHEN r.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')
            AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
            THEN TRY_STRPTIME(r.THRU_DT, '%Y%m%d') END) AS chemo_date_last,

        MAX(CASE WHEN (r.HCPCS_CD IN (
                '77402','77407','77412',
                '77385','77386',
                'G6015','G6016',
                '77373',
                '77520','77522','77523','77525',
                '77771','77772','77773')
             OR r.REV_CNTR IN ('0330','0331','0332','0333'))
            AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date AND o.first_hnc_date + INTERVAL 12 MONTH
            THEN TRY_STRPTIME(r.THRU_DT, '%Y%m%d') END) AS rt_date_last

    FROM out_revenuek_all r
    JOIN opscc_cohort o ON r.DSYSRTKY = o.DSYSRTKY
    GROUP BY r.DSYSRTKY
),

-- ─── 4. OUTPATIENT CLAIMS ─────────────────────────────────────────────────────
-- Extracts: metastatic/nodal dx only (diagnosis codes on claim header).
outc_scan AS (
    SELECT
        oc.DSYSRTKY,

        MAX(CASE WHEN len(list_filter(
            [oc.PRNCPAL_DGNS_CD,
             oc.ICD_DGNS_CD1,  oc.ICD_DGNS_CD2,  oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4,
             oc.ICD_DGNS_CD5,  oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
             oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10, oc.ICD_DGNS_CD11, oc.ICD_DGNS_CD12,
             oc.ICD_DGNS_CD13, oc.ICD_DGNS_CD14, oc.ICD_DGNS_CD15, oc.ICD_DGNS_CD16,
             oc.ICD_DGNS_CD17, oc.ICD_DGNS_CD18, oc.ICD_DGNS_CD19, oc.ICD_DGNS_CD20,
             oc.ICD_DGNS_CD21, oc.ICD_DGNS_CD22, oc.ICD_DGNS_CD23, oc.ICD_DGNS_CD24,
             oc.ICD_DGNS_CD25],
            x -> x LIKE 'C78%' OR x LIKE 'C79%'
        )) > 0
        AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
        THEN 1 ELSE 0 END) AS is_metastatic,

        MAX(CASE WHEN len(list_filter(
            [oc.PRNCPAL_DGNS_CD,
             oc.ICD_DGNS_CD1,  oc.ICD_DGNS_CD2,  oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4,
             oc.ICD_DGNS_CD5,  oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
             oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10, oc.ICD_DGNS_CD11, oc.ICD_DGNS_CD12,
             oc.ICD_DGNS_CD13, oc.ICD_DGNS_CD14, oc.ICD_DGNS_CD15, oc.ICD_DGNS_CD16,
             oc.ICD_DGNS_CD17, oc.ICD_DGNS_CD18, oc.ICD_DGNS_CD19, oc.ICD_DGNS_CD20,
             oc.ICD_DGNS_CD21, oc.ICD_DGNS_CD22, oc.ICD_DGNS_CD23, oc.ICD_DGNS_CD24,
             oc.ICD_DGNS_CD25],
            x -> x LIKE 'C77%'
        )) > 0
        AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY AND o.first_hnc_date + INTERVAL 90 DAY
        THEN 1 ELSE 0 END) AS is_nodal

    FROM out_claimsk_all oc
    JOIN opscc_cohort o ON oc.DSYSRTKY = o.DSYSRTKY
    GROUP BY oc.DSYSRTKY
),

-- ─── 5. Combine all sources ───────────────────────────────────────────────────
updates AS (
    SELECT
        o.DSYSRTKY,
        -- Earliest treatment date across all sources (list_min ignores NULLs)
        list_min([inp.tors_date,  car.tors_date,  out.tors_date])  AS first_tors_date,
        list_min([inp.chemo_date, car.chemo_date, out.chemo_date]) AS first_chemo_date,
        list_min([inp.rt_date,    car.rt_date,    out.rt_date])    AS first_rt_date,
        -- Latest treatment date across all sources (list_max ignores NULLs)
        list_max([inp.chemo_date_last, car.chemo_date_last, out.chemo_date_last]) AS last_chemo_date,
        list_max([inp.rt_date_last,    car.rt_date_last,    out.rt_date_last])    AS last_rt_date,
        -- Metastatic / nodal: TRUE if any source flagged it
        (COALESCE(inp.is_metastatic, 0) + COALESCE(car.is_metastatic, 0) + COALESCE(outc.is_metastatic, 0)) > 0 AS has_metastatic_dx,
        (COALESCE(inp.is_nodal,      0) + COALESCE(car.is_nodal,      0) + COALESCE(outc.is_nodal,      0)) > 0 AS has_nodal_dx
    FROM opscc_cohort o
    LEFT JOIN inp_scan  inp  ON o.DSYSRTKY = inp.DSYSRTKY
    LEFT JOIN car_scan  car  ON o.DSYSRTKY = car.DSYSRTKY
    LEFT JOIN out_scan  out  ON o.DSYSRTKY = out.DSYSRTKY
    LEFT JOIN outc_scan outc ON o.DSYSRTKY = outc.DSYSRTKY
)

UPDATE opscc_cohort
SET
    first_tors_date   = updates.first_tors_date,
    first_chemo_date  = updates.first_chemo_date,
    first_rt_date     = updates.first_rt_date,
    last_chemo_date   = updates.last_chemo_date,
    last_rt_date      = updates.last_rt_date,
    has_metastatic_dx = updates.has_metastatic_dx,
    has_nodal_dx      = updates.has_nodal_dx
FROM updates
WHERE opscc_cohort.DSYSRTKY = updates.DSYSRTKY;
