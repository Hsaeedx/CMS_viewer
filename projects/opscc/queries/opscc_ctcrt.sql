-- Annotate opscc_cohort with first TORS date, first CT/CRT date, and metastatic flag
-- Treatment dates are unconstrained; apply treatment window (e.g. 6 months) at analysis time
-- Metastatic flag: any C76/C77/C78/C79 diagnosis within 90 days of first_hnc_date

ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS first_tors_date DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS first_ctcrt_date DATE;
ALTER TABLE opscc_cohort ADD COLUMN IF NOT EXISTS has_metastatic_dx BOOLEAN;

WITH tors_claims AS (

    -- Inpatient TORS (ICD-10-PCS, any of 25 procedure slots)
    SELECT DISTINCT i.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT), '%Y%m%d') AS tx_date
    FROM inp_claimsk_all i
    JOIN opscc_cohort o ON i.DSYSRTKY = o.DSYSRTKY,
    UNNEST([
        i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
        i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
        i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
        i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
        i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
        i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
        i.ICD_PRCDR_CD25
    ]) AS t(code)
    WHERE code IN ('8E09XCZ','8E097CZ','8E090CZ','8E098CZ')

    UNION ALL

    -- Carrier TORS (CPT, professional billing)
    SELECT DISTINCT c.DSYSRTKY, TRY_STRPTIME(c.THRU_DT, '%Y%m%d') AS tx_date
    FROM car_linek_all c
    JOIN opscc_cohort o ON c.DSYSRTKY = o.DSYSRTKY
    WHERE c.HCPCS_CD IN ('1007190','42842','42844','42845')

    UNION ALL

    -- Outpatient TORS (CPT, hospital/ASC billing)
    SELECT DISTINCT r.DSYSRTKY, TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS tx_date
    FROM out_revenuek_all r
    JOIN opscc_cohort o ON r.DSYSRTKY = o.DSYSRTKY
    WHERE r.HCPCS_CD IN ('1007190','42842','42844','42845')
),

chemo_claims AS (

    SELECT DISTINCT c.DSYSRTKY, TRY_STRPTIME(c.THRU_DT,'%Y%m%d') AS tx_date
    FROM car_linek_all c
    JOIN opscc_cohort o ON c.DSYSRTKY = o.DSYSRTKY
    WHERE c.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')

    UNION ALL

    SELECT DISTINCT r.DSYSRTKY, TRY_STRPTIME(r.THRU_DT,'%Y%m%d') AS tx_date
    FROM out_revenuek_all r
    JOIN opscc_cohort o ON r.DSYSRTKY = o.DSYSRTKY
    WHERE r.HCPCS_CD IN ('J9060','J9045','J9190','J9171','J9201','J9055')

    UNION ALL

    SELECT DISTINCT i.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''),i.THRU_DT),'%Y%m%d') AS tx_date
    FROM inp_claimsk_all i
    JOIN opscc_cohort o ON i.DSYSRTKY = o.DSYSRTKY,
    UNNEST([
        i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
        i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
        i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
        i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
        i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
        i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
        i.ICD_PRCDR_CD25
    ]) AS t(code)
    WHERE code LIKE '3E0%'
),

rt_claims AS (

    SELECT DISTINCT c.DSYSRTKY, TRY_STRPTIME(c.THRU_DT,'%Y%m%d') AS tx_date
    FROM car_linek_all c
    JOIN opscc_cohort o ON c.DSYSRTKY = o.DSYSRTKY
    WHERE c.HCPCS_CD IN ('77402','77407','77412','77385','77386','77373')

    UNION ALL

    SELECT DISTINCT r.DSYSRTKY, TRY_STRPTIME(r.THRU_DT,'%Y%m%d') AS tx_date
    FROM out_revenuek_all r
    JOIN opscc_cohort o ON r.DSYSRTKY = o.DSYSRTKY
    WHERE r.HCPCS_CD IN ('77402','77407','77412','77385','77386','77373')

    UNION ALL

    SELECT DISTINCT i.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''),i.THRU_DT),'%Y%m%d') AS tx_date
    FROM inp_claimsk_all i
    JOIN opscc_cohort o ON i.DSYSRTKY = o.DSYSRTKY,
    UNNEST([
        i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
        i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
        i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
        i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
        i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
        i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
        i.ICD_PRCDR_CD25
    ]) AS t(code)
    WHERE (
              code LIKE 'D9%'   -- ENT
           OR code LIKE 'D7_3%' -- Lymphatics, Neck
           OR code LIKE 'DW_1%' -- Head and Neck (anatomic region)
          )
),

metastatic_claims AS (

    -- Inpatient: principal + all secondary diagnosis slots
    SELECT DISTINCT i.DSYSRTKY
    FROM inp_claimsk_all i
    JOIN opscc_cohort o ON i.DSYSRTKY = o.DSYSRTKY,
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
    WHERE (code LIKE 'C76%' OR code LIKE 'C77%' OR code LIKE 'C78%' OR code LIKE 'C79%')
      AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                  AND o.first_hnc_date + INTERVAL 90 DAY

    UNION ALL

    -- Carrier: one diagnosis code per line
    SELECT DISTINCT c.DSYSRTKY
    FROM car_linek_all c
    JOIN opscc_cohort o ON c.DSYSRTKY = o.DSYSRTKY
    WHERE (c.LINE_ICD_DGNS_CD LIKE 'C76%' OR c.LINE_ICD_DGNS_CD LIKE 'C77%'
        OR c.LINE_ICD_DGNS_CD LIKE 'C78%' OR c.LINE_ICD_DGNS_CD LIKE 'C79%')
      AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                  AND o.first_hnc_date + INTERVAL 90 DAY

    UNION ALL

    -- Outpatient: principal + all secondary diagnosis slots
    SELECT DISTINCT oc.DSYSRTKY
    FROM out_claimsk_all oc
    JOIN opscc_cohort o ON oc.DSYSRTKY = o.DSYSRTKY,
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
    WHERE (code LIKE 'C76%' OR code LIKE 'C77%' OR code LIKE 'C78%' OR code LIKE 'C79%')
      AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
              BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                  AND o.first_hnc_date + INTERVAL 90 DAY
),

first_tors AS (
    SELECT DSYSRTKY, MIN(tx_date) AS first_tors_date
    FROM tors_claims
    GROUP BY DSYSRTKY
),

first_chemo AS (
    SELECT DSYSRTKY, MIN(tx_date) AS first_chemo_date
    FROM chemo_claims
    GROUP BY DSYSRTKY
),

first_rt AS (
    SELECT DSYSRTKY, MIN(tx_date) AS first_rt_date
    FROM rt_claims
    GROUP BY DSYSRTKY
),

updates AS (
    SELECT
        o.DSYSRTKY,
        ft.first_tors_date,
        LEAST(
            COALESCE(fc.first_chemo_date, fr.first_rt_date),
            COALESCE(fr.first_rt_date,    fc.first_chemo_date)
        ) AS first_ctcrt_date,
        CASE WHEN m.DSYSRTKY IS NOT NULL THEN TRUE ELSE FALSE END AS has_metastatic_dx
    FROM opscc_cohort o
    LEFT JOIN first_tors  ft ON o.DSYSRTKY = ft.DSYSRTKY
    LEFT JOIN first_chemo fc ON o.DSYSRTKY = fc.DSYSRTKY
    LEFT JOIN first_rt    fr ON o.DSYSRTKY = fr.DSYSRTKY
    LEFT JOIN (SELECT DISTINCT DSYSRTKY FROM metastatic_claims) m ON o.DSYSRTKY = m.DSYSRTKY
)

UPDATE opscc_cohort
SET
    first_tors_date   = updates.first_tors_date,
    first_ctcrt_date  = updates.first_ctcrt_date,
    has_metastatic_dx = updates.has_metastatic_dx
FROM updates
WHERE opscc_cohort.DSYSRTKY = updates.DSYSRTKY;
