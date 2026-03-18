-- Step 7: Curative-intent therapy before last IO episode start
-- Curative intent = surgery OR radiation (platinum removed per attending)
-- Uses pre-filtered staging tables (io_car_lines, io_out_revenue, io_inp_claims)
--   instead of scanning full large tables — see 07a_io_staging.sql
-- Pattern adapted from: opscc/queries/opscc_ctcrt.sql
-- Output: io_curative

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_curative;

CREATE TABLE io_curative AS

WITH hnc_io_episodes AS (
    -- HNC+IO patients with their last IO episode start date
    SELECT e.DSYSRTKY, e.last_io_episode_start
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
),

carrier_curative AS (
    -- Carrier line: radiation and surgery CPTs only (platinum removed)
    SELECT
        c.DSYSRTKY,
        TRY_STRPTIME(c.THRU_DT,'%Y%m%d') AS therapy_date,
        CASE
            WHEN (c.HCPCS_CD BETWEEN '77401' AND '77425')
                 OR c.HCPCS_CD = '77427'
                 OR (c.HCPCS_CD BETWEEN '77431' AND '77499')
            THEN 'radiation'
            ELSE 'surgery'
        END AS therapy_type
    FROM io_car_lines c
    JOIN hnc_io_episodes e ON c.DSYSRTKY = e.DSYSRTKY
    WHERE (
        (c.HCPCS_CD BETWEEN '77401' AND '77425')
        OR c.HCPCS_CD = '77427'
        OR (c.HCPCS_CD BETWEEN '77431' AND '77499')
        OR c.HCPCS_CD IN (
            '41110','41112','41113','41114','41116',
            '41120','41130','41135','41140','41145','41150','41153','41155',
            '21030','21032','21034','21044','21045','21046','21047','21048','21049',
            '42104','42106','42107','42120','42140','42145',
            '42410','42415','42420','42425','42426',
            '42842','42844','42845','42890','42892','42894',
            '31300','31360','31365','31367','31368',
            '31370','31375','31380','31382','31390','31395',
            '31540','31541','31546','31551','31561',
            '38700','38720','38724','1007190'
        )
    )
    AND TRY_STRPTIME(c.THRU_DT,'%Y%m%d') < e.last_io_episode_start
),

revenue_curative AS (
    -- Outpatient revenue: radiation and surgery CPTs only (platinum removed)
    SELECT
        r.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT),'%Y%m%d') AS therapy_date,
        CASE
            WHEN (r.HCPCS_CD BETWEEN '77401' AND '77425')
                 OR r.HCPCS_CD = '77427'
                 OR (r.HCPCS_CD BETWEEN '77431' AND '77499')
            THEN 'radiation'
            ELSE 'surgery'
        END AS therapy_type
    FROM io_out_revenue r
    JOIN hnc_io_episodes e ON r.DSYSRTKY = e.DSYSRTKY
    WHERE (
        (r.HCPCS_CD BETWEEN '77401' AND '77425')
        OR r.HCPCS_CD = '77427'
        OR (r.HCPCS_CD BETWEEN '77431' AND '77499')
        OR r.HCPCS_CD IN (
            '41110','41112','41113','41114','41116',
            '41120','41130','41135','41140','41145','41150','41153','41155',
            '21030','21032','21034','21044','21045','21046','21047','21048','21049',
            '42104','42106','42107','42120','42140','42145',
            '42410','42415','42420','42425','42426',
            '42842','42844','42845','42890','42892','42894',
            '31300','31360','31365','31367','31368',
            '31370','31375','31380','31382','31390','31395',
            '31540','31541','31546','31551','31561',
            '38700','38720','38724','1007190'
        )
    )
    AND TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT),'%Y%m%d') < e.last_io_episode_start
),

inpatient_curative AS (
    -- Inpatient ICD-10-PCS: radiation (D9%) AND surgical resections
    SELECT DISTINCT
        i.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT),'%Y%m%d') AS therapy_date,
        CASE
            WHEN code LIKE 'D9%' OR code LIKE 'DW%' THEN 'radiation'
            ELSE 'surgery'
        END AS therapy_type
    FROM io_inp_claims i
    JOIN hnc_io_episodes e ON i.DSYSRTKY = e.DSYSRTKY,
    UNNEST([
        i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
        i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
        i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
        i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
        i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
        i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
        i.ICD_PRCDR_CD25
    ]) AS t(code)
    -- PCS code list injected from codes.json by io_pipeline.py
    WHERE code IN ({INPATIENT_PCS_WHERE})
    AND TRY_STRPTIME(COALESCE(NULLIF(i.PRCDR_DT1,''), i.THRU_DT),'%Y%m%d') < e.last_io_episode_start
),

all_curative AS (
    SELECT * FROM carrier_curative
    UNION ALL
    SELECT * FROM revenue_curative
    UNION ALL
    SELECT * FROM inpatient_curative
)

SELECT
    DSYSRTKY,
    MIN(therapy_date) AS curative_therapy_date,
    MODE() WITHIN GROUP (ORDER BY therapy_type) AS primary_curative_type,
    BOOL_OR(therapy_type = 'surgery')   AS had_surgery,
    BOOL_OR(therapy_type = 'radiation') AS had_radiation

FROM all_curative
WHERE therapy_date IS NOT NULL
GROUP BY DSYSRTKY;
