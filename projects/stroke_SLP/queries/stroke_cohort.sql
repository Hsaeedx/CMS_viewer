-- stroke_cohort.sql
-- First acute stroke inpatient admission per patient
-- with 6-month continuous FFS enrollment before admission.
--
-- Acute stroke (principal diagnosis, LEFT(code,3)):
--   I60  Subarachnoid hemorrhage
--   I61  Intracerebral hemorrhage
--   I63  Ischemic stroke
--   I64  Unspecified stroke
--
-- Exclusions:
--   I69.x / G45.x as principal (sequelae or TIA, not acute stroke)
--   No 6-month continuous FFS in months before admission
--
-- Output table: stroke_cohort

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

-- ── Step 1: All acute stroke admissions ───────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _stroke_raw AS
SELECT
    i.DSYSRTKY,
    TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')  AS adm_date,
    TRY_STRPTIME(i.DSCHRGDT, '%Y%m%d')  AS dschg_date,
    i.PRNCPAL_DGNS_CD,
    CASE LEFT(i.PRNCPAL_DGNS_CD, 3)
        WHEN 'I60' THEN 'SAH'
        WHEN 'I61' THEN 'ICH'
        WHEN 'I63' THEN 'Ischemic'
        WHEN 'I64' THEN 'Unspecified'
    END AS stroke_type,
    i.DRG_CD,
    TRY_CAST(i.PMT_AMT  AS DOUBLE) AS index_pmt,
    TRY_CAST(i.TOT_CHRG AS DOUBLE) AS index_chrg,
    i.STUS_CD   AS dschg_status,
    i.SRC_ADMS  AS adm_source,
    DATEDIFF('day',
        TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d'),
        TRY_STRPTIME(i.DSCHRGDT, '%Y%m%d')) AS index_los
FROM inp_claimsk_all i
WHERE LEFT(i.PRNCPAL_DGNS_CD, 3) IN ('I60', 'I61', 'I63', 'I64')
  AND i.ADMSN_DT IS NOT NULL
  AND i.ADMSN_DT != '';

-- ── Step 2: First admission per patient ───────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _stroke_first AS
SELECT
    s.DSYSRTKY,
    MIN(s.adm_date) AS first_adm_date
FROM _stroke_raw s
GROUP BY s.DSYSRTKY;

-- ── Step 3: Dysphagia and aspiration POA flags on the index admission ─────────
-- CLM_POA_IND_SW1-25 correspond positionally to ICD_DGNS_CD1-25.
-- Two parallel unnest() calls in a SELECT subquery pair codes with POA flags.

CREATE OR REPLACE TEMP TABLE _poa_flags AS
SELECT
    DSYSRTKY,
    first_adm_date,
    MAX(CASE WHEN LEFT(code, 4) = 'R131' AND poa = 'Y' THEN 1 ELSE 0 END) AS dysphagia_poa,
    MAX(CASE WHEN code IN ('J690', 'J698') AND poa = 'Y' THEN 1 ELSE 0 END) AS aspiration_poa
FROM (
    SELECT
        r.DSYSRTKY,
        r.first_adm_date,
        unnest([i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
                i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
                i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
                i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15, i.ICD_DGNS_CD16,
                i.ICD_DGNS_CD17, i.ICD_DGNS_CD18, i.ICD_DGNS_CD19, i.ICD_DGNS_CD20,
                i.ICD_DGNS_CD21, i.ICD_DGNS_CD22, i.ICD_DGNS_CD23, i.ICD_DGNS_CD24,
                i.ICD_DGNS_CD25]) AS code,
        unnest([i.CLM_POA_IND_SW1,  i.CLM_POA_IND_SW2,  i.CLM_POA_IND_SW3,  i.CLM_POA_IND_SW4,
                i.CLM_POA_IND_SW5,  i.CLM_POA_IND_SW6,  i.CLM_POA_IND_SW7,  i.CLM_POA_IND_SW8,
                i.CLM_POA_IND_SW9,  i.CLM_POA_IND_SW10, i.CLM_POA_IND_SW11, i.CLM_POA_IND_SW12,
                i.CLM_POA_IND_SW13, i.CLM_POA_IND_SW14, i.CLM_POA_IND_SW15, i.CLM_POA_IND_SW16,
                i.CLM_POA_IND_SW17, i.CLM_POA_IND_SW18, i.CLM_POA_IND_SW19, i.CLM_POA_IND_SW20,
                i.CLM_POA_IND_SW21, i.CLM_POA_IND_SW22, i.CLM_POA_IND_SW23, i.CLM_POA_IND_SW24,
                i.CLM_POA_IND_SW25]) AS poa
    FROM _stroke_first r
    JOIN inp_claimsk_all i
      ON i.DSYSRTKY = r.DSYSRTKY
     AND TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') = r.first_adm_date
) sub
GROUP BY DSYSRTKY, first_adm_date;

-- ── Step 4: Procedure flags from the index admission ─────────────────────────
-- ICD-10-PCS codes present on the same claim as the index stroke admission.
-- mech_vent  : 5A1935Z / 5A1945Z / 5A1955Z
-- peg_placed : 0DH63UZ / 0DH64UZ / 0DH65UZ
-- trach_placed: any code starting with 0B11

CREATE OR REPLACE TEMP TABLE _proc_flags AS
SELECT
    r.DSYSRTKY,
    r.first_adm_date,
    MAX(CASE WHEN
        i.ICD_PRCDR_CD1  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD2  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD3  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD4  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD5  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD6  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD7  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD8  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD9  IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD10 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD11 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD12 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD13 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD14 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD15 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD16 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD17 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD18 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD19 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD20 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD21 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD22 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD23 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD24 IN ('5A1935Z','5A1945Z','5A1955Z') OR
        i.ICD_PRCDR_CD25 IN ('5A1935Z','5A1945Z','5A1955Z')
    THEN 1 ELSE 0 END) AS mech_vent,

    MAX(CASE WHEN
        i.ICD_PRCDR_CD1  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD2  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD3  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD4  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD5  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD6  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD7  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD8  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD9  IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD10 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD11 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD12 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD13 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD14 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD15 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD16 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD17 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD18 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD19 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD20 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD21 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD22 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD23 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD24 IN ('0DH63UZ','0DH64UZ','0DH65UZ') OR
        i.ICD_PRCDR_CD25 IN ('0DH63UZ','0DH64UZ','0DH65UZ')
    THEN 1 ELSE 0 END) AS peg_placed,

    MAX(CASE WHEN
        LEFT(i.ICD_PRCDR_CD1,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD2,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD3,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD4,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD5,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD6,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD7,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD8,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD9,  4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD10, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD11, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD12, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD13, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD14, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD15, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD16, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD17, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD18, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD19, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD20, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD21, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD22, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD23, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD24, 4) = '0B11' OR
        LEFT(i.ICD_PRCDR_CD25, 4) = '0B11'
    THEN 1 ELSE 0 END) AS trach_placed

FROM _stroke_first r
JOIN inp_claimsk_all i
  ON i.DSYSRTKY = r.DSYSRTKY
 AND TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') = r.first_adm_date
GROUP BY r.DSYSRTKY, r.first_adm_date;

-- ── Step 5: FFS enrollment — 6 months before admission ───────────────────────
-- Reuse same HMOIND + MDCR_STUS_CD logic as opscc_cohort.sql

CREATE OR REPLACE TEMP TABLE _enroll AS
SELECT
    m.DSYSRTKY,
    make_date(CAST(m.RFRNC_YR AS INTEGER), mth, 1) AS month_start,
    CASE WHEN
        (CASE mth
            WHEN 1  THEN m.HMOIND1  WHEN 2  THEN m.HMOIND2
            WHEN 3  THEN m.HMOIND3  WHEN 4  THEN m.HMOIND4
            WHEN 5  THEN m.HMOIND5  WHEN 6  THEN m.HMOIND6
            WHEN 7  THEN m.HMOIND7  WHEN 8  THEN m.HMOIND8
            WHEN 9  THEN m.HMOIND9  WHEN 10 THEN m.HMOIND10
            WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
        END) = '0'
        AND
        (CASE mth
            WHEN 1  THEN m.MDCR_STUS_CD_01 WHEN 2  THEN m.MDCR_STUS_CD_02
            WHEN 3  THEN m.MDCR_STUS_CD_03 WHEN 4  THEN m.MDCR_STUS_CD_04
            WHEN 5  THEN m.MDCR_STUS_CD_05 WHEN 6  THEN m.MDCR_STUS_CD_06
            WHEN 7  THEN m.MDCR_STUS_CD_07 WHEN 8  THEN m.MDCR_STUS_CD_08
            WHEN 9  THEN m.MDCR_STUS_CD_09 WHEN 10 THEN m.MDCR_STUS_CD_10
            WHEN 11 THEN m.MDCR_STUS_CD_11 WHEN 12 THEN m.MDCR_STUS_CD_12
        END) IN ('10', '11', '20', '21')
    THEN 1 ELSE 0 END AS ffs_month
FROM mbsf_all m
JOIN _stroke_first sf ON m.DSYSRTKY = sf.DSYSRTKY,
UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mth);

CREATE OR REPLACE TEMP TABLE _enrolled AS
SELECT sf.DSYSRTKY
FROM _stroke_first sf
JOIN _enroll e ON e.DSYSRTKY = sf.DSYSRTKY
  AND e.month_start BETWEEN
      date_trunc('month', sf.first_adm_date) - INTERVAL 5 MONTH
      AND date_trunc('month', sf.first_adm_date)
GROUP BY sf.DSYSRTKY
HAVING SUM(e.ffs_month) = 6;

-- ── Step 6: Demographics from MBSF ───────────────────────────────────────────

CREATE OR REPLACE TEMP TABLE _demo AS
SELECT
    m.DSYSRTKY,
    CAST(m.AGE AS INTEGER) AS age_at_adm,
    CASE m.SEX
        WHEN '1' THEN 'Male'
        WHEN '2' THEN 'Female'
        ELSE 'Unknown'
    END AS sex,
    CASE m.RACE
        WHEN '1' THEN 'White'
        WHEN '2' THEN 'Black'
        WHEN '4' THEN 'Asian/PI'
        WHEN '5' THEN 'Hispanic'
        WHEN '6' THEN 'AIAN'
        WHEN '3' THEN 'Other'
        ELSE 'Unknown'
    END AS race
FROM mbsf_all m
JOIN _stroke_first sf ON m.DSYSRTKY = sf.DSYSRTKY
  AND CAST(m.RFRNC_YR AS INTEGER) = YEAR(sf.first_adm_date);

-- ── Step 7: Dedup: one row per (DSYSRTKY, adm_date) — keep longest LOS, then highest payment
-- Handles patients with multiple inpatient claims on the same date (e.g. transfers)

CREATE OR REPLACE TEMP TABLE _stroke_dedup AS
SELECT * EXCLUDE (rn)
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY DSYSRTKY, adm_date
            ORDER BY index_los DESC NULLS LAST,
                     index_pmt DESC NULLS LAST
        ) AS rn
    FROM _stroke_raw
)
WHERE rn = 1;

-- ── Step 8: Exclusions — hospice discharge and 30-day post-discharge death ────
-- Hospice discharge: dschg_status 50 (hospice home) or 51 (hospice medical).
-- 30-day mortality: patient died within 30 days of the index discharge date.
-- Death date sourced from MBSF DEATH_DT (format YYYYMMDD).

CREATE OR REPLACE TEMP TABLE _death AS
SELECT DSYSRTKY, MAX(TRY_STRPTIME(DEATH_DT, '%Y%m%d')) AS death_date
FROM mbsf_all
WHERE DEATH_DT IS NOT NULL AND DEATH_DT != ''
GROUP BY DSYSRTKY;

CREATE OR REPLACE TEMP TABLE _eligible AS
SELECT sf.DSYSRTKY
FROM _stroke_first sf
JOIN _stroke_dedup s ON s.DSYSRTKY = sf.DSYSRTKY
                    AND s.adm_date = sf.first_adm_date
LEFT JOIN _death   d ON d.DSYSRTKY = sf.DSYSRTKY
WHERE s.dschg_status NOT IN ('50', '51')          -- exclude hospice discharge
  AND (d.death_date IS NULL
       OR d.death_date > s.dschg_date + INTERVAL 30 DAY);  -- exclude 30-day post-discharge death

-- ── Final: Build stroke_cohort ────────────────────────────────────────────────

DROP TABLE IF EXISTS stroke_cohort;

CREATE TABLE stroke_cohort AS
SELECT
    sf.DSYSRTKY,
    sf.first_adm_date       AS index_adm_date,
    s.dschg_date            AS index_dschg_date,
    s.index_los,
    s.stroke_type,
    s.dschg_status,
    s.drg_cd,
    s.adm_source,
    s.index_pmt,
    s.index_chrg,
    COALESCE(p.dysphagia_poa,  0) AS dysphagia_poa,
    COALESCE(p.aspiration_poa, 0) AS aspiration_poa,
    COALESCE(pr.mech_vent,     0) AS mech_vent,
    COALESCE(pr.peg_placed,    0) AS peg_placed,
    COALESCE(pr.trach_placed,  0) AS trach_placed,
    d.age_at_adm,
    d.sex,
    d.race
FROM _stroke_first sf
JOIN _enrolled        en ON en.DSYSRTKY = sf.DSYSRTKY
JOIN _eligible        el ON el.DSYSRTKY = sf.DSYSRTKY
JOIN _stroke_dedup     s ON  s.DSYSRTKY = sf.DSYSRTKY
                         AND s.adm_date = sf.first_adm_date
LEFT JOIN _poa_flags   p ON  p.DSYSRTKY = sf.DSYSRTKY
LEFT JOIN _proc_flags pr ON pr.DSYSRTKY = sf.DSYSRTKY
LEFT JOIN _demo        d ON  d.DSYSRTKY = sf.DSYSRTKY;

-- ── Summary ───────────────────────────────────────────────────────────────────
SELECT
    stroke_type,
    COUNT(*)                                AS n,
    ROUND(AVG(age_at_adm), 1)              AS mean_age,
    COUNT(CASE WHEN sex = 'Male' THEN 1 END) AS n_male,
    SUM(dysphagia_poa)                      AS n_dysphagia_poa,
    SUM(aspiration_poa)                     AS n_aspiration_poa,
    ROUND(AVG(index_los), 1)               AS mean_los,
    ROUND(AVG(index_pmt), 0)               AS mean_pmt
FROM stroke_cohort
GROUP BY stroke_type
ORDER BY stroke_type;
