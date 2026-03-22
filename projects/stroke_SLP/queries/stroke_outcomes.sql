-- stroke_outcomes.sql
-- Post-discharge outcomes for stroke_cohort patients.
-- Follow-up starts at index_dschg_date; outcomes occurring during the index
-- hospitalization are flagged separately (index_* fields) but excluded from
-- the post-discharge outcome columns.
--
-- Two-step approach (same as opscc_outcomes.sql):
--   Step 1 — Extract cohort patients' post-discharge claims into temp tables
--   Step 2 — Look up outcome codes on the small temp tables
--
-- Outcomes:
--   Mortality          — from MBSF DEATH_DT
--   All-cause readmit  — any inpatient admission after index discharge
--   Recurrent stroke   — inpatient admission with I60/I61/I63/I64 as principal dx
--   Aspiration PNA     — J690, J698 (inp + out + car only; SNF/HHA excluded)  [primary]
--   All Pneumonia      — J09–J18, J690, J698 (inp + out + car only; SNF/HHA excluded)  [secondary; breakdown by 3-char code]
--   Dysphagia          — LEFT(code,4)='R131' (inp + out + car)
--   G-tube             — Z931, CPT 43246/49440, HCPCS B4087/B4088, PCS 0DH63UZ/0DH60UZ
--   SNF placement      — any SNF admission within 30d of discharge
--   Home health use    — any HHA claim within 90d of discharge
--   Total Medicare cost — sum of payments across all claim types in 365d
--
-- Output table: stroke_outcomes

SET memory_limit='24GB';
SET threads=12;
-- temp_directory set by run_pipeline.py via SET temp_directory

-- ── STEP 1: Extract post-discharge claims (one scan per large table) ──────────

-- Inpatient (readmissions and recurrent stroke)
CREATE OR REPLACE TEMP TABLE _inp AS
SELECT
    c.DSYSRTKY,
    c.index_dschg_date,
    TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')  AS adm_date,
    TRY_STRPTIME(i.THRU_DT,  '%Y%m%d')  AS thru_date,
    i.PRNCPAL_DGNS_CD,
    i.ADMTG_DGNS_CD,
    i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
    i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
    i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
    i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15,
    i.ICD_PRCDR_CD1, i.ICD_PRCDR_CD2, i.ICD_PRCDR_CD3, i.ICD_PRCDR_CD4,
    i.ICD_PRCDR_CD5, i.ICD_PRCDR_CD6, i.ICD_PRCDR_CD7, i.ICD_PRCDR_CD8,
    TRY_CAST(i.PMT_AMT AS DOUBLE) AS PMT_AMT
FROM stroke_cohort c
JOIN inp_claimsk_all i ON i.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY;

-- Outpatient (dysphagia, aspiration PNA, G-tube dx)
CREATE OR REPLACE TEMP TABLE _out AS
SELECT
    c.DSYSRTKY,
    c.index_dschg_date,
    TRY_STRPTIME(o.THRU_DT, '%Y%m%d') AS thru_date,
    o.PRNCPAL_DGNS_CD,
    o.ICD_DGNS_CD1,  o.ICD_DGNS_CD2,  o.ICD_DGNS_CD3,  o.ICD_DGNS_CD4,
    o.ICD_DGNS_CD5,  o.ICD_DGNS_CD6,  o.ICD_DGNS_CD7,  o.ICD_DGNS_CD8,
    o.ICD_DGNS_CD9,  o.ICD_DGNS_CD10,
    TRY_CAST(o.PMT_AMT AS DOUBLE) AS PMT_AMT
FROM stroke_cohort c
JOIN out_claimsk_all o ON o.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(o.THRU_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(o.THRU_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY;

-- Carrier (dysphagia, aspiration PNA, G-tube CPT/HCPCS)
CREATE OR REPLACE TEMP TABLE _car AS
SELECT
    c.DSYSRTKY,
    c.index_dschg_date,
    TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') AS thru_date,
    cl.LINE_ICD_DGNS_CD,
    cl.HCPCS_CD,
    TRY_CAST(cl.LINEPMT AS DOUBLE) AS PMT_AMT
FROM stroke_cohort c
JOIN car_linek_all cl ON cl.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY
  AND (LEFT(cl.LINE_ICD_DGNS_CD, 4) IN ('R131', 'Z931', 'J690', 'J698')
    OR LEFT(cl.LINE_ICD_DGNS_CD, 3) IN ('J09','J10','J11','J12','J13','J14','J15','J16','J17','J18')
    OR cl.HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088'));

-- Outpatient revenue center (G-tube HCPCS)
CREATE OR REPLACE TEMP TABLE _rev AS
SELECT
    c.DSYSRTKY,
    c.index_dschg_date,
    TRY_STRPTIME(r.THRU_DT, '%Y%m%d') AS thru_date,
    r.HCPCS_CD,
    TRY_CAST(r.REVPMT AS DOUBLE) AS PMT_AMT
FROM stroke_cohort c
JOIN out_revenuek_all r ON r.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(r.THRU_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY
  AND r.HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088');

-- SNF (skilled nursing facility placement within 30d)
CREATE OR REPLACE TEMP TABLE _snf AS
SELECT
    c.DSYSRTKY,
    MIN(TRY_STRPTIME(s.ADMSN_DT, '%Y%m%d')) AS first_snf_date,
    SUM(TRY_CAST(s.PMT_AMT AS DOUBLE)) AS snf_pmt_365d
FROM stroke_cohort c
JOIN snf_claimsk_all s ON s.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(s.ADMSN_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(s.ADMSN_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY
GROUP BY c.DSYSRTKY;

-- Home health (any HHA claim within 90d)
CREATE OR REPLACE TEMP TABLE _hha AS
SELECT
    c.DSYSRTKY,
    MIN(TRY_STRPTIME(h.THRU_DT, '%Y%m%d')) AS first_hha_date,
    SUM(TRY_CAST(h.PMT_AMT AS DOUBLE)) AS hha_pmt_365d
FROM stroke_cohort c
JOIN hha_claimsk_all h ON h.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(h.THRU_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(h.THRU_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY
GROUP BY c.DSYSRTKY;

-- ── STEP 2: Outcome lookups on small temp tables ──────────────────────────────

-- Recurrent stroke (inpatient, principal dx)
CREATE OR REPLACE TEMP TABLE _recur_stroke AS
SELECT
    DSYSRTKY,
    MIN(adm_date) AS first_recur_stroke_date
FROM _inp
WHERE LEFT(PRNCPAL_DGNS_CD, 3) IN ('I60', 'I61', 'I63', 'I64')
GROUP BY DSYSRTKY;

-- All-cause readmission
CREATE OR REPLACE TEMP TABLE _readmit AS
SELECT
    DSYSRTKY,
    MIN(adm_date)  AS first_readmit_date,
    COUNT(DISTINCT adm_date) AS n_readmissions_365d
FROM _inp
GROUP BY DSYSRTKY;

-- Aspiration pneumonia (all sources, post-discharge)
CREATE OR REPLACE TEMP TABLE _aspiration AS
SELECT DSYSRTKY, MIN(thru_date) AS first_aspiration_date FROM (
    SELECT DSYSRTKY, thru_date FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,  ICD_DGNS_CD4,
            ICD_DGNS_CD5,  ICD_DGNS_CD6,  ICD_DGNS_CD7,  ICD_DGNS_CD8,
            ICD_DGNS_CD9,  ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE code IN ('J690', 'J698')
    UNION ALL
    SELECT DSYSRTKY, thru_date FROM _out,
    UNNEST([PRNCPAL_DGNS_CD, ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7,
            ICD_DGNS_CD8,  ICD_DGNS_CD9, ICD_DGNS_CD10]) AS t(code)
    WHERE code IN ('J690', 'J698')
    UNION ALL
    SELECT DSYSRTKY, thru_date FROM _car
    WHERE LINE_ICD_DGNS_CD IN ('J690', 'J698')
) GROUP BY DSYSRTKY;

-- All pneumonia (J09-J18 + J69, any source) — secondary outcome with code breakdown
-- Also captures the 3-character ICD code of the first qualifying event for breakdown analysis.
CREATE OR REPLACE TEMP TABLE _pneumonia_all AS
SELECT
    DSYSRTKY,
    MIN(thru_date)    AS first_pneumonia_date,
    -- 3-char code of earliest event (for breakdown table)
    FIRST(pna_code ORDER BY thru_date) AS first_pneumonia_code
FROM (
    SELECT DSYSRTKY, thru_date,
           LEFT(code, 3) AS pna_code
    FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,  ICD_DGNS_CD4,
            ICD_DGNS_CD5,  ICD_DGNS_CD6,  ICD_DGNS_CD7,  ICD_DGNS_CD8,
            ICD_DGNS_CD9,  ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE LEFT(code, 3) IN ('J09','J10','J11','J12','J13','J14','J15','J16','J17','J18','J69')
    UNION ALL
    SELECT DSYSRTKY, thru_date,
           LEFT(code, 3) AS pna_code
    FROM _out,
    UNNEST([PRNCPAL_DGNS_CD, ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7,
            ICD_DGNS_CD8,  ICD_DGNS_CD9, ICD_DGNS_CD10]) AS t(code)
    WHERE LEFT(code, 3) IN ('J09','J10','J11','J12','J13','J14','J15','J16','J17','J18','J69')
    UNION ALL
    SELECT DSYSRTKY, thru_date,
           LEFT(LINE_ICD_DGNS_CD, 3) AS pna_code
    FROM _car
    WHERE LEFT(LINE_ICD_DGNS_CD, 3) IN ('J09','J10','J11','J12','J13','J14','J15','J16','J17','J18','J69')
) GROUP BY DSYSRTKY;

-- Dysphagia diagnosis (prefix R131, all sources)
CREATE OR REPLACE TEMP TABLE _dysphagia AS
SELECT DSYSRTKY, MIN(thru_date) AS first_dysphagia_date FROM (
    SELECT DSYSRTKY, thru_date FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,  ICD_DGNS_CD4,
            ICD_DGNS_CD5,  ICD_DGNS_CD6,  ICD_DGNS_CD7,  ICD_DGNS_CD8,
            ICD_DGNS_CD9,  ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE LEFT(code, 4) = 'R131'
    UNION ALL
    SELECT DSYSRTKY, thru_date FROM _out,
    UNNEST([PRNCPAL_DGNS_CD, ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7,
            ICD_DGNS_CD8,  ICD_DGNS_CD9, ICD_DGNS_CD10]) AS t(code)
    WHERE LEFT(code, 4) = 'R131'
    UNION ALL
    SELECT DSYSRTKY, thru_date FROM _car
    WHERE LEFT(LINE_ICD_DGNS_CD, 4) = 'R131'
) GROUP BY DSYSRTKY;

-- G-tube (dx Z931 + CPT/HCPCS + ICD-10-PCS, all sources)
CREATE OR REPLACE TEMP TABLE _gtube AS
SELECT DSYSRTKY, MIN(thru_date) AS first_gtube_date FROM (
    -- Inpatient dx Z931
    SELECT DSYSRTKY, thru_date FROM _inp,
    UNNEST([PRNCPAL_DGNS_CD, ADMTG_DGNS_CD,
            ICD_DGNS_CD1,  ICD_DGNS_CD2,  ICD_DGNS_CD3,  ICD_DGNS_CD4,
            ICD_DGNS_CD5,  ICD_DGNS_CD6,  ICD_DGNS_CD7,  ICD_DGNS_CD8,
            ICD_DGNS_CD9,  ICD_DGNS_CD10, ICD_DGNS_CD11, ICD_DGNS_CD12,
            ICD_DGNS_CD13, ICD_DGNS_CD14, ICD_DGNS_CD15]) AS t(code)
    WHERE LEFT(code, 4) = 'Z931'
    UNION ALL
    -- Inpatient ICD-10-PCS procedure
    SELECT DSYSRTKY, thru_date FROM _inp,
    UNNEST([ICD_PRCDR_CD1, ICD_PRCDR_CD2, ICD_PRCDR_CD3, ICD_PRCDR_CD4,
            ICD_PRCDR_CD5, ICD_PRCDR_CD6, ICD_PRCDR_CD7, ICD_PRCDR_CD8]) AS t(code)
    WHERE code IN ('0DH63UZ', '0DH60UZ')
    UNION ALL
    -- Outpatient dx Z931
    SELECT DSYSRTKY, thru_date FROM _out,
    UNNEST([PRNCPAL_DGNS_CD, ICD_DGNS_CD1, ICD_DGNS_CD2, ICD_DGNS_CD3,
            ICD_DGNS_CD4,  ICD_DGNS_CD5, ICD_DGNS_CD6, ICD_DGNS_CD7,
            ICD_DGNS_CD8,  ICD_DGNS_CD9, ICD_DGNS_CD10]) AS t(code)
    WHERE LEFT(code, 4) = 'Z931'
    UNION ALL
    -- Carrier CPT/HCPCS
    SELECT DSYSRTKY, thru_date FROM _car
    WHERE HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088')
    UNION ALL
    -- Revenue center HCPCS
    SELECT DSYSRTKY, thru_date FROM _rev
    WHERE HCPCS_CD IN ('43246', '49440', 'B4087', 'B4088')
) GROUP BY DSYSRTKY;

-- Pre-stroke tube: any tube placement or supply claim in 730 days before index admission
-- Used to exclude pre-existing tube patients from the PEG/G-tube outcome model.
-- Structured as pre-filtered temp tables (not correlated subqueries) for performance.

-- Step A: carrier tube codes in any cohort patient's history
CREATE OR REPLACE TEMP TABLE _pre_car_tube AS
SELECT DISTINCT cl.DSYSRTKY
FROM car_linek_all cl
JOIN stroke_cohort c ON cl.DSYSRTKY = c.DSYSRTKY
WHERE (   cl.HCPCS_CD IN ('43246','43750','44500','44372','74355','74350',
                           '49440','49441','B4087','B4088')
       OR cl.HCPCS_CD BETWEEN 'B4034' AND 'B4036'
       OR cl.HCPCS_CD BETWEEN 'B4081' AND 'B4088'
       OR cl.HCPCS_CD = 'B4100'
       OR cl.HCPCS_CD BETWEEN 'B4102' AND 'B4104'
       OR cl.HCPCS_CD BETWEEN 'B4149' AND 'B4162')
  AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') < c.index_adm_date
  AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') >= c.index_adm_date - INTERVAL 730 DAY;

-- Step B: inpatient Z931 diagnosis or gastrostomy PCS procedure in prior admissions
CREATE OR REPLACE TEMP TABLE _pre_inp_tube AS
SELECT DISTINCT i.DSYSRTKY
FROM inp_claimsk_all i
JOIN stroke_cohort c ON i.DSYSRTKY = c.DSYSRTKY,
UNNEST([i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
        i.ICD_DGNS_CD1, i.ICD_DGNS_CD2, i.ICD_DGNS_CD3,
        i.ICD_DGNS_CD4, i.ICD_DGNS_CD5,
        i.ICD_PRCDR_CD1, i.ICD_PRCDR_CD2,
        i.ICD_PRCDR_CD3, i.ICD_PRCDR_CD4]) AS t(code)
WHERE TRY_STRPTIME(i.THRU_DT, '%Y%m%d') < c.index_adm_date
  AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d') >= c.index_adm_date - INTERVAL 730 DAY
  AND (LEFT(code, 4) = 'Z931' OR code IN ('0DH63UZ', '0DH60UZ'));

-- Step C: union into single exclusion set
CREATE OR REPLACE TEMP TABLE _pre_stroke_tube AS
SELECT DSYSRTKY FROM _pre_car_tube
UNION
SELECT DSYSRTKY FROM _pre_inp_tube;

-- All carrier claims (unfiltered) for cost — separate from the dx-filtered _car used for outcomes
CREATE OR REPLACE TEMP TABLE _car_cost AS
SELECT
    c.DSYSRTKY,
    TRY_CAST(cl.LINEPMT AS DOUBLE) AS PMT_AMT
FROM stroke_cohort c
JOIN car_linek_all cl ON cl.DSYSRTKY = c.DSYSRTKY
WHERE TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') > c.index_dschg_date
  AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d') <= c.index_dschg_date + INTERVAL 365 DAY;

-- Total Medicare cost in 365d post-discharge (all claim types)
-- Inpatient + Outpatient (claim-level PMT_AMT) + all Carrier lines + SNF + HHA
-- _rev (revenue line level) excluded to avoid double-counting outpatient claim PMT_AMT
CREATE OR REPLACE TEMP TABLE _cost AS
SELECT DSYSRTKY, SUM(PMT_AMT) AS total_pmt_365d FROM (
    SELECT DSYSRTKY, PMT_AMT                       FROM _inp
    UNION ALL SELECT DSYSRTKY, PMT_AMT             FROM _out
    UNION ALL SELECT DSYSRTKY, PMT_AMT             FROM _car_cost
    UNION ALL SELECT DSYSRTKY, snf_pmt_365d       FROM _snf
    UNION ALL SELECT DSYSRTKY, hha_pmt_365d       FROM _hha
) GROUP BY DSYSRTKY;

-- MBSF: death date
CREATE OR REPLACE TEMP TABLE _death AS
SELECT
    m.DSYSRTKY,
    MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
             THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END) AS death_date
FROM mbsf_all m
JOIN stroke_cohort c ON m.DSYSRTKY = c.DSYSRTKY
GROUP BY m.DSYSRTKY;

-- ── Final: Build stroke_outcomes ──────────────────────────────────────────────

DROP TABLE IF EXISTS stroke_outcomes;

CREATE TABLE stroke_outcomes AS
SELECT
    c.DSYSRTKY,
    c.index_adm_date,
    c.index_dschg_date,

    -- Mortality
    d.death_date,
    DATEDIFF('day', c.index_dschg_date, d.death_date) AS days_to_death,

    -- All-cause readmission
    ra.first_readmit_date,
    DATEDIFF('day', c.index_dschg_date, ra.first_readmit_date) AS days_to_readmit,
    COALESCE(ra.n_readmissions_365d, 0) AS n_readmissions_365d,

    -- Recurrent stroke
    rs.first_recur_stroke_date,
    DATEDIFF('day', c.index_dschg_date, rs.first_recur_stroke_date) AS days_to_recur_stroke,

    -- Aspiration pneumonia
    ap.first_aspiration_date,
    DATEDIFF('day', c.index_dschg_date, ap.first_aspiration_date)   AS days_to_aspiration,
    (ap.first_aspiration_date IS NOT NULL)                           AS has_aspiration,

    -- All pneumonia (secondary; breakdown by code)
    pn.first_pneumonia_date,
    DATEDIFF('day', c.index_dschg_date, pn.first_pneumonia_date)     AS days_to_pneumonia,
    (pn.first_pneumonia_date IS NOT NULL)                             AS has_pneumonia,
    pn.first_pneumonia_code,

    -- Dysphagia diagnosis
    dy.first_dysphagia_date,
    DATEDIFF('day', c.index_dschg_date, dy.first_dysphagia_date)    AS days_to_dysphagia,
    (dy.first_dysphagia_date IS NOT NULL)                            AS has_dysphagia,

    -- G-tube dependence
    gt.first_gtube_date,
    DATEDIFF('day', c.index_dschg_date, gt.first_gtube_date)        AS days_to_gtube,
    (gt.first_gtube_date IS NOT NULL)                                AS has_gtube,
    (pt.DSYSRTKY IS NOT NULL)::INT                                   AS pre_stroke_tube,

    -- SNF placement
    snf.first_snf_date,
    DATEDIFF('day', c.index_dschg_date, snf.first_snf_date)         AS days_to_snf,
    (snf.first_snf_date IS NOT NULL AND
     DATEDIFF('day', c.index_dschg_date, snf.first_snf_date) <= 30) AS snf_30d,
    COALESCE(snf.snf_pmt_365d, 0)                                   AS snf_pmt_365d,

    -- Home health
    hha.first_hha_date,
    (hha.first_hha_date IS NOT NULL AND
     DATEDIFF('day', c.index_dschg_date, hha.first_hha_date) <= 90) AS hha_90d,
    COALESCE(hha.hha_pmt_365d, 0)                                   AS hha_pmt_365d,

    -- Total Medicare cost (post-discharge 365d, excludes index admission)
    COALESCE(co.total_pmt_365d, 0)                                  AS total_pmt_365d

FROM stroke_cohort c
LEFT JOIN _death         d   ON d.DSYSRTKY  = c.DSYSRTKY
LEFT JOIN _readmit       ra  ON ra.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _recur_stroke  rs  ON rs.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _aspiration    ap  ON ap.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _pneumonia_all pn  ON pn.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _dysphagia     dy  ON dy.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _gtube         gt  ON gt.DSYSRTKY  = c.DSYSRTKY
LEFT JOIN _pre_stroke_tube pt ON pt.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _snf           snf ON snf.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _hha           hha ON hha.DSYSRTKY = c.DSYSRTKY
LEFT JOIN _cost          co  ON co.DSYSRTKY  = c.DSYSRTKY;

-- ── Summary ───────────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                                                AS n_total,
    SUM(CASE WHEN days_to_death IS NOT NULL AND days_to_death <=  90 THEN 1 ELSE 0 END) AS died_90d,
    SUM(CASE WHEN days_to_death IS NOT NULL AND days_to_death <= 180 THEN 1 ELSE 0 END) AS died_180d,
    SUM(CASE WHEN days_to_death IS NOT NULL AND days_to_death <= 365 THEN 1 ELSE 0 END) AS died_365d,
    SUM(CASE WHEN days_to_readmit IS NOT NULL AND days_to_readmit <=  90 THEN 1 ELSE 0 END) AS readmit_90d,
    SUM(CASE WHEN days_to_readmit IS NOT NULL AND days_to_readmit <= 180 THEN 1 ELSE 0 END) AS readmit_180d,
    SUM(CASE WHEN days_to_readmit IS NOT NULL AND days_to_readmit <= 365 THEN 1 ELSE 0 END) AS readmit_365d,
    SUM(CASE WHEN days_to_recur_stroke IS NOT NULL AND days_to_recur_stroke <=  90 THEN 1 ELSE 0 END) AS recur_90d,
    SUM(CASE WHEN days_to_recur_stroke IS NOT NULL AND days_to_recur_stroke <= 180 THEN 1 ELSE 0 END) AS recur_180d,
    SUM(CASE WHEN days_to_recur_stroke IS NOT NULL AND days_to_recur_stroke <= 365 THEN 1 ELSE 0 END) AS recur_365d,
    SUM(snf_30d::INT)                                                       AS snf_30d,
    SUM(hha_90d::INT)                                                       AS hha_90d,
    ROUND(AVG(total_pmt_365d), 0)                                          AS mean_cost_365d
FROM stroke_outcomes;
