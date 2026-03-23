-- Step 07a: Pre-filter large tables to 7K HNC+IO patients
-- Scans each large claim table ONCE; downstream steps (07, 10, 11, 12) use these instead
-- Join target: io_episodes INTERSECT io_subsite (~7K patients)
-- NO date filtering here — each downstream step applies its own date window
-- Output tables: io_car_lines, io_out_revenue, io_inp_claims, io_out_claims, io_hosp_claims

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

-- ── 1. Carrier line claims ────────────────────────────────────────────────────
DROP TABLE IF EXISTS io_car_lines;

CREATE TABLE io_car_lines AS
SELECT
    l.DSYSRTKY,
    l.HCPCS_CD,
    l.THRU_DT,
    l.LINE_ICD_DGNS_CD
FROM car_linek_all l
JOIN (
    SELECT e.DSYSRTKY
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
) h ON l.DSYSRTKY = h.DSYSRTKY;

-- ── 2. Outpatient revenue line claims ─────────────────────────────────────────
DROP TABLE IF EXISTS io_out_revenue;

CREATE TABLE io_out_revenue AS
SELECT
    r.DSYSRTKY,
    r.HCPCS_CD,
    r.REV_DT,
    r.THRU_DT
FROM out_revenuek_all r
JOIN (
    SELECT e.DSYSRTKY
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
) h ON r.DSYSRTKY = h.DSYSRTKY;

-- ── 3. Inpatient claims (all columns needed by steps 07, 10, 11) ──────────────
DROP TABLE IF EXISTS io_inp_claims;

CREATE TABLE io_inp_claims AS
SELECT
    i.DSYSRTKY,
    i.ADMSN_DT,
    i.THRU_DT,
    i.STUS_CD,
    i.DSCHRGDT,
    i.PRNCPAL_DGNS_CD,
    i.ADMTG_DGNS_CD,
    i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,  i.ICD_DGNS_CD4,
    i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,  i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,
    i.ICD_DGNS_CD9,  i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
    i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15, i.ICD_DGNS_CD16,
    i.ICD_DGNS_CD17, i.ICD_DGNS_CD18, i.ICD_DGNS_CD19, i.ICD_DGNS_CD20,
    i.ICD_DGNS_CD21, i.ICD_DGNS_CD22, i.ICD_DGNS_CD23, i.ICD_DGNS_CD24,
    i.ICD_DGNS_CD25,
    i.ICD_PRCDR_CD1,  i.ICD_PRCDR_CD2,  i.ICD_PRCDR_CD3,  i.ICD_PRCDR_CD4,
    i.ICD_PRCDR_CD5,  i.ICD_PRCDR_CD6,  i.ICD_PRCDR_CD7,  i.ICD_PRCDR_CD8,
    i.ICD_PRCDR_CD9,  i.ICD_PRCDR_CD10, i.ICD_PRCDR_CD11, i.ICD_PRCDR_CD12,
    i.ICD_PRCDR_CD13, i.ICD_PRCDR_CD14, i.ICD_PRCDR_CD15, i.ICD_PRCDR_CD16,
    i.ICD_PRCDR_CD17, i.ICD_PRCDR_CD18, i.ICD_PRCDR_CD19, i.ICD_PRCDR_CD20,
    i.ICD_PRCDR_CD21, i.ICD_PRCDR_CD22, i.ICD_PRCDR_CD23, i.ICD_PRCDR_CD24,
    i.ICD_PRCDR_CD25,
    i.PRCDR_DT1
FROM inp_claimsk_all i
JOIN (
    SELECT e.DSYSRTKY
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
) h ON i.DSYSRTKY = h.DSYSRTKY;

-- ── 4. Outpatient claims (diagnoses for Elixhauser step 10) ───────────────────
DROP TABLE IF EXISTS io_out_claims;

CREATE TABLE io_out_claims AS
SELECT
    oc.DSYSRTKY,
    oc.THRU_DT,
    oc.PRNCPAL_DGNS_CD,
    oc.ICD_DGNS_CD1,  oc.ICD_DGNS_CD2,  oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4,
    oc.ICD_DGNS_CD5,  oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
    oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10, oc.ICD_DGNS_CD11, oc.ICD_DGNS_CD12,
    oc.ICD_DGNS_CD13, oc.ICD_DGNS_CD14, oc.ICD_DGNS_CD15, oc.ICD_DGNS_CD16,
    oc.ICD_DGNS_CD17, oc.ICD_DGNS_CD18, oc.ICD_DGNS_CD19, oc.ICD_DGNS_CD20,
    oc.ICD_DGNS_CD21, oc.ICD_DGNS_CD22, oc.ICD_DGNS_CD23, oc.ICD_DGNS_CD24,
    oc.ICD_DGNS_CD25
FROM out_claimsk_all oc
JOIN (
    SELECT e.DSYSRTKY
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
) h ON oc.DSYSRTKY = h.DSYSRTKY;

-- ── 5. Hospice claims (outcomes step 11) ──────────────────────────────────────
-- Checkpoint table: io_hosp_claims (last created — existence confirms all 5 are done)
DROP TABLE IF EXISTS io_hosp_claims;

CREATE TABLE io_hosp_claims AS
SELECT
    h.DSYSRTKY,
    h.HSPCSTRT
FROM hosp_claimsk_all h
JOIN (
    SELECT e.DSYSRTKY
    FROM io_episodes e
    JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
) p ON h.DSYSRTKY = p.DSYSRTKY;
