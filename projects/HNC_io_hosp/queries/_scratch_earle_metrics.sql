-- Earle-benchmark end-of-life metrics for HNC+ICI decedent cohort.
--
-- Per-patient flags computed against 30-day window prior to death:
--   ed_encounters_last_30d       — distinct outpatient ED visit dates (REV_CNTR 0450-0459)
--                                   PLUS admitted-from-ED inpatient stays (TYPE_ADM='1')
--                                   (unique ADMSN_DTs, avoiding double-count of ED->admit)
--   ge_2_ed_encounters_last_30d  — Earle threshold (>=2 ED encounters)
--   icu_stay_last_30d            — any inpatient stay with ICU/CCU rev code overlapping
--                                   the last 30d before death (rev codes 020x, 021x)
--   admission_last_30d           — any inpatient stay with ADMSN_DT within last 30d
--   ge_1_ed_last_30d             — helper: any ED at all (outpatient or admitted-from)

SET memory_limit='24GB';
SET threads=12;

WITH cohort AS (
    SELECT DSYSRTKY, death_dt, hospice_enrolled
    FROM io_analytic
),

-- Outpatient ED visits (distinct dates only, so multi-line ED claims count once per date)
ed_outpatient AS (
    SELECT DISTINCT
        r.DSYSRTKY,
        TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d') AS ed_dt
    FROM io_out_revenue r
    JOIN cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE r.REV_CNTR BETWEEN '0450' AND '0459'
      AND TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),

-- Admitted-from-ED (TYPE_ADM = '1') — 1 row per hospitalization
ed_admitted AS (
    SELECT DISTINCT
        i.DSYSRTKY,
        TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d') AS ed_dt
    FROM io_inp_claims i
    JOIN cohort c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE i.TYPE_ADM = '1'
      AND TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),

-- Combined ED encounters (unique dates per patient; outpatient ED that led to same-day
-- inpatient admission collapses to one encounter)
ed_all AS (
    SELECT DSYSRTKY, ed_dt FROM ed_outpatient
    UNION
    SELECT DSYSRTKY, ed_dt FROM ed_admitted
),

ed_counts AS (
    SELECT DSYSRTKY, COUNT(DISTINCT ed_dt) AS n_ed
    FROM ed_all
    GROUP BY 1
),

-- ICU/CCU: inpatient stays whose ICU-line THRU_DT overlaps the last 30d of life.
-- (Approximation: use rev-line THRU_DT rather than a full stay-overlap window.
--  Nearly all short EOL ICU stays fall inside 30d anyway; erring toward simple.)
icu_stays AS (
    SELECT DISTINCT
        r.DSYSRTKY,
        r.CLAIMNO
    FROM io_inp_revenue r
    JOIN cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE (r.REV_CNTR BETWEEN '0200' AND '0209'
        OR r.REV_CNTR BETWEEN '0210' AND '0219')
      AND TRY_STRPTIME(r.THRU_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),

-- Any inpatient admission in last 30d (ADMSN_DT window)
adm_last_30d AS (
    SELECT DISTINCT i.DSYSRTKY, i.CLAIMNO
    FROM io_inp_claims i
    JOIN cohort c ON i.DSYSRTKY = c.DSYSRTKY
    WHERE TRY_STRPTIME(i.ADMSN_DT, '%Y%m%d')
            BETWEEN (c.death_dt - INTERVAL 30 DAY) AND c.death_dt
),

per_patient AS (
    SELECT
        c.DSYSRTKY,
        c.hospice_enrolled,
        COALESCE(ec.n_ed, 0)                                         AS n_ed_last_30d,
        (COALESCE(ec.n_ed, 0) >= 1)::INT                             AS any_ed_last_30d,
        (COALESCE(ec.n_ed, 0) >= 2)::INT                             AS ge_2_ed_last_30d,
        (EXISTS (SELECT 1 FROM icu_stays i WHERE i.DSYSRTKY = c.DSYSRTKY))::INT
                                                                     AS icu_last_30d,
        (EXISTS (SELECT 1 FROM adm_last_30d a WHERE a.DSYSRTKY = c.DSYSRTKY))::INT
                                                                     AS admission_last_30d
    FROM cohort c
    LEFT JOIN ed_counts ec ON c.DSYSRTKY = ec.DSYSRTKY
)

-- Summary
SELECT
    'OVERALL' AS stratum,
    COUNT(*) AS n,
    SUM(any_ed_last_30d)     AS n_any_ed,
    ROUND(100.0 * SUM(any_ed_last_30d) / COUNT(*), 1)     AS pct_any_ed,
    SUM(ge_2_ed_last_30d)    AS n_ge2_ed,
    ROUND(100.0 * SUM(ge_2_ed_last_30d) / COUNT(*), 1)    AS pct_ge2_ed,
    SUM(icu_last_30d)        AS n_icu,
    ROUND(100.0 * SUM(icu_last_30d) / COUNT(*), 1)        AS pct_icu,
    SUM(admission_last_30d)  AS n_adm,
    ROUND(100.0 * SUM(admission_last_30d) / COUNT(*), 1)  AS pct_adm
FROM per_patient
UNION ALL
SELECT
    CASE hospice_enrolled WHEN 1 THEN 'HOSPICE' ELSE 'NO HOSPICE' END,
    COUNT(*),
    SUM(any_ed_last_30d),  ROUND(100.0 * SUM(any_ed_last_30d) / COUNT(*), 1),
    SUM(ge_2_ed_last_30d), ROUND(100.0 * SUM(ge_2_ed_last_30d) / COUNT(*), 1),
    SUM(icu_last_30d),     ROUND(100.0 * SUM(icu_last_30d) / COUNT(*), 1),
    SUM(admission_last_30d), ROUND(100.0 * SUM(admission_last_30d) / COUNT(*), 1)
FROM per_patient
GROUP BY hospice_enrolled
ORDER BY stratum;
