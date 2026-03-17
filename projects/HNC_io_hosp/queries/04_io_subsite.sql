-- Step 4: Restrict to eligible HNC subsites; assign predominant subsite
-- Source: io_hnc_dx_raw (joined with io_hnc_confirmed to limit to confirmed patients)
-- dx_prefix is now 4-char (SUBSTRING(code,1,4)) from step 2
-- Excluded subsites: C11 (nasopharynx), C07/C08 (salivary glands),
--   C30/C31 (nasal/sinus), C44 (squamous cell skin)
-- Output: io_subsite

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_subsite;

CREATE TABLE io_subsite AS

WITH subsite_counts AS (
    SELECT
        r.DSYSRTKY,
        r.dx_prefix,
        COUNT(*) AS cnt
    FROM io_hnc_dx_raw r
    JOIN io_hnc_confirmed c ON r.DSYSRTKY = c.DSYSRTKY
    -- Exclude ineligible subsites using 3-char prefix for consistency
    WHERE LEFT(r.dx_prefix, 3) NOT IN ('C11','C07','C08','C30','C31','C44')
    GROUP BY r.DSYSRTKY, r.dx_prefix
),

ranked AS (
    SELECT
        DSYSRTKY,
        dx_prefix,
        cnt,
        ROW_NUMBER() OVER (PARTITION BY DSYSRTKY ORDER BY cnt DESC, dx_prefix ASC) AS rn
    FROM subsite_counts
)

SELECT
    r.DSYSRTKY,
    r.dx_prefix AS predominant_subsite,
    CASE
        -- Oral Cavity
        WHEN LEFT(r.dx_prefix,3) = 'C00'              THEN 'Oral Cavity'
        WHEN r.dx_prefix = 'C01'                       THEN 'Oral Cavity'  -- base of tongue
        WHEN LEFT(r.dx_prefix,3) = 'C02'
             AND r.dx_prefix <> 'C024'                 THEN 'Oral Cavity'  -- oral tongue (not lingual tonsil)
        WHEN LEFT(r.dx_prefix,3) IN ('C03','C04','C06') THEN 'Oral Cavity'
        WHEN r.dx_prefix IN ('C050','C058','C059')      THEN 'Oral Cavity'  -- hard palate
        WHEN r.dx_prefix = 'C148'                       THEN 'Oral Cavity'  -- overlapping lip/oral/pharynx
        -- Oropharynx
        WHEN r.dx_prefix = 'C024'                       THEN 'Oropharynx'  -- lingual tonsil
        WHEN r.dx_prefix IN ('C051','C052')             THEN 'Oropharynx'  -- soft palate/uvula
        WHEN LEFT(r.dx_prefix,3) IN ('C09','C10')       THEN 'Oropharynx'
        WHEN r.dx_prefix IN ('C140','C142')             THEN 'Oropharynx'
        -- Hypopharynx
        WHEN r.dx_prefix = 'C12'                        THEN 'Hypopharynx' -- pyriform sinus
        WHEN LEFT(r.dx_prefix,3) = 'C13'                THEN 'Hypopharynx'
        -- Larynx
        WHEN LEFT(r.dx_prefix,3) = 'C32'                THEN 'Larynx'
        -- Other HNC (C14x NEC, C43/C49/C76/C77 H&N)
        ELSE 'Other HNC'
    END AS subsite_category

FROM ranked r
WHERE r.rn = 1
  AND LEFT(r.dx_prefix, 3) NOT IN ('C11','C07','C08','C30','C31','C44');
