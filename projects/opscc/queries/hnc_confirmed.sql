DROP TABLE IF EXISTS hnc_confirmed;

CREATE TABLE hnc_confirmed AS

WITH base AS (
    SELECT
        DSYSRTKY,
        dx_date,
        source
    FROM hnc_dx_raw
),

-- Count inpatient claims per patient
inpatient_flag AS (
    SELECT
        DSYSRTKY,
        COUNT(*) AS inpatient_claims
    FROM base
    WHERE source = 'inp'
    GROUP BY DSYSRTKY
),

-- For outpatient-only cases, check spacing
outpatient_spacing AS (
    SELECT
        DSYSRTKY,
        MIN(dx_date) AS first_date,
        MAX(dx_date) AS last_date,
        COUNT(*) AS total_claims
    FROM base
    WHERE source <> 'inp'
    GROUP BY DSYSRTKY
)

SELECT
    b.DSYSRTKY,
    MIN(b.dx_date) AS first_hnc_date,
    COUNT(*) AS total_claims,
    MAX(COALESCE(i.inpatient_claims,0)) AS inpatient_claims
FROM base b
LEFT JOIN inpatient_flag i ON b.DSYSRTKY = i.DSYSRTKY
LEFT JOIN outpatient_spacing o ON b.DSYSRTKY = o.DSYSRTKY
GROUP BY b.DSYSRTKY
HAVING
    MAX(COALESCE(i.inpatient_claims,0)) >= 1
    OR (
        MAX(o.total_claims) >= 2
        AND MAX(o.last_date) >= MIN(o.first_date) + INTERVAL 30 DAY
    );