-- CBE/NQF 0210 metric — expanded drug list
-- Question: % of io_cohort decedents who received ANY systemic anti-cancer
-- therapy within 14 days before death, when the drug list is expanded from
-- (ICI + platinum only) to (ICI + platinum + taxanes + cetuximab + 5-FU).
--
-- Baseline (current paper text): ICI + platinum → 270 (10.7%)
-- Expanded (PI request):        ICI + platinum + cetuximab + docetaxel + paclitaxel + 5-FU
--
-- Sources: io_car_lines (Part B carrier) + io_out_revenue (outpatient revenue).
-- Both are already restricted to cohort DSYSRTKYs in staging.

SET memory_limit='24GB';
SET threads=12;

WITH drug_claims AS (
    SELECT l.DSYSRTKY,
           TRY_STRPTIME(l.THRU_DT, '%Y%m%d') AS drug_dt,
           l.HCPCS_CD
    FROM io_car_lines l
    JOIN io_cohort c ON l.DSYSRTKY = c.DSYSRTKY
    WHERE l.HCPCS_CD IN (
        'J9271','J9299',                    -- ICI: pembrolizumab, nivolumab
        'J9060','J9045',                    -- platinum: cisplatin, carboplatin
        'J9055',                            -- cetuximab
        'J9171','J9267',                    -- taxanes: docetaxel, paclitaxel
        'J9190'                             -- 5-FU
    )

    UNION ALL

    SELECT r.DSYSRTKY,
           TRY_STRPTIME(COALESCE(NULLIF(r.REV_DT,''), r.THRU_DT), '%Y%m%d') AS drug_dt,
           r.HCPCS_CD
    FROM io_out_revenue r
    JOIN io_cohort c ON r.DSYSRTKY = c.DSYSRTKY
    WHERE r.HCPCS_CD IN (
        'J9271','J9299',
        'J9060','J9045',
        'J9055',
        'J9171','J9267',
        'J9190'
    )
),

pt_flags AS (
    SELECT
        c.DSYSRTKY,
        BOOL_OR(
            dc.HCPCS_CD IN ('J9271','J9299')
            AND datediff('day', dc.drug_dt, c.death_dt) BETWEEN 0 AND 14
        ) AS ici_14d,
        BOOL_OR(
            dc.HCPCS_CD IN ('J9060','J9045')
            AND datediff('day', dc.drug_dt, c.death_dt) BETWEEN 0 AND 14
        ) AS plat_14d,
        BOOL_OR(
            dc.HCPCS_CD = 'J9055'
            AND datediff('day', dc.drug_dt, c.death_dt) BETWEEN 0 AND 14
        ) AS cetux_14d,
        BOOL_OR(
            dc.HCPCS_CD IN ('J9171','J9267')
            AND datediff('day', dc.drug_dt, c.death_dt) BETWEEN 0 AND 14
        ) AS taxane_14d,
        BOOL_OR(
            dc.HCPCS_CD = 'J9190'
            AND datediff('day', dc.drug_dt, c.death_dt) BETWEEN 0 AND 14
        ) AS fu_14d
    FROM io_cohort c
    LEFT JOIN drug_claims dc ON c.DSYSRTKY = dc.DSYSRTKY
    GROUP BY c.DSYSRTKY
)

SELECT
    COUNT(*) AS n_cohort,
    -- Current definition (paper): ICI + platinum within 14d
    COUNT(*) FILTER (WHERE COALESCE(ici_14d,FALSE) OR COALESCE(plat_14d,FALSE))                    AS n_current,
    ROUND(100.0 * COUNT(*) FILTER (WHERE COALESCE(ici_14d,FALSE) OR COALESCE(plat_14d,FALSE)) / COUNT(*), 2) AS pct_current,
    -- Expanded definition: ICI + platinum + taxanes + cetuximab + 5-FU within 14d
    COUNT(*) FILTER (WHERE COALESCE(ici_14d,FALSE) OR COALESCE(plat_14d,FALSE)
                        OR COALESCE(cetux_14d,FALSE)
                        OR COALESCE(taxane_14d,FALSE)
                        OR COALESCE(fu_14d,FALSE))                                                 AS n_expanded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE COALESCE(ici_14d,FALSE) OR COALESCE(plat_14d,FALSE)
                        OR COALESCE(cetux_14d,FALSE)
                        OR COALESCE(taxane_14d,FALSE)
                        OR COALESCE(fu_14d,FALSE)) / COUNT(*), 2)                                  AS pct_expanded,
    -- Contribution of each added agent alone (had that agent 14d, but not ICI or platinum 14d)
    COUNT(*) FILTER (WHERE COALESCE(cetux_14d,FALSE)  AND NOT COALESCE(ici_14d,FALSE) AND NOT COALESCE(plat_14d,FALSE)) AS n_cetux_only,
    COUNT(*) FILTER (WHERE COALESCE(taxane_14d,FALSE) AND NOT COALESCE(ici_14d,FALSE) AND NOT COALESCE(plat_14d,FALSE)) AS n_taxane_only,
    COUNT(*) FILTER (WHERE COALESCE(fu_14d,FALSE)     AND NOT COALESCE(ici_14d,FALSE) AND NOT COALESCE(plat_14d,FALSE)) AS n_fu_only,
    -- Individual agent totals within 14d (may overlap with each other)
    COUNT(*) FILTER (WHERE COALESCE(ici_14d,FALSE))    AS n_ici_14d,
    COUNT(*) FILTER (WHERE COALESCE(plat_14d,FALSE))   AS n_plat_14d,
    COUNT(*) FILTER (WHERE COALESCE(cetux_14d,FALSE))  AS n_cetux_14d,
    COUNT(*) FILTER (WHERE COALESCE(taxane_14d,FALSE)) AS n_taxane_14d,
    COUNT(*) FILTER (WHERE COALESCE(fu_14d,FALSE))     AS n_fu_14d
FROM pt_flags;
