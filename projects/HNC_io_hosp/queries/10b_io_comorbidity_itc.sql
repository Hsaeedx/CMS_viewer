-- Step 10b: Elixhauser comorbidity for ITT cohort (io_cohort_itc)
-- Identical to 10_io_elixhauser.sql; io_cohort replaced with io_cohort_itc
-- Output: io_comorbidity_itc

SET memory_limit='24GB';
SET threads=12;
SET temp_directory='F:\CMS\duckdb_temp';

DROP TABLE IF EXISTS io_comorbidity_itc;

CREATE TABLE io_comorbidity_itc AS

WITH dx_raw AS (

    -- Inpatient: principal + admitting + 25 secondary diagnosis slots
    SELECT h.DSYSRTKY, code
    FROM io_inp_claims i
    JOIN io_cohort_itc h ON i.DSYSRTKY = h.DSYSRTKY,
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
    WHERE code IS NOT NULL
      AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
              BETWEEN h.death_dt - INTERVAL 13 MONTH
                  AND h.death_dt - INTERVAL 2 MONTH

    UNION ALL

    -- Carrier: one diagnosis code per line (LINE_ICD_DGNS_CD)
    SELECT h.DSYSRTKY, c.LINE_ICD_DGNS_CD AS code
    FROM io_car_lines c
    JOIN io_cohort_itc h ON c.DSYSRTKY = h.DSYSRTKY
    WHERE c.LINE_ICD_DGNS_CD IS NOT NULL
      AND TRY_STRPTIME(c.THRU_DT, '%Y%m%d')
              BETWEEN h.death_dt - INTERVAL 13 MONTH
                  AND h.death_dt - INTERVAL 2 MONTH

    UNION ALL

    -- Outpatient: principal + 25 secondary diagnosis slots
    SELECT h.DSYSRTKY, code
    FROM io_out_claims oc
    JOIN io_cohort_itc h ON oc.DSYSRTKY = h.DSYSRTKY,
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
    WHERE code IS NOT NULL
      AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
              BETWEEN h.death_dt - INTERVAL 13 MONTH
                  AND h.death_dt - INTERVAL 2 MONTH
),

flags AS (
    SELECT
        DSYSRTKY,
        MAX(CASE WHEN
            code LIKE 'I099%' OR code LIKE 'I110%' OR code LIKE 'I130%' OR code LIKE 'I132%' OR
            code LIKE 'I255%' OR code LIKE 'I420%' OR code LIKE 'I425%' OR code LIKE 'I426%' OR
            code LIKE 'I427%' OR code LIKE 'I428%' OR code LIKE 'I429%' OR code LIKE 'I43%'  OR
            code LIKE 'I50%'  OR code LIKE 'P290%'
        THEN 1 ELSE 0 END) AS chf,
        MAX(CASE WHEN
            code LIKE 'I441%' OR code LIKE 'I442%' OR code LIKE 'I443%' OR code LIKE 'I456%' OR
            code LIKE 'I459%' OR code LIKE 'I47%'  OR code LIKE 'I48%'  OR code LIKE 'I49%'  OR
            code LIKE 'R000%' OR code LIKE 'R001%' OR code LIKE 'R008%' OR code LIKE 'T821%' OR
            code LIKE 'Z450%' OR code LIKE 'Z950%'
        THEN 1 ELSE 0 END) AS carit,
        MAX(CASE WHEN
            code LIKE 'A520%' OR code LIKE 'I05%'  OR code LIKE 'I06%'  OR code LIKE 'I07%'  OR
            code LIKE 'I08%'  OR code LIKE 'I091%' OR code LIKE 'I098%' OR code LIKE 'I34%'  OR
            code LIKE 'I35%'  OR code LIKE 'I36%'  OR code LIKE 'I37%'  OR code LIKE 'I38%'  OR
            code LIKE 'I39%'  OR code LIKE 'Q230%' OR code LIKE 'Q231%' OR code LIKE 'Q232%' OR
            code LIKE 'Q233%' OR code LIKE 'Z952%' OR code LIKE 'Z953%' OR code LIKE 'Z954%'
        THEN 1 ELSE 0 END) AS valv,
        MAX(CASE WHEN
            code LIKE 'I26%'  OR code LIKE 'I27%'  OR code LIKE 'I280%' OR
            code LIKE 'I288%' OR code LIKE 'I289%'
        THEN 1 ELSE 0 END) AS pcd,
        MAX(CASE WHEN
            code LIKE 'I70%'  OR code LIKE 'I71%'  OR code LIKE 'I731%' OR code LIKE 'I738%' OR
            code LIKE 'I739%' OR code LIKE 'I771%' OR code LIKE 'I790%' OR code LIKE 'I792%' OR
            code LIKE 'K551%' OR code LIKE 'K558%' OR code LIKE 'K559%' OR code LIKE 'Z958%' OR
            code LIKE 'Z959%'
        THEN 1 ELSE 0 END) AS pvd,
        MAX(CASE WHEN code LIKE 'I10%' THEN 1 ELSE 0 END) AS hypunc,
        MAX(CASE WHEN
            code LIKE 'I11%' OR code LIKE 'I12%' OR code LIKE 'I13%' OR code LIKE 'I15%'
        THEN 1 ELSE 0 END) AS hypc,
        MAX(CASE WHEN
            code LIKE 'G041%' OR code LIKE 'G114%' OR code LIKE 'G801%' OR code LIKE 'G802%' OR
            code LIKE 'G81%'  OR code LIKE 'G82%'  OR code LIKE 'G830%' OR code LIKE 'G831%' OR
            code LIKE 'G832%' OR code LIKE 'G833%' OR code LIKE 'G834%' OR code LIKE 'G839%'
        THEN 1 ELSE 0 END) AS para,
        MAX(CASE WHEN
            code LIKE 'G10%'  OR code LIKE 'G11%'  OR code LIKE 'G12%'  OR code LIKE 'G13%'  OR
            code LIKE 'G20%'  OR code LIKE 'G21%'  OR code LIKE 'G22%'  OR code LIKE 'G254%' OR
            code LIKE 'G255%' OR code LIKE 'G312%' OR code LIKE 'G318%' OR code LIKE 'G319%' OR
            code LIKE 'G32%'  OR code LIKE 'G35%'  OR code LIKE 'G36%'  OR code LIKE 'G37%'  OR
            code LIKE 'G40%'  OR code LIKE 'G41%'  OR code LIKE 'G931%' OR code LIKE 'G934%' OR
            code LIKE 'R470%' OR code LIKE 'R56%'
        THEN 1 ELSE 0 END) AS ond,
        MAX(CASE WHEN
            code LIKE 'I278%' OR code LIKE 'I279%' OR code LIKE 'J40%'  OR code LIKE 'J41%'  OR
            code LIKE 'J42%'  OR code LIKE 'J43%'  OR code LIKE 'J44%'  OR code LIKE 'J45%'  OR
            code LIKE 'J46%'  OR code LIKE 'J47%'  OR code LIKE 'J60%'  OR code LIKE 'J61%'  OR
            code LIKE 'J62%'  OR code LIKE 'J63%'  OR code LIKE 'J64%'  OR code LIKE 'J65%'  OR
            code LIKE 'J66%'  OR code LIKE 'J67%'  OR code LIKE 'J684%' OR code LIKE 'J701%' OR
            code LIKE 'J703%'
        THEN 1 ELSE 0 END) AS cpd,
        MAX(CASE WHEN
            code LIKE 'E100%' OR code LIKE 'E101%' OR code LIKE 'E109%' OR
            code LIKE 'E110%' OR code LIKE 'E111%' OR code LIKE 'E119%' OR
            code LIKE 'E120%' OR code LIKE 'E121%' OR code LIKE 'E129%' OR
            code LIKE 'E130%' OR code LIKE 'E131%' OR code LIKE 'E139%' OR
            code LIKE 'E140%' OR code LIKE 'E141%' OR code LIKE 'E149%'
        THEN 1 ELSE 0 END) AS diabunc,
        MAX(CASE WHEN
            code LIKE 'E102%' OR code LIKE 'E103%' OR code LIKE 'E104%' OR code LIKE 'E105%' OR
            code LIKE 'E106%' OR code LIKE 'E107%' OR code LIKE 'E108%' OR code LIKE 'E112%' OR
            code LIKE 'E113%' OR code LIKE 'E114%' OR code LIKE 'E115%' OR code LIKE 'E116%' OR
            code LIKE 'E117%' OR code LIKE 'E118%' OR code LIKE 'E122%' OR code LIKE 'E123%' OR
            code LIKE 'E124%' OR code LIKE 'E125%' OR code LIKE 'E126%' OR code LIKE 'E127%' OR
            code LIKE 'E128%' OR code LIKE 'E132%' OR code LIKE 'E133%' OR code LIKE 'E134%' OR
            code LIKE 'E135%' OR code LIKE 'E136%' OR code LIKE 'E137%' OR code LIKE 'E138%' OR
            code LIKE 'E142%' OR code LIKE 'E143%' OR code LIKE 'E144%' OR code LIKE 'E145%' OR
            code LIKE 'E146%' OR code LIKE 'E147%' OR code LIKE 'E148%'
        THEN 1 ELSE 0 END) AS diabc,
        MAX(CASE WHEN
            code LIKE 'E00%'  OR code LIKE 'E01%' OR code LIKE 'E02%' OR
            code LIKE 'E03%'  OR code LIKE 'E890%'
        THEN 1 ELSE 0 END) AS hypothy,
        MAX(CASE WHEN
            code LIKE 'I120%' OR code LIKE 'I131%' OR code LIKE 'N18%'  OR code LIKE 'N19%'  OR
            code LIKE 'N250%' OR code LIKE 'Z490%' OR code LIKE 'Z491%' OR code LIKE 'Z492%' OR
            code LIKE 'Z940%' OR code LIKE 'Z992%'
        THEN 1 ELSE 0 END) AS rf,
        MAX(CASE WHEN
            code LIKE 'B18%'  OR code LIKE 'I85%'  OR code LIKE 'I864%' OR code LIKE 'I982%' OR
            code LIKE 'K70%'  OR code LIKE 'K711%' OR code LIKE 'K713%' OR code LIKE 'K714%' OR
            code LIKE 'K715%' OR code LIKE 'K717%' OR code LIKE 'K72%'  OR code LIKE 'K73%'  OR
            code LIKE 'K74%'  OR code LIKE 'K760%' OR code LIKE 'K762%' OR code LIKE 'K763%' OR
            code LIKE 'K764%' OR code LIKE 'K765%' OR code LIKE 'K766%' OR code LIKE 'K767%' OR
            code LIKE 'K768%' OR code LIKE 'K769%' OR code LIKE 'Z944%'
        THEN 1 ELSE 0 END) AS ld,
        MAX(CASE WHEN
            code LIKE 'K257%' OR code LIKE 'K259%' OR code LIKE 'K267%' OR code LIKE 'K269%' OR
            code LIKE 'K277%' OR code LIKE 'K279%' OR code LIKE 'K287%' OR code LIKE 'K289%'
        THEN 1 ELSE 0 END) AS pud,
        MAX(CASE WHEN
            code LIKE 'B20%' OR code LIKE 'B21%' OR code LIKE 'B22%' OR code LIKE 'B24%'
        THEN 1 ELSE 0 END) AS aids,
        MAX(CASE WHEN
            code LIKE 'C81%'  OR code LIKE 'C82%'  OR code LIKE 'C83%'  OR code LIKE 'C84%'  OR
            code LIKE 'C85%'  OR code LIKE 'C88%'  OR code LIKE 'C96%'  OR code LIKE 'C900%' OR
            code LIKE 'C902%'
        THEN 1 ELSE 0 END) AS lymph,
        MAX(CASE WHEN
            code LIKE 'C77%' OR code LIKE 'C78%' OR code LIKE 'C79%' OR code LIKE 'C80%'
        THEN 1 ELSE 0 END) AS metacanc,
        MAX(CASE WHEN
            code LIKE 'C00%' OR code LIKE 'C01%' OR code LIKE 'C02%' OR code LIKE 'C03%' OR
            code LIKE 'C04%' OR code LIKE 'C05%' OR code LIKE 'C06%' OR code LIKE 'C07%' OR
            code LIKE 'C08%' OR code LIKE 'C09%' OR code LIKE 'C10%' OR code LIKE 'C11%' OR
            code LIKE 'C12%' OR code LIKE 'C13%' OR code LIKE 'C14%' OR code LIKE 'C15%' OR
            code LIKE 'C16%' OR code LIKE 'C17%' OR code LIKE 'C18%' OR code LIKE 'C19%' OR
            code LIKE 'C20%' OR code LIKE 'C21%' OR code LIKE 'C22%' OR code LIKE 'C23%' OR
            code LIKE 'C24%' OR code LIKE 'C25%' OR code LIKE 'C26%' OR code LIKE 'C30%' OR
            code LIKE 'C31%' OR code LIKE 'C32%' OR code LIKE 'C33%' OR code LIKE 'C34%' OR
            code LIKE 'C37%' OR code LIKE 'C38%' OR code LIKE 'C39%' OR code LIKE 'C40%' OR
            code LIKE 'C41%' OR code LIKE 'C43%' OR code LIKE 'C45%' OR code LIKE 'C46%' OR
            code LIKE 'C47%' OR code LIKE 'C48%' OR code LIKE 'C49%' OR code LIKE 'C50%' OR
            code LIKE 'C51%' OR code LIKE 'C52%' OR code LIKE 'C53%' OR code LIKE 'C54%' OR
            code LIKE 'C55%' OR code LIKE 'C56%' OR code LIKE 'C57%' OR code LIKE 'C58%' OR
            code LIKE 'C60%' OR code LIKE 'C61%' OR code LIKE 'C62%' OR code LIKE 'C63%' OR
            code LIKE 'C64%' OR code LIKE 'C65%' OR code LIKE 'C66%' OR code LIKE 'C67%' OR
            code LIKE 'C68%' OR code LIKE 'C69%' OR code LIKE 'C70%' OR code LIKE 'C71%' OR
            code LIKE 'C72%' OR code LIKE 'C73%' OR code LIKE 'C74%' OR code LIKE 'C75%' OR
            code LIKE 'C76%' OR code LIKE 'C97%'
        THEN 1 ELSE 0 END) AS solidtum,
        MAX(CASE WHEN
            code LIKE 'L940%' OR code LIKE 'L941%' OR code LIKE 'L943%' OR code LIKE 'M05%'  OR
            code LIKE 'M06%'  OR code LIKE 'M08%'  OR code LIKE 'M120%' OR code LIKE 'M123%' OR
            code LIKE 'M30%'  OR code LIKE 'M310%' OR code LIKE 'M311%' OR code LIKE 'M312%' OR
            code LIKE 'M313%' OR code LIKE 'M32%'  OR code LIKE 'M33%'  OR code LIKE 'M34%'  OR
            code LIKE 'M35%'  OR code LIKE 'M45%'  OR code LIKE 'M461%' OR code LIKE 'M468%' OR
            code LIKE 'M469%'
        THEN 1 ELSE 0 END) AS rheumd,
        MAX(CASE WHEN
            code LIKE 'D65%'  OR code LIKE 'D66%'  OR code LIKE 'D67%'  OR code LIKE 'D68%'  OR
            code LIKE 'D691%' OR code LIKE 'D693%' OR code LIKE 'D694%' OR code LIKE 'D695%' OR
            code LIKE 'D696%'
        THEN 1 ELSE 0 END) AS coag,
        MAX(CASE WHEN code LIKE 'E66%' THEN 1 ELSE 0 END) AS obes,
        MAX(CASE WHEN
            code LIKE 'E40%'  OR code LIKE 'E41%'  OR code LIKE 'E42%'  OR code LIKE 'E43%'  OR
            code LIKE 'E44%'  OR code LIKE 'E45%'  OR code LIKE 'E46%'  OR code LIKE 'R634%' OR
            code LIKE 'R64%'
        THEN 1 ELSE 0 END) AS wloss,
        MAX(CASE WHEN
            code LIKE 'E222%' OR code LIKE 'E86%' OR code LIKE 'E87%'
        THEN 1 ELSE 0 END) AS fed,
        MAX(CASE WHEN code LIKE 'D500%' THEN 1 ELSE 0 END) AS blane,
        MAX(CASE WHEN
            code LIKE 'D508%' OR code LIKE 'D509%' OR code LIKE 'D51%' OR
            code LIKE 'D52%'  OR code LIKE 'D53%'
        THEN 1 ELSE 0 END) AS dane,
        MAX(CASE WHEN
            code LIKE 'F10%'  OR code LIKE 'E52%'  OR code LIKE 'G621%' OR code LIKE 'I426%' OR
            code LIKE 'K292%' OR code LIKE 'K700%' OR code LIKE 'K703%' OR code LIKE 'K709%' OR
            code LIKE 'T51%'  OR code LIKE 'Z502%' OR code LIKE 'Z714%' OR code LIKE 'Z721%'
        THEN 1 ELSE 0 END) AS alcohol,
        MAX(CASE WHEN
            code LIKE 'F11%'  OR code LIKE 'F12%'  OR code LIKE 'F13%'  OR code LIKE 'F14%'  OR
            code LIKE 'F15%'  OR code LIKE 'F16%'  OR code LIKE 'F18%'  OR code LIKE 'F19%'  OR
            code LIKE 'Z715%' OR code LIKE 'Z722%'
        THEN 1 ELSE 0 END) AS drug,
        MAX(CASE WHEN
            code LIKE 'F20%'  OR code LIKE 'F22%'  OR code LIKE 'F23%'  OR code LIKE 'F24%'  OR
            code LIKE 'F25%'  OR code LIKE 'F28%'  OR code LIKE 'F29%'  OR code LIKE 'F302%' OR
            code LIKE 'F312%' OR code LIKE 'F315%'
        THEN 1 ELSE 0 END) AS psycho,
        MAX(CASE WHEN
            code LIKE 'F204%' OR code LIKE 'F313%' OR code LIKE 'F314%' OR code LIKE 'F315%' OR
            code LIKE 'F32%'  OR code LIKE 'F33%'  OR code LIKE 'F341%' OR code LIKE 'F412%' OR
            code LIKE 'F432%'
        THEN 1 ELSE 0 END) AS depre

    FROM dx_raw
    GROUP BY DSYSRTKY
)

SELECT
    h.DSYSRTKY,
    COALESCE(f.chf,      0) AS chf,
    COALESCE(f.carit,    0) AS carit,
    COALESCE(f.valv,     0) AS valv,
    COALESCE(f.pcd,      0) AS pcd,
    COALESCE(f.pvd,      0) AS pvd,
    COALESCE(f.hypunc,   0) AS hypunc,
    COALESCE(f.hypc,     0) AS hypc,
    COALESCE(f.para,     0) AS para,
    COALESCE(f.ond,      0) AS ond,
    COALESCE(f.cpd,      0) AS cpd,
    COALESCE(f.diabunc,  0) AS diabunc,
    COALESCE(f.diabc,    0) AS diabc,
    COALESCE(f.hypothy,  0) AS hypothy,
    COALESCE(f.rf,       0) AS rf,
    COALESCE(f.ld,       0) AS ld,
    COALESCE(f.pud,      0) AS pud,
    COALESCE(f.aids,     0) AS aids,
    COALESCE(f.lymph,    0) AS lymph,
    COALESCE(f.metacanc, 0) AS metacanc,
    COALESCE(f.solidtum, 0) AS solidtum,
    COALESCE(f.rheumd,   0) AS rheumd,
    COALESCE(f.coag,     0) AS coag,
    COALESCE(f.obes,     0) AS obes,
    COALESCE(f.wloss,    0) AS wloss,
    COALESCE(f.fed,      0) AS fed,
    COALESCE(f.blane,    0) AS blane,
    COALESCE(f.dane,     0) AS dane,
    COALESCE(f.alcohol,  0) AS alcohol,
    COALESCE(f.drug,     0) AS drug,
    COALESCE(f.psycho,   0) AS psycho,
    COALESCE(f.depre,    0) AS depre,
    (
        COALESCE(f.chf,      0) *  7 +
        COALESCE(f.carit,    0) *  5 +
        COALESCE(f.valv,     0) * -1 +
        COALESCE(f.pcd,      0) *  4 +
        COALESCE(f.pvd,      0) *  2 +
        COALESCE(f.para,     0) *  7 +
        COALESCE(f.ond,      0) *  6 +
        COALESCE(f.cpd,      0) *  3 +
        COALESCE(f.rf,       0) *  5 +
        COALESCE(f.ld,       0) * 11 +
        COALESCE(f.lymph,    0) *  9 +
        COALESCE(f.metacanc, 0) * 12 +
        COALESCE(f.solidtum, 0) *  4 +
        COALESCE(f.coag,     0) *  3 +
        COALESCE(f.obes,     0) * -4 +
        COALESCE(f.wloss,    0) *  6 +
        COALESCE(f.fed,      0) *  5 +
        COALESCE(f.blane,    0) * -2 +
        COALESCE(f.dane,     0) * -2 +
        COALESCE(f.drug,     0) * -7 +
        COALESCE(f.depre,    0) * -3
    ) AS van_walraven_score

FROM io_cohort_itc h
LEFT JOIN flags f ON h.DSYSRTKY = f.DSYSRTKY;
