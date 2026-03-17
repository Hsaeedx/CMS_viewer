"""
outcomes_analysis.py
Odds ratios for dysphagia, G-tube, and tracheostomy
at 6-month, 1-year, 3-year, 5-year, and anytime
Stratified by age group (<75 / >=75) and Elixhauser tertile (Low/Mid/High)
Population: PSM-matched cohort, C77 (nodal disease) excluded
Reference: TORS only vs CT/CRT only
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

DB_PATH = r"F:\CMS\cms_data.duckdb"

# ── 1. Pull matched cohort + outcomes + follow-up ─────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

df = con.execute("""
    WITH c77 AS (
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN inp_claimsk_all i ON i.DSYSRTKY = o.DSYSRTKY,
        UNNEST([i.PRNCPAL_DGNS_CD, i.ADMTG_DGNS_CD,
                i.ICD_DGNS_CD1,  i.ICD_DGNS_CD2,  i.ICD_DGNS_CD3,
                i.ICD_DGNS_CD4,  i.ICD_DGNS_CD5,  i.ICD_DGNS_CD6,
                i.ICD_DGNS_CD7,  i.ICD_DGNS_CD8,  i.ICD_DGNS_CD9,
                i.ICD_DGNS_CD10, i.ICD_DGNS_CD11, i.ICD_DGNS_CD12,
                i.ICD_DGNS_CD13, i.ICD_DGNS_CD14, i.ICD_DGNS_CD15]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(i.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN out_claimsk_all oc ON oc.DSYSRTKY = o.DSYSRTKY,
        UNNEST([oc.PRNCPAL_DGNS_CD, oc.ICD_DGNS_CD1, oc.ICD_DGNS_CD2,
                oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4, oc.ICD_DGNS_CD5,
                oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7, oc.ICD_DGNS_CD8,
                oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10]) AS t(code)
        WHERE code LIKE 'C77%'
          AND TRY_STRPTIME(oc.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
        UNION ALL
        SELECT DISTINCT o.DSYSRTKY
        FROM opscc_cohort o
        JOIN car_linek_all cl ON cl.DSYSRTKY = o.DSYSRTKY
        WHERE cl.LINE_ICD_DGNS_CD LIKE 'C77%'
          AND TRY_STRPTIME(cl.THRU_DT, '%Y%m%d')
                  BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                      AND o.first_hnc_date + INTERVAL 90 DAY
    ),
    matched AS (
        SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
               p.age_at_dx, p.van_walraven_score
        FROM opscc_propensity p
        WHERE p.psm_matched = TRUE
          AND p.tx_group IN ('TORS only', 'CT/CRT only')
          AND p.DSYSRTKY NOT IN (SELECT DSYSRTKY FROM c77)
    ),
    mbsf_sum AS (
        SELECT
            sub.DSYSRTKY,
            -- Last day of the month before the first disenrollment month
            -- (first month after tx where BUYIN != '3' OR HMOIND NOT IN ('0','4'))
            -- If no disenrollment found, use last available month-end in MBSF
            COALESCE(
                MIN(CASE WHEN NOT (sub.buyin = '3' AND sub.hmoind IN ('0','4'))
                         THEN sub.mo_start END) - INTERVAL 1 DAY,
                MAX(sub.mo_end)
            ) AS last_ffs_date,
            MAX(CASE WHEN sub.DEATH_DT IS NOT NULL AND sub.DEATH_DT != ''
                     THEN TRY_STRPTIME(sub.DEATH_DT, '%Y%m%d') END)  AS death_date
        FROM (
            SELECT
                m.DSYSRTKY,
                m.DEATH_DT,
                make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)          AS mo_start,
                make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
                    + INTERVAL 1 MONTH - INTERVAL 1 DAY                  AS mo_end,
                CASE t.mo
                    WHEN 1  THEN m.BUYIN1  WHEN 2  THEN m.BUYIN2  WHEN 3  THEN m.BUYIN3
                    WHEN 4  THEN m.BUYIN4  WHEN 5  THEN m.BUYIN5  WHEN 6  THEN m.BUYIN6
                    WHEN 7  THEN m.BUYIN7  WHEN 8  THEN m.BUYIN8  WHEN 9  THEN m.BUYIN9
                    WHEN 10 THEN m.BUYIN10 WHEN 11 THEN m.BUYIN11 WHEN 12 THEN m.BUYIN12
                END AS buyin,
                CASE t.mo
                    WHEN 1  THEN m.HMOIND1  WHEN 2  THEN m.HMOIND2  WHEN 3  THEN m.HMOIND3
                    WHEN 4  THEN m.HMOIND4  WHEN 5  THEN m.HMOIND5  WHEN 6  THEN m.HMOIND6
                    WHEN 7  THEN m.HMOIND7  WHEN 8  THEN m.HMOIND8  WHEN 9  THEN m.HMOIND9
                    WHEN 10 THEN m.HMOIND10 WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
                END AS hmoind
            FROM mbsf_all m
            JOIN matched p ON m.DSYSRTKY = p.DSYSRTKY,
            UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mo)
            -- Start checking the month AFTER the treatment month
            WHERE make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
                      >= date_trunc('month', p.first_tx_date) + INTERVAL 1 MONTH
        ) sub
        GROUP BY sub.DSYSRTKY
    )
    SELECT
        p.DSYSRTKY, p.tx_group, p.first_tx_date,
        p.age_at_dx, p.van_walraven_score,
        s.death_date,
        s.last_ffs_date,
        o.has_dysphagia,
        DATEDIFF('day', p.first_tx_date, o.first_dysphagia_date)      AS days_dys,
        o.has_gtube,
        DATEDIFF('day', p.first_tx_date, o.first_gtube_date)          AS days_gt,
        o.has_tracheostomy,
        DATEDIFF('day', p.first_tx_date, o.first_trach_date)          AS days_tr
    FROM matched p
    JOIN mbsf_sum    s ON s.DSYSRTKY = p.DSYSRTKY
    JOIN opscc_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()

# ── 2. Derived fields ─────────────────────────────────────────────────────────
df['death_date']     = pd.to_datetime(df['death_date'])
df['last_ffs_date']  = pd.to_datetime(df['last_ffs_date'])
df['first_tx_date']  = pd.to_datetime(df['first_tx_date'])
df['censor_date'] = df['last_ffs_date']
mask = df['death_date'].notna() & (df['death_date'] < df['last_ffs_date'])
df.loc[mask, 'censor_date'] = df.loc[mask, 'death_date']
df['follow_up_days'] = (df['censor_date'] - df['first_tx_date']).dt.days
df['tors']           = (df['tx_group'] == 'TORS only').astype(int)

print(f"PSM-matched cohort (C77 excluded): {len(df):,}  |  "
      f"TORS: {df['tors'].sum():,}  CT/CRT: {(df['tors']==0).sum():,}")

# ── 3. Elixhauser tertiles (from this matched population) ─────────────────────
q1 = df['van_walraven_score'].quantile(1/3)
q2 = df['van_walraven_score'].quantile(2/3)
df['elix_grp'] = pd.cut(df['van_walraven_score'],
                         bins=[-np.inf, q1, q2, np.inf],
                         labels=['Low', 'Mid', 'High'])
print(f"Elixhauser tertiles: Low <= {q1:.0f}  |  Mid {q1:.0f}-{q2:.0f}  |  High > {q2:.0f}\n")

# ── 4. OR function ────────────────────────────────────────────────────────────
def compute_or(sub, has_col, days_col, cutoff):
    """
    Odds ratio for TORS vs CT/CRT at a fixed time cutoff (days) or anytime.

    Eligibility at time T:
      - Patient had the event within T days (event = 1), OR
      - Patient was followed >= T days without the event (event = 0)
    Patients censored before T without the event are excluded.
    Pre-existing patients (has_col == NULL) are always excluded.
    """
    valid = sub[sub[has_col].notna()].copy()

    if cutoff is None:
        elig = valid.copy()
        elig['ev'] = elig[has_col].astype(int)
    else:
        mask = (
            (valid['follow_up_days'] >= cutoff) |
            ((valid[has_col] == True) & (valid[days_col] <= cutoff))
        )
        elig = valid[mask].copy()
        elig['ev'] = ((elig[has_col] == True) & (elig[days_col] <= cutoff)).astype(int)

    t_  = elig[elig['tors'] == 1]
    c_  = elig[elig['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    pt = f"{100*et/nt:.1f}" if nt > 0 else "N/A"
    pc = f"{100*ec/nc:.1f}" if nc > 0 else "N/A"

    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return nt, et, pt, nc, ec, pc, float('nan'), float('nan'), float('nan'), float('nan')

    or_v    = (a * d) / (b * c)
    log_or  = np.log(or_v)
    se      = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi  = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return nt, et, pt, nc, ec, pc, or_v, lo, hi, p

# ── 5. Strata and outcomes ────────────────────────────────────────────────────
strata = [
    ('All matched (C77 excl)',  df),
    ('Age < 75',               df[df['age_at_dx'] < 75]),
    ('Age >= 75',              df[df['age_at_dx'] >= 75]),
    (f'Elix Low  (VW<={q1:.0f})',  df[df['elix_grp'] == 'Low']),
    (f'Elix Mid  (VW {q1:.0f}-{q2:.0f})', df[df['elix_grp'] == 'Mid']),
    (f'Elix High (VW>{q2:.0f})',   df[df['elix_grp'] == 'High']),
]

outcomes = [
    ('Dysphagia',    'has_dysphagia',    'days_dys'),
    ('G-tube',       'has_gtube',        'days_gt'),
    ('Tracheostomy', 'has_tracheostomy', 'days_tr'),
]

timepoints = [
    ('6-month',  182),
    ('1-year',   365),
    ('3-year',  1095),
    ('5-year',  1825),
    ('Anytime',  None),
]

# ── 6. Print results ──────────────────────────────────────────────────────────
W = 118
for out_lbl, has_col, days_col in outcomes:
    print(f"\n{'='*W}")
    print(f"  OUTCOME: {out_lbl}   (OR < 1 favors TORS)")
    print(f"{'='*W}")
    print(f"  {'Stratum':<26} {'Time':<9}"
          f" {'N(T)':>6} {'Ev(T)':>6} {'%(T)':>6}"
          f" {'N(C)':>6} {'Ev(C)':>6} {'%(C)':>6}"
          f"  {'OR':>6}  {'95% CI':<16}  {'p':>8}")
    print(f"  {'-'*W}")

    for s_lbl, sub in strata:
        first_row = True
        for tp_lbl, cutoff in timepoints:
            nt, et, pt, nc, ec, pc, or_v, lo, hi, p = compute_or(
                sub, has_col, days_col, cutoff)

            sl = s_lbl if first_row else ''
            first_row = False

            if np.isnan(or_v):
                or_str, ci_str, p_str = 'N/A', 'N/A', 'N/A'
            else:
                or_str = f"{or_v:.2f}"
                ci_str = f"({lo:.2f}-{hi:.2f})"
                p_str  = f"<0.0001" if p < 0.0001 else f"{p:.4f}"

            print(f"  {sl:<26} {tp_lbl:<9}"
                  f" {nt:>6,} {et:>6} {pt:>5}%"
                  f" {nc:>6,} {ec:>6} {pc:>5}%"
                  f"  {or_str:>6}  {ci_str:<16}  {p_str:>8}")

        print(f"  {'-'*W}")
