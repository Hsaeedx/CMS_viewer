"""
outcomes_analysis.py
Odds ratios for dysphagia, G-tube, and tracheostomy
at 6-month, 1-year, 3-year, 5-year, and anytime
Stratified by age group (<75 / >=75) and Elixhauser tertile (Low/Mid/High)
Population: PSM-matched cohort, C77 (nodal disease) excluded

Two comparisons:
  Comparison A: TORS alone vs RT alone
  Comparison B: TORS + RT  vs CT/CRT

Reads from pre-built SQL tables:
  opscc_survival    — death date + Dec-31 censor (step 11)
  opscc_ffs_dates   — FFS dropout date per patient (step 12)
  opscc_c77         — C77 exclusion set (step 10)
  opscc_propensity  — PSM match flags (step 13)
  opscc_outcomes    — dysphagia / G-tube / trach flags (step 9)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

DB_PATH = r"F:\CMS\cms_data.duckdb"

COMPARISONS = [
    ('A', 'psm_matched_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'TORS + RT',  'CT/CRT'),
]

OUTCOMES = [
    ('Dysphagia',    'has_dysphagia',    'days_dys'),
    ('G-tube',       'has_gtube',        'days_gt'),
    ('Tracheostomy', 'has_tracheostomy', 'days_tr'),
]

TIMEPOINTS = [
    ('6-month',  182),
    ('1-year',   365),
    ('3-year',  1095),
    ('5-year',  1825),
    ('Anytime',  None),
]


def compute_or(sub, has_col, days_col, cutoff):
    valid = sub[sub[has_col].notna()].copy()
    if cutoff is None:
        elig     = valid.copy()
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

    or_v   = (a * d) / (b * c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return nt, et, pt, nc, ec, pc, or_v, lo, hi, p


con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

for comp, match_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*70}")
    print(f"  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}  (C77 excluded)")
    print(f"{'#'*70}")

    df = con.execute(f"""
        SELECT
            s.DSYSRTKY, s.tx_group, s.first_tx_date,
            s.age_at_dx, s.van_walraven_score,
            f.ffs_censor_date,
            DATEDIFF('day', s.first_tx_date, f.ffs_censor_date)          AS follow_up_days,
            o.has_dysphagia,
            DATEDIFF('day', s.first_tx_date, o.first_dysphagia_date)     AS days_dys,
            o.has_gtube,
            DATEDIFF('day', s.first_tx_date, o.first_gtube_date)         AS days_gt,
            o.has_tracheostomy,
            DATEDIFF('day', s.first_tx_date, o.first_trach_date)         AS days_tr
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        JOIN opscc_ffs_dates  f USING (DSYSRTKY)
        JOIN opscc_outcomes   o USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
    """).df()

    df['tors'] = (df['tx_group'] == tors_label).astype(int)

    print(f"N (C77 excl): {len(df):,}  |  {tors_label}: {df['tors'].sum():,}  "
          f"{ctrl_label}: {(df['tors']==0).sum():,}")

    q1 = df['van_walraven_score'].quantile(1/3)
    q2 = df['van_walraven_score'].quantile(2/3)
    df['elix_grp'] = pd.cut(df['van_walraven_score'],
                             bins=[-np.inf, q1, q2, np.inf],
                             labels=['Low', 'Mid', 'High'])

    strata = [
        ('All matched (C77 excl)',          df),
        ('Age < 75',                        df[df['age_at_dx'] < 75]),
        ('Age \u2265 75',                   df[df['age_at_dx'] >= 75]),
        (f'Elix Low  (VW\u2264{q1:.0f})',   df[df['elix_grp'] == 'Low']),
        (f'Elix Mid  (VW {q1:.0f}\u2013{q2:.0f})', df[df['elix_grp'] == 'Mid']),
        (f'Elix High (VW>{q2:.0f})',        df[df['elix_grp'] == 'High']),
    ]

    W = 118
    for out_lbl, has_col, days_col in OUTCOMES:
        print(f"\n{'='*W}")
        print(f"  OUTCOME: {out_lbl}   (OR < 1 favors {tors_label})")
        print(f"{'='*W}")
        print(f"  {'Stratum':<26} {'Time':<9}"
              f" {'N(T)':>6} {'Ev(T)':>6} {'%(T)':>6}"
              f" {'N(C)':>6} {'Ev(C)':>6} {'%(C)':>6}"
              f"  {'OR':>6}  {'95% CI':<16}  {'p':>8}")
        print(f"  {'-'*W}")

        for s_lbl, sub in strata:
            first_row = True
            for tp_lbl, cutoff in TIMEPOINTS:
                nt, et, pt, nc, ec, pc, or_v, lo, hi, p = compute_or(
                    sub, has_col, days_col, cutoff)

                sl = s_lbl if first_row else ''
                first_row = False

                if np.isnan(or_v):
                    or_str, ci_str, p_str = 'N/A', 'N/A', 'N/A'
                else:
                    or_str = f"{or_v:.2f}"
                    ci_str = f"({lo:.2f}\u2013{hi:.2f})"
                    p_str  = "<0.0001" if p < 0.0001 else f"{p:.4f}"

                print(f"  {sl:<26} {tp_lbl:<9}"
                      f" {nt:>6,} {et:>6} {pt:>5}%"
                      f" {nc:>6,} {ec:>6} {pc:>5}%"
                      f"  {or_str:>6}  {ci_str:<16}  {p_str:>8}")

            print(f"  {'-'*W}")

con.close()
