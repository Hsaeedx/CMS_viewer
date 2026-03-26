"""
iptw_sensitivity.py
IPTW sensitivity analysis ? side-by-side comparison with PSM for both comparisons.

Uses the same propensity model covariates as iptw_analysis.py.
Stabilized weights trimmed at 1st/99th percentile.
Weighted Cox PH (robust sandwich SE) for survival.
Weighted OR (sandwich variance, delta method) for outcomes.

Read-only ? does not write to database or overwrite parquet files.
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import warnings
warnings.filterwarnings('ignore')
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter

DB_PATH = r"F:\CMS\cms_data.duckdb"

ELIXHAUSER_FLAGS = [
    'chf','carit','valv','pcd','pvd','hypunc','hypc','para','ond','cpd',
    'diabunc','diabc','hypothy','rf','ld','pud','aids','lymph','metacanc',
    'solidtum','rheumd','coag','obes','wloss','fed','blane','dane',
    'alcohol','drug','psycho','depre'
]
FEATURE_COLS = ['age_at_dx', 'male', 'white', 'black', 'hispanic', 'asian_pi'] + ELIXHAUSER_FLAGS

COMPARISONS = [
    ('A', 'TORS alone', 'RT alone'),
    ('B', 'TORS + RT',  'CT/CRT'),
]
OUTCOMES_COLS = [
    ('Dysphagia',    'has_dysphagia',    'days_dys'),
    ('G-tube',       'has_gtube',        'days_gt'),
    ('Tracheostomy', 'has_tracheostomy', 'days_tr'),
]

# -- PSM reference results (from pipeline run, anytime / all matched, C77 excl) -
PSM_REF = {
    'A': {
        'n_pairs': 624,
        'hr': 0.567, 'hr_lo': 0.460, 'hr_hi': 0.699, 'hr_p': 0.0,
        'os5_tors': 73.2, 'os5_ctrl': 61.2,
        'n_imbalanced': 1,
        'or': {'Dysphagia': (1.15, 0.88, 1.50, 0.307),
               'G-tube':    (0.74, 0.56, 0.98, 0.036),
               'Tracheostomy': (0.45, 0.27, 0.75, 0.002)},
    },
    'B': {
        'n_pairs': 129,
        'hr': 0.536, 'hr_lo': 0.349, 'hr_hi': 0.824, 'hr_p': 0.004,
        'os5_tors': 66.5, 'os5_ctrl': 51.8,
        'n_imbalanced': 10,
        'or': {'Dysphagia': (1.07, 0.53, 2.16, 0.848),
               'G-tube':    (0.30, 0.17, 0.53, 0.0),
               'Tracheostomy': (0.80, 0.40, 1.60, 0.524)},
    },
}


# -- helpers --------------------------------------------------------------------
def smd_bin(a, b):
    p1, p2 = a.mean(), b.mean()
    d = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / d if d > 0 else 0.0

def smd_cont(a, b):
    d = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / d if d > 0 else 0.0

def smd_bin_w(x1, w1, x2, w2):
    p1 = np.average(x1, weights=w1)
    p2 = np.average(x2, weights=w2)
    d  = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / d if d > 0 else 0.0

def smd_cont_w(x1, w1, x2, w2):
    mu1 = np.average(x1, weights=w1)
    mu2 = np.average(x2, weights=w2)
    v1  = np.average((x1 - mu1)**2, weights=w1)
    v2  = np.average((x2 - mu2)**2, weights=w2)
    d   = np.sqrt((v1 + v2) / 2)
    return (mu1 - mu2) / d if d > 0 else 0.0

def eff_n(w):
    return w.sum()**2 / (w**2).sum()

def weighted_or(t_ev, t_w, c_ev, c_w):
    """Weighted OR with sandwich variance + delta-method 95% CI."""
    p_t = np.average(t_ev, weights=t_w)
    p_c = np.average(c_ev, weights=c_w)
    if not (0 < p_t < 1 and 0 < p_c < 1):
        return (float('nan'),) * 4
    or_v   = (p_t / (1-p_t)) / (p_c / (1-p_c))
    log_or = np.log(or_v)
    var_pt = np.sum(t_w**2 * (t_ev - p_t)**2) / t_w.sum()**2
    var_pc = np.sum(c_w**2 * (c_ev - p_c)**2) / c_w.sum()**2
    se     = np.sqrt(var_pt / (p_t*(1-p_t))**2 + var_pc / (p_c*(1-p_c))**2)
    lo     = np.exp(log_or - 1.96*se)
    hi     = np.exp(log_or + 1.96*se)
    p      = 2 * (1 - norm.cdf(abs(log_or / se)))
    return or_v, lo, hi, p

def pstr(p):
    if np.isnan(p): return "  N/A  "
    if p < 0.001:   return " <0.001"
    return f"{p:7.3f}"

def orstr(o, lo, hi, p):
    if np.isnan(o): return "N/A"
    return f"{o:.2f} ({lo:.2f}-{hi:.2f}) p={pstr(p).strip()}"


# -- load propensity table ------------------------------------------------------
print("Loading opscc_propensity...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")
df_all = con.execute("SELECT * FROM opscc_propensity").df()
print(f"  Loaded {len(df_all):,} patients\n")


# ==============================================================================
for comp, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*70}")
    print(f"  PSM vs IPTW ? COMPARISON {comp}: {tors_label}  vs  {ctrl_label}")
    print(f"{'#'*70}")
    psm = PSM_REF[comp]

    # -- 1. Compute IPTW weights ------------------------------------------------
    df = df_all[df_all['tx_group'].isin([tors_label, ctrl_label])].copy()
    df['treatment'] = (df['tx_group'] == tors_label).astype(int)
    n_tors = (df['treatment'] == 1).sum()
    n_ctrl = (df['treatment'] == 0).sum()

    df['male']     = (df['sex']  == 'Male').astype(int)
    df['white']    = (df['race'] == 'White').astype(int)
    df['black']    = (df['race'] == 'Black').astype(int)
    df['hispanic'] = (df['race'] == 'Hispanic').astype(int)
    df['asian_pi'] = (df['race'] == 'Asian/PI').astype(int)
    df[ELIXHAUSER_FLAGS]     = df[ELIXHAUSER_FLAGS].fillna(0).astype(int)
    df['van_walraven_score'] = df['van_walraven_score'].fillna(0)

    X = df[FEATURE_COLS].values
    y = df['treatment'].values
    scaler = StandardScaler()
    model  = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(scaler.fit_transform(X), y)

    df['ps']   = model.predict_proba(scaler.transform(X))[:, 1]
    auc        = roc_auc_score(y, df['ps'])
    p_t        = y.mean()
    df['iptw'] = np.where(df['treatment'] == 1,
                          p_t / df['ps'],
                          (1 - p_t) / (1 - df['ps']))
    lo_pct, hi_pct  = np.percentile(df['iptw'], [1, 99])
    df['iptw_trim'] = df['iptw'].clip(lo_pct, hi_pct)
    ess = eff_n(df['iptw_trim'])

    # -- 2. Balance -------------------------------------------------------------
    t_df = df[df['treatment'] == 1]
    c_df = df[df['treatment'] == 0]
    n_imb_iptw = 0
    imb_rows   = []
    for var in ['age_at_dx'] + ['male','white','black','hispanic','asian_pi'] + ELIXHAUSER_FLAGS:
        binary = var != 'age_at_dx'
        sb = abs((smd_bin if binary else smd_cont)(t_df[var], c_df[var]))
        sa = abs((smd_bin_w if binary else smd_cont_w)(
            t_df[var].values, t_df['iptw_trim'].values,
            c_df[var].values, c_df['iptw_trim'].values))
        if sa >= 0.10:
            n_imb_iptw += 1
            imb_rows.append({'variable': var, 'smd_before': round(sb, 3),
                             'smd_psm_after': 'see PSM', 'smd_iptw': round(sa, 3)})

    print(f"\nPS model AUC: {auc:.3f}  |  "
          f"N: {tors_label}={n_tors:,}  {ctrl_label}={n_ctrl:,}  "
          f"ESS={ess:.0f}  Weights [{lo_pct:.3f}-{hi_pct:.3f}]")
    print(f"\n{'-'*50}")
    print(f"  {'Metric':<32} {'PSM':>8}  {'IPTW':>8}")
    print(f"{'-'*50}")
    print(f"  {'N treated (TORS/surg arm)':<32} {psm['n_pairs']:>8,}  {n_tors:>8,}")
    print(f"  {'N control':<32} {psm['n_pairs']:>8,}  {n_ctrl:>8,}")
    print(f"  {'Effective sample size':<32} {'N/A':>8}  {ess:>8.0f}")
    print(f"  {'Variables imbalanced (SMD?0.10)':<32} {psm['n_imbalanced']:>8}  {n_imb_iptw:>8}")
    if imb_rows:
        print(f"\n  IPTW residual imbalance:")
        print(pd.DataFrame(imb_rows)[['variable','smd_before','smd_iptw']].to_string(index=False, col_space=10))

    # weight map for joining
    wmap = df.set_index('DSYSRTKY')['iptw_trim'].to_dict()

    # -- 3. Survival ------------------------------------------------------------
    surv = con.execute(f"""
        WITH cohort AS (
            SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
                   p.age_at_dx, p.van_walraven_score
            FROM opscc_propensity p
            WHERE p.tx_group IN ('{tors_label}', '{ctrl_label}')
        ),
        mbsf_sum AS (
            SELECT m.DSYSRTKY,
                   MAX(CAST(m.RFRNC_YR AS INTEGER)) AS last_enrl_year,
                   MAX(CASE WHEN m.DEATH_DT IS NOT NULL AND m.DEATH_DT != ''
                            THEN TRY_STRPTIME(m.DEATH_DT, '%Y%m%d') END) AS death_date
            FROM mbsf_all m
            JOIN cohort p ON m.DSYSRTKY = p.DSYSRTKY
            GROUP BY m.DSYSRTKY
        )
        SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
               p.age_at_dx, p.van_walraven_score,
               s.death_date,
               make_date(s.last_enrl_year, 12, 31) AS censor_date
        FROM cohort p
        JOIN mbsf_sum s ON p.DSYSRTKY = s.DSYSRTKY
    """).df()

    surv['iptw']       = surv['DSYSRTKY'].map(wmap)
    surv['event_date'] = surv['death_date'].combine_first(surv['censor_date'])
    surv['event']      = surv['death_date'].notna().astype(int)
    surv['t_days']     = (pd.to_datetime(surv['event_date'])
                          - pd.to_datetime(surv['first_tx_date'])).dt.days
    surv = surv[surv['first_tx_date'].notna() & (surv['t_days'] >= 0)].dropna(subset=['iptw'])
    surv['tors'] = (surv['tx_group'] == tors_label).astype(int)

    cph = CoxPHFitter()
    cph.fit(surv[['t_days','event','tors','age_at_dx','van_walraven_score','iptw']],
            duration_col='t_days', event_col='event',
            weights_col='iptw', robust=True)
    hr    = cph.summary.loc['tors', 'exp(coef)']
    hr_lo = cph.summary.loc['tors', 'exp(coef) lower 95%']
    hr_hi = cph.summary.loc['tors', 'exp(coef) upper 95%']
    hr_p  = cph.summary.loc['tors', 'p']

    t_surv = surv[surv['tors'] == 1]
    c_surv = surv[surv['tors'] == 0]
    kmf_t  = KaplanMeierFitter()
    kmf_c  = KaplanMeierFitter()
    kmf_t.fit(t_surv['t_days'], t_surv['event'], weights=t_surv['iptw'])
    kmf_c.fit(c_surv['t_days'], c_surv['event'], weights=c_surv['iptw'])
    s5_t = kmf_t.survival_function_at_times(1825).iloc[0] * 100
    s5_c = kmf_c.survival_function_at_times(1825).iloc[0] * 100

    psm_hr_str  = (f"{psm['hr']:.3f} ({psm['hr_lo']:.3f}-{psm['hr_hi']:.3f})"
                   f" p={pstr(psm['hr_p']).strip()}")
    iptw_hr_str = f"{hr:.3f} ({hr_lo:.3f}-{hr_hi:.3f}) p={pstr(hr_p).strip()}"

    print(f"\n{'-'*70}")
    print(f"  SURVIVAL")
    print(f"{'-'*70}")
    print(f"  {'HR (95% CI) for TORS/surg':<32} {psm_hr_str:>26}   {iptw_hr_str}")
    print(f"  {'5-yr OS ? TORS/surg arm':<32} {psm['os5_tors']:>25.1f}%   {s5_t:.1f}%")
    print(f"  {'5-yr OS ? control arm':<32} {psm['os5_ctrl']:>25.1f}%   {s5_c:.1f}%")

    # -- 4. Outcomes ------------------------------------------------------------
    outc = con.execute(f"""
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
              AND TRY_STRPTIME(i.THRU_DT,'%Y%m%d')
                      BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                          AND o.first_hnc_date + INTERVAL 90 DAY
            UNION ALL
            SELECT DISTINCT o.DSYSRTKY
            FROM opscc_cohort o
            JOIN out_claimsk_all oc ON oc.DSYSRTKY = o.DSYSRTKY,
            UNNEST([oc.PRNCPAL_DGNS_CD, oc.ICD_DGNS_CD1, oc.ICD_DGNS_CD2,
                    oc.ICD_DGNS_CD3,  oc.ICD_DGNS_CD4,  oc.ICD_DGNS_CD5,
                    oc.ICD_DGNS_CD6,  oc.ICD_DGNS_CD7,  oc.ICD_DGNS_CD8,
                    oc.ICD_DGNS_CD9,  oc.ICD_DGNS_CD10]) AS t(code)
            WHERE code LIKE 'C77%'
              AND TRY_STRPTIME(oc.THRU_DT,'%Y%m%d')
                      BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                          AND o.first_hnc_date + INTERVAL 90 DAY
            UNION ALL
            SELECT DISTINCT o.DSYSRTKY
            FROM opscc_cohort o
            JOIN car_linek_all cl ON cl.DSYSRTKY = o.DSYSRTKY
            WHERE cl.LINE_ICD_DGNS_CD LIKE 'C77%'
              AND TRY_STRPTIME(cl.THRU_DT,'%Y%m%d')
                      BETWEEN o.first_hnc_date - INTERVAL 90 DAY
                          AND o.first_hnc_date + INTERVAL 90 DAY
        ),
        cohort AS (
            SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
                   p.age_at_dx, p.van_walraven_score
            FROM opscc_propensity p
            WHERE p.tx_group IN ('{tors_label}', '{ctrl_label}')
              AND p.DSYSRTKY NOT IN (SELECT DSYSRTKY FROM c77)
        ),
        mbsf_sum AS (
            SELECT sub.DSYSRTKY,
                COALESCE(
                    MIN(CASE WHEN NOT (sub.buyin='3' AND sub.hmoind IN ('0','4'))
                             THEN sub.mo_start END) - INTERVAL 1 DAY,
                    MAX(sub.mo_end)
                ) AS last_ffs_date,
                MAX(CASE WHEN sub.DEATH_DT IS NOT NULL AND sub.DEATH_DT != ''
                         THEN TRY_STRPTIME(sub.DEATH_DT,'%Y%m%d') END) AS death_date
            FROM (
                SELECT m.DSYSRTKY, m.DEATH_DT,
                    make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)      AS mo_start,
                    make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
                        + INTERVAL 1 MONTH - INTERVAL 1 DAY              AS mo_end,
                    CASE t.mo WHEN 1 THEN m.BUYIN1  WHEN 2 THEN m.BUYIN2
                               WHEN 3 THEN m.BUYIN3  WHEN 4 THEN m.BUYIN4
                               WHEN 5 THEN m.BUYIN5  WHEN 6 THEN m.BUYIN6
                               WHEN 7 THEN m.BUYIN7  WHEN 8 THEN m.BUYIN8
                               WHEN 9 THEN m.BUYIN9  WHEN 10 THEN m.BUYIN10
                               WHEN 11 THEN m.BUYIN11 WHEN 12 THEN m.BUYIN12
                    END AS buyin,
                    CASE t.mo WHEN 1 THEN m.HMOIND1  WHEN 2 THEN m.HMOIND2
                               WHEN 3 THEN m.HMOIND3  WHEN 4 THEN m.HMOIND4
                               WHEN 5 THEN m.HMOIND5  WHEN 6 THEN m.HMOIND6
                               WHEN 7 THEN m.HMOIND7  WHEN 8 THEN m.HMOIND8
                               WHEN 9 THEN m.HMOIND9  WHEN 10 THEN m.HMOIND10
                               WHEN 11 THEN m.HMOIND11 WHEN 12 THEN m.HMOIND12
                    END AS hmoind
                FROM mbsf_all m
                JOIN cohort p ON m.DSYSRTKY = p.DSYSRTKY,
                UNNEST([1,2,3,4,5,6,7,8,9,10,11,12]) AS t(mo)
                WHERE make_date(CAST(m.RFRNC_YR AS INTEGER), t.mo, 1)
                          >= date_trunc('month', p.first_tx_date) + INTERVAL 1 MONTH
            ) sub
            GROUP BY sub.DSYSRTKY
        )
        SELECT p.DSYSRTKY, p.tx_group, p.first_tx_date,
               s.death_date, s.last_ffs_date,
               o.has_dysphagia,
               DATEDIFF('day', p.first_tx_date, o.first_dysphagia_date) AS days_dys,
               o.has_gtube,
               DATEDIFF('day', p.first_tx_date, o.first_gtube_date)     AS days_gt,
               o.has_tracheostomy,
               DATEDIFF('day', p.first_tx_date, o.first_trach_date)     AS days_tr
        FROM cohort p
        JOIN mbsf_sum       s ON s.DSYSRTKY = p.DSYSRTKY
        JOIN opscc_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
    """).df()

    outc['iptw']          = outc['DSYSRTKY'].map(wmap)
    outc['death_date']    = pd.to_datetime(outc['death_date'])
    outc['last_ffs_date'] = pd.to_datetime(outc['last_ffs_date'])
    outc['first_tx_date'] = pd.to_datetime(outc['first_tx_date'])
    outc['censor_date']   = outc['last_ffs_date']
    mask = outc['death_date'].notna() & (outc['death_date'] < outc['last_ffs_date'])
    outc.loc[mask, 'censor_date'] = outc.loc[mask, 'death_date']
    outc['follow_up_days'] = (outc['censor_date'] - outc['first_tx_date']).dt.days
    outc['tors'] = (outc['tx_group'] == tors_label).astype(int)
    outc = outc.dropna(subset=['iptw'])

    print(f"\n{'-'*70}")
    print(f"  OUTCOMES (Anytime, C77 excluded)   N={len(outc):,}")
    print(f"{'-'*70}")
    print(f"  {'Outcome':<16}  {'PSM OR (95% CI) p':<32}  {'IPTW OR (95% CI) p':<32}")
    print(f"  {'-'*78}")

    for out_lbl, has_col, _ in OUTCOMES_COLS:
        valid = outc[outc[has_col].notna()].copy()
        valid['ev'] = valid[has_col].astype(int)
        t_ = valid[valid['tors'] == 1]
        c_ = valid[valid['tors'] == 0]
        or_w, lo_w, hi_w, p_w = weighted_or(
            t_['ev'].values, t_['iptw'].values,
            c_['ev'].values, c_['iptw'].values)
        psm_o, psm_lo, psm_hi, psm_p = psm['or'][out_lbl]
        print(f"  {out_lbl:<16}  {orstr(psm_o,psm_lo,psm_hi,psm_p):<32}  {orstr(or_w,lo_w,hi_w,p_w)}")

con.close()
print(f"\n{'-'*70}")
print("  Done. PSM reference values are from the most recent pipeline run.")
print(f"{'-'*70}")

