"""
make_stroke_table.py

Exports stroke + SLP analysis results to a formatted Excel workbook:
  F:\CMS\projects\stroke_SLP\stroke_slp_tables.xlsx

Sheets:
  1. Table 1  — Cohort characteristics (matched vs unmatched)
  2. Table 2  — Covariate balance (pre- and post-match)
  3. Table 3  — OR table: all outcomes, all matched patients
  4. Table 4  — OR table stratified by stroke type
  5. Table 5  — OR table stratified by age group
  6. Table 6  — Cox HR table (mortality, recurrent stroke, aspiration)
  7. Table 7  — Cost analysis
  8. Table 8  — KM survival / cumulative incidence at 180d/365d/1095d/1825d
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

DB_PATH  = r"F:\CMS\cms_data.duckdb"
OUT_PATH = r"F:\CMS\projects\stroke_SLP\stroke_slp_tables.xlsx"

MAX_FOLLOW = 1825  # 5 years
TIMEPOINTS = [180, 365, 1095, 1825]

# ── Style helpers ──────────────────────────────────────────────────────────────

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
SUBHDR_FILL = PatternFill("solid", fgColor="2E75B6")
ALT_FILL    = PatternFill("solid", fgColor="D6E4F0")
BOLD        = Font(bold=True)
WHITE_BOLD  = Font(bold=True, color="FFFFFF")
CENTER      = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT        = Alignment(horizontal="left",   vertical="center")
THIN        = Side(style="thin",   color="BFBFBF")
THICK       = Side(style="medium", color="1F4E79")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(cell, sub=False):
    cell.fill  = SUBHDR_FILL if sub else HEADER_FILL
    cell.font  = WHITE_BOLD
    cell.alignment = CENTER
    cell.border = THIN_BORDER


def style_data(cell, alt=False):
    cell.fill      = ALT_FILL if alt else PatternFill()
    cell.alignment = LEFT
    cell.border    = THIN_BORDER


def autofit(ws, min_width=8, max_width=40):
    for col in ws.columns:
        length = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(length + 2, min_width), max_width)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Connecting ...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")

df = con.execute("""
    SELECT
        p.DSYSRTKY,
        p.slp_any_30d       AS slp,
        p.slp_group,
        p.psm_matched,
        p.age_at_adm,
        p.sex,
        p.race,
        p.stroke_type,
        p.adm_year,
        p.index_los,
        p.dysphagia_poa,
        p.aspiration_poa,
        p.mech_vent,
        p.peg_placed,
        p.trach_placed,
        p.prior_stroke,
        p.van_walraven_score,
        p.afib,
        p.hypertension,
        o.death_date,
        o.days_to_death,
        o.days_to_readmit,
        o.n_readmissions_1825d,
        o.days_to_recur_stroke,
        o.days_to_aspiration,
        o.has_aspiration,
        o.days_to_dysphagia,
        o.has_dysphagia,
        o.days_to_gtube,
        o.has_gtube,
        o.days_to_snf,
        o.snf_30d,
        o.hha_90d,
        o.total_pmt_1825d,
        o.snf_pmt_1825d,
        o.hha_pmt_1825d,
        c.index_pmt,
        c.index_dschg_date
    FROM stroke_propensity p
    JOIN stroke_outcomes   o ON o.DSYSRTKY = p.DSYSRTKY
    JOIN stroke_cohort     c ON c.DSYSRTKY = p.DSYSRTKY
""").df()
con.close()

# ── Derived fields ─────────────────────────────────────────────────────────────

# Censor at MAX_FOLLOW days
df['censor_days'] = np.where(
    df['days_to_death'].notna(),
    df['days_to_death'].clip(upper=MAX_FOLLOW),
    MAX_FOLLOW
)

# Mortality at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'died_{label}'] = (df['days_to_death'].notna() & (df['days_to_death'] <= days)).astype(int)

# Readmission at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'readmit_{label}'] = (df['days_to_readmit'].notna() & (df['days_to_readmit'] <= days)).astype(int)

# Recurrent stroke at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'recur_{label}'] = (df['days_to_recur_stroke'].notna() & (df['days_to_recur_stroke'] <= days)).astype(int)

# Aspiration PNA at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'asp_{label}'] = (df['days_to_aspiration'].notna() & (df['days_to_aspiration'] <= days)).astype(int)

# Dysphagia at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'dysphagia_{label}'] = (df['days_to_dysphagia'].notna() & (df['days_to_dysphagia'] <= days)).astype(int)

# G-tube at each time point
for days, label in [(180, '180d'), (365, '365d'), (1095, '1095d'), (1825, '1825d')]:
    df[f'gtube_{label}'] = (df['days_to_gtube'].notna() & (df['days_to_gtube'] <= days)).astype(int)

# Elixhauser tertiles
q1 = df['van_walraven_score'].quantile(1/3)
q2 = df['van_walraven_score'].quantile(2/3)
df['elix_grp'] = pd.cut(df['van_walraven_score'],
                          bins=[-np.inf, q1, q2, np.inf],
                          labels=['Low', 'Mid', 'High'])

matched_df = df[df['psm_matched'] == True].copy()
print(f"Loaded:  total={len(df):,}  matched={len(matched_df):,}")


# ── OR function ────────────────────────────────────────────────────────────────
def compute_or(sub, ev_col, days_col, cutoff):
    if cutoff is None:
        elig = sub.copy()
        elig['ev'] = elig[ev_col].astype(int)
    else:
        mask = (
            (sub['censor_days'] >= cutoff) |
            (sub[ev_col].astype(bool) & (sub[days_col] <= cutoff))
        )
        elig = sub[mask].copy()
        elig['ev'] = (elig[ev_col].astype(bool) & (elig[days_col] <= cutoff)).astype(int)

    t_ = elig[elig['slp'] == 1]
    c_ = elig[elig['slp'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())

    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return dict(n_slp=nt, ev_slp=et, pct_slp="N/A",
                    n_noslp=nc, ev_noslp=ec, pct_noslp="N/A",
                    OR="N/A", CI="N/A", p="N/A")

    or_v   = (a * d) / (b * c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)

    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    return dict(
        n_slp=nt, ev_slp=et, pct_slp=f"{100*et/nt:.1f}%",
        n_noslp=nc, ev_noslp=ec, pct_noslp=f"{100*ec/nc:.1f}%",
        OR=f"{or_v:.2f}", CI=f"[{lo:.2f}, {hi:.2f}]", p=p_str
    )


# ── Build OR DataFrame ─────────────────────────────────────────────────────────
def build_or_rows(sub, strat_label):
    outcomes = [
        # (display label,   ev_col template,      days_col,               cutoffs)
        ("Mortality",        "died_{tp}",           "days_to_death",        TIMEPOINTS),
        ("Readmission",      "readmit_{tp}",        "days_to_readmit",      TIMEPOINTS),
        ("Recurrent Stroke", "recur_{tp}",          "days_to_recur_stroke", TIMEPOINTS),
        ("Aspiration PNA",   "asp_{tp}",            "days_to_aspiration",   TIMEPOINTS),
        ("Dysphagia Dx",     "dysphagia_{tp}",      "days_to_dysphagia",    TIMEPOINTS),
        ("G-tube",           "gtube_{tp}",          "days_to_gtube",        TIMEPOINTS),
        ("SNF 30d",          "snf_30d",             "days_to_snf",          [None]),
        ("HHA 90d",          "hha_90d",             "days_to_snf",          [None]),
    ]
    rows = []
    for label_o, ev_tmpl, days_col, cutoffs in outcomes:
        for cutoff in cutoffs:
            if '{tp}' in ev_tmpl:
                tp_label = f"{cutoff}d" if cutoff else "any"
                ev_col = ev_tmpl.replace('{tp}', tp_label)
                display = f"{label_o} {tp_label}"
            else:
                ev_col = ev_tmpl
                display = label_o
            d = compute_or(sub, ev_col, days_col, cutoff)
            rows.append({"Stratum": strat_label, "Outcome": display, **d})
    return rows


# ── Write Excel ────────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
wb.remove(wb.active)


def write_or_sheet(wb, sheet_name, rows):
    ws = wb.create_sheet(sheet_name)
    cols = ["Stratum", "Outcome",
            "n (SLP)", "Events (SLP)", "% (SLP)",
            "n (No SLP)", "Events (No SLP)", "% (No SLP)",
            "OR", "95% CI", "p"]
    col_keys = ["Stratum", "Outcome",
                "n_slp", "ev_slp", "pct_slp",
                "n_noslp", "ev_noslp", "pct_noslp",
                "OR", "CI", "p"]

    for ci, h in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        style_header(cell)

    for ri, row in enumerate(rows, 2):
        alt = ri % 2 == 0
        for ci, key in enumerate(col_keys, 1):
            cell = ws.cell(row=ri, column=ci, value=row.get(key, ""))
            style_data(cell, alt)

    autofit(ws)
    ws.freeze_panes = "A2"


# ── Table 1: Cohort characteristics ───────────────────────────────────────────

def _pct_str(sub, col):
    n = sub[col].notna().sum()
    s = sub[col].fillna(0).sum()
    return f"{int(s):,} ({100*s/n:.1f}%)" if n > 0 else "N/A"

def _mean_sd(sub, col):
    v = sub[col].dropna()
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 0 else "N/A"

def _cat_pct(sub, col, val):
    n = len(sub)
    s = (sub[col] == val).sum()
    return f"{s:,} ({100*s/n:.1f}%)" if n > 0 else "N/A"


def write_table1(wb, df, matched_df):
    ws = wb.create_sheet("Table1_Cohort", 0)

    hdr = ["Variable",
           "All — SLP", "All — No SLP",
           "Matched — SLP", "Matched — No SLP"]
    for ci, h in enumerate(hdr, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    a_slp  = df[df['slp'] == 1]
    a_ctrl = df[df['slp'] == 0]
    m_slp  = matched_df[matched_df['slp'] == 1]
    m_ctrl = matched_df[matched_df['slp'] == 0]

    def row4(label, f_slp, f_ctrl, f_mslp, f_mctrl):
        return [label, f_slp(a_slp), f_ctrl(a_ctrl), f_mslp(m_slp), f_mctrl(m_ctrl)]

    def cont(col):
        return lambda s: _mean_sd(s, col)

    def bin_(col):
        return lambda s: _pct_str(s, col)

    def cat(col, val):
        return lambda s: _cat_pct(s, col, val)

    rows1 = [
        ["N", len(a_slp), len(a_ctrl), len(m_slp), len(m_ctrl)],
        row4("Age, mean (SD)",                cont('age_at_adm'), cont('age_at_adm'), cont('age_at_adm'), cont('age_at_adm')),
        row4("Male sex, n (%)",                cat('sex','Male'),  cat('sex','Male'),  cat('sex','Male'),  cat('sex','Male')),
        row4("Race — White, n (%)",            cat('race','White'), cat('race','White'), cat('race','White'), cat('race','White')),
        row4("Race — Black, n (%)",            cat('race','Black'), cat('race','Black'), cat('race','Black'), cat('race','Black')),
        row4("Race — Hispanic, n (%)",         cat('race','Hispanic'), cat('race','Hispanic'), cat('race','Hispanic'), cat('race','Hispanic')),
        row4("Stroke type — Ischemic, n (%)",  cat('stroke_type','Ischemic'),  cat('stroke_type','Ischemic'),  cat('stroke_type','Ischemic'),  cat('stroke_type','Ischemic')),
        row4("Stroke type — ICH, n (%)",       cat('stroke_type','ICH'),       cat('stroke_type','ICH'),       cat('stroke_type','ICH'),       cat('stroke_type','ICH')),
        row4("Stroke type — SAH, n (%)",       cat('stroke_type','SAH'),       cat('stroke_type','SAH'),       cat('stroke_type','SAH'),       cat('stroke_type','SAH')),
        row4("Index LOS, mean (SD) days",      cont('index_los'), cont('index_los'), cont('index_los'), cont('index_los')),
        row4("van Walraven score, mean (SD)",  cont('van_walraven_score'), cont('van_walraven_score'), cont('van_walraven_score'), cont('van_walraven_score')),
        row4("Dysphagia POA, n (%)",           bin_('dysphagia_poa'),  bin_('dysphagia_poa'),  bin_('dysphagia_poa'),  bin_('dysphagia_poa')),
        row4("Aspiration POA, n (%)",          bin_('aspiration_poa'), bin_('aspiration_poa'), bin_('aspiration_poa'), bin_('aspiration_poa')),
        row4("Mechanical ventilation, n (%)",  bin_('mech_vent'),      bin_('mech_vent'),      bin_('mech_vent'),      bin_('mech_vent')),
        row4("PEG placed, n (%)",              bin_('peg_placed'),     bin_('peg_placed'),     bin_('peg_placed'),     bin_('peg_placed')),
        row4("Tracheostomy, n (%)",            bin_('trach_placed'),   bin_('trach_placed'),   bin_('trach_placed'),   bin_('trach_placed')),
        row4("Prior stroke, n (%)",            bin_('prior_stroke'),   bin_('prior_stroke'),   bin_('prior_stroke'),   bin_('prior_stroke')),
        row4("Atrial fibrillation, n (%)",     bin_('afib'),           bin_('afib'),           bin_('afib'),           bin_('afib')),
        row4("Hypertension, n (%)",            bin_('hypertension'),   bin_('hypertension'),   bin_('hypertension'),   bin_('hypertension')),
    ]

    for ri, row in enumerate(rows1, 2):
        alt = ri % 2 == 0
        for ci, val in enumerate(row, 1):
            style_data(ws.cell(row=ri, column=ci, value=val), alt)

    autofit(ws)
    ws.freeze_panes = "B2"


# ── Table 2: Covariate balance (pre- and post-match SMDs) ─────────────────────

def _smd(x0, x1):
    mu0, mu1 = x0.mean(), x1.mean()
    s0, s1   = x0.std(),  x1.std()
    pooled   = np.sqrt((s0**2 + s1**2) / 2)
    return abs(mu1 - mu0) / pooled if pooled > 0 else 0.0


def write_table2(wb, df, matched_df):
    ws = wb.create_sheet("Table2_Balance", 1)

    hdr = ["Variable", "Pre-match SMD", "Post-match SMD", "Balance OK (<0.1)"]
    for ci, h in enumerate(hdr, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    balance_vars = [
        ('age_at_adm',        'Age at admission'),
        ('index_los',         'Index LOS'),
        ('van_walraven_score','van Walraven score'),
        ('dysphagia_poa',     'Dysphagia POA'),
        ('aspiration_poa',    'Aspiration POA'),
        ('mech_vent',         'Mechanical ventilation'),
        ('peg_placed',        'PEG placed'),
        ('trach_placed',      'Tracheostomy'),
        ('prior_stroke',      'Prior stroke'),
        ('afib',              'Atrial fibrillation'),
        ('hypertension',      'Hypertension'),
    ]

    for ri, (col, label) in enumerate(balance_vars, 2):
        if col not in df.columns:
            continue
        pre_slp  = df[df['slp'] == 1][col].fillna(0)
        pre_ctrl = df[df['slp'] == 0][col].fillna(0)
        post_slp  = matched_df[matched_df['slp'] == 1][col].fillna(0)
        post_ctrl = matched_df[matched_df['slp'] == 0][col].fillna(0)

        smd_pre  = _smd(pre_ctrl,  pre_slp)
        smd_post = _smd(post_ctrl, post_slp)
        ok = "Yes" if smd_post < 0.1 else "No  ***"

        alt = ri % 2 == 0
        for ci, val in enumerate([label, round(smd_pre, 3), round(smd_post, 3), ok], 1):
            style_data(ws.cell(row=ri, column=ci, value=val), alt)

    autofit(ws)
    ws.freeze_panes = "A2"


# Build Tables 1 and 2
print("Building Table 1 (cohort characteristics) ...")
write_table1(wb, df, matched_df)

print("Building Table 2 (covariate balance) ...")
write_table2(wb, df, matched_df)

# Table 3: All matched
print("Building Table 3 (all matched) ...")
rows_all = build_or_rows(matched_df, "All matched")
write_or_sheet(wb, "Table3_All", rows_all)

# Table 4: By stroke type
print("Building Table 4 (by stroke type) ...")
rows_stype = []
for stype in ['Ischemic', 'ICH', 'SAH', 'Unspecified']:
    sub = matched_df[matched_df['stroke_type'] == stype]
    if len(sub) > 20:
        rows_stype.extend(build_or_rows(sub, stype))
write_or_sheet(wb, "Table4_ByStrokeType", rows_stype)

# Table 5: By age group
print("Building Table 5 (by age group) ...")
rows_age = []
for label, sub in [('<75', matched_df[matched_df['age_at_adm'] < 75]),
                    ('75-84', matched_df[(matched_df['age_at_adm'] >= 75) & (matched_df['age_at_adm'] < 85)]),
                    ('85+', matched_df[matched_df['age_at_adm'] >= 85])]:
    rows_age.extend(build_or_rows(sub, f"Age {label}"))
write_or_sheet(wb, "Table5_ByAge", rows_age)

# Table 6: Cox PH
print("Building Table 6 (Cox PH) ...")
ws6 = wb.create_sheet("Table6_CoxPH")
cox_cols = ["Outcome", "HR", "95% CI Lower", "95% CI Upper", "p", "N", "Events"]
for ci, h in enumerate(cox_cols, 1):
    style_header(ws6.cell(row=1, column=ci, value=h))

cox_outcomes = [
    ("All-cause mortality (5yr)", "died_1825d",  "censor_days"),
    ("Recurrent stroke (5yr)",    "recur_1825d", "days_to_recur_stroke"),
    ("Aspiration PNA (5yr)",      "asp_1825d",   "days_to_aspiration"),
    ("Dysphagia Dx (5yr)",        "dysphagia_1825d", "days_to_dysphagia"),
    ("G-tube (5yr)",              "gtube_1825d", "days_to_gtube"),
]
covariates = ['slp', 'age_at_adm', 'van_walraven_score', 'dysphagia_poa',
              'aspiration_poa', 'mech_vent', 'peg_placed', 'trach_placed',
              'index_los']

for ri, (label_c, ev_col, dur_col) in enumerate(cox_outcomes, 2):
    sub = matched_df.copy()
    sub['_dur'] = sub.get(dur_col, sub['censor_days']).fillna(sub['censor_days']).astype(float)
    sub['_dur'] = sub['_dur'].clip(lower=0.5)
    sub['_ev']  = sub[ev_col].astype(int)
    cox_df = sub[['_dur', '_ev'] + covariates].dropna()
    cox_df = cox_df.rename(columns={'_dur': 'duration', '_ev': 'event'})
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='duration', event_col='event', show_progress=False)
        row = cph.summary.loc['slp']
        hr   = round(float(np.exp(row['coef'])), 2)
        lo95 = round(float(np.exp(row['coef lower 95%'])), 2)
        hi95 = round(float(np.exp(row['coef upper 95%'])), 2)
        p    = round(float(row['p']), 4)
        vals = [label_c, hr, lo95, hi95, p, len(cox_df), int(cox_df['event'].sum())]
    except Exception as e:
        vals = [label_c, "ERROR", str(e), "", "", "", ""]
    alt = ri % 2 == 0
    for ci, v in enumerate(vals, 1):
        style_data(ws6.cell(row=ri, column=ci, value=v), alt)

autofit(ws6)
ws6.freeze_panes = "A2"

# Table 7: Cost
print("Building Table 7 (costs) ...")
ws7 = wb.create_sheet("Table7_Costs")
cost_cols = ["Group", "N",
             "Mean 5yr Cost ($)", "Median 5yr Cost ($)",
             "Mean Index Cost ($)", "Median Index Cost ($)",
             "Mann-Whitney p (5yr)"]
for ci, h in enumerate(cost_cols, 1):
    style_header(ws7.cell(row=1, column=ci, value=h))

cost_rows = []
slp_c     = matched_df[matched_df['slp'] == 1]['total_pmt_1825d'].dropna()
noslp_c   = matched_df[matched_df['slp'] == 0]['total_pmt_1825d'].dropna()
slp_idx   = matched_df[matched_df['slp'] == 1]['index_pmt'].dropna()
noslp_idx = matched_df[matched_df['slp'] == 0]['index_pmt'].dropna()

_, mw_p = mannwhitneyu(slp_c, noslp_c, alternative='two-sided')
p_str = "<0.001" if mw_p < 0.001 else f"{mw_p:.3f}"

cost_rows.append(["SLP",    len(slp_c),
                  round(slp_c.mean(), 0),   round(slp_c.median(), 0),
                  round(slp_idx.mean(), 0), round(slp_idx.median(), 0),
                  p_str])
cost_rows.append(["No SLP", len(noslp_c),
                  round(noslp_c.mean(), 0), round(noslp_c.median(), 0),
                  round(noslp_idx.mean(), 0), round(noslp_idx.median(), 0),
                  ""])

for ri, row in enumerate(cost_rows, 2):
    alt = ri % 2 == 0
    for ci, v in enumerate(row, 1):
        style_data(ws7.cell(row=ri, column=ci, value=v), alt)

autofit(ws7)

# Table 8: KM survival / cumulative incidence at each time point
print("Building Table 8 (KM survival) ...")

def km_estimate(kmf, t):
    """Return (point_estimate, ci_lower, ci_upper) at time t."""
    s = float(kmf.predict(t))
    ci = kmf.confidence_interval_survival_function_
    idx = min(ci.index.searchsorted(t, side='right'), len(ci) - 1)
    lo = float(ci.iloc[idx, 0])
    hi = float(ci.iloc[idx, 1])
    return s, lo, hi


km_outcomes = [
    # (label, event_col, dur_col, survival=True → report survival %; False → cum. incidence %)
    ('All-cause mortality',  'died_1825d',     'censor_days',         True),
    ('Recurrent stroke',     'recur_1825d',    'days_to_recur_stroke', False),
    ('Aspiration PNA',       'asp_1825d',      'days_to_aspiration',   False),
    ('Dysphagia Dx',         'dysphagia_1825d','days_to_dysphagia',    False),
    ('G-tube',               'gtube_1825d',    'days_to_gtube',        False),
]

ws8 = wb.create_sheet("Table8_KMSurvival")
km_cols = ["Outcome", "Metric", "Time (days)",
           "SLP %", "SLP CI Lower", "SLP CI Upper",
           "No SLP %", "No SLP CI Lower", "No SLP CI Upper",
           "Log-rank p"]
for ci, h in enumerate(km_cols, 1):
    style_header(ws8.cell(row=1, column=ci, value=h))

ri = 2
for label_km, event_col, dur_col, show_surv in km_outcomes:
    sub = matched_df.copy()
    sub['_dur'] = sub[dur_col].fillna(sub['censor_days']).astype(float).clip(lower=0.5)
    sub['_ev']  = sub[event_col].astype(int)

    slp_s  = sub[sub['slp'] == 1]
    ctrl_s = sub[sub['slp'] == 0]

    kmf_slp  = KaplanMeierFitter()
    kmf_ctrl = KaplanMeierFitter()
    kmf_slp.fit(slp_s['_dur'],  slp_s['_ev'])
    kmf_ctrl.fit(ctrl_s['_dur'], ctrl_s['_ev'])

    lr   = logrank_test(slp_s['_dur'], ctrl_s['_dur'],
                        event_observed_A=slp_s['_ev'],
                        event_observed_B=ctrl_s['_ev'])
    p_lr = lr.p_value
    p_str = "<0.001" if p_lr < 0.001 else f"{p_lr:.4f}"

    metric = "Survival %" if show_surv else "Cum. incidence %"

    for t in TIMEPOINTS:
        s_slp,  lo_slp,  hi_slp  = km_estimate(kmf_slp,  t)
        s_ctrl, lo_ctrl, hi_ctrl = km_estimate(kmf_ctrl, t)

        if show_surv:
            pct_slp,  ci_lo_slp,  ci_hi_slp  = round(100*s_slp,  1), round(100*lo_slp,  1), round(100*hi_slp,  1)
            pct_ctrl, ci_lo_ctrl, ci_hi_ctrl = round(100*s_ctrl, 1), round(100*lo_ctrl, 1), round(100*hi_ctrl, 1)
        else:
            pct_slp,  ci_lo_slp,  ci_hi_slp  = round(100*(1-s_slp),  1), round(100*(1-hi_slp),  1), round(100*(1-lo_slp),  1)
            pct_ctrl, ci_lo_ctrl, ci_hi_ctrl = round(100*(1-s_ctrl), 1), round(100*(1-hi_ctrl), 1), round(100*(1-lo_ctrl), 1)

        alt = ri % 2 == 0
        row_vals = [label_km, metric, t,
                    pct_slp, ci_lo_slp, ci_hi_slp,
                    pct_ctrl, ci_lo_ctrl, ci_hi_ctrl,
                    p_str if t == TIMEPOINTS[0] else ""]
        for ci_col, v in enumerate(row_vals, 1):
            style_data(ws8.cell(row=ri, column=ci_col, value=v), alt)
        ri += 1

autofit(ws8)
ws8.freeze_panes = "A2"

# Save
wb.save(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print("Sheets: Table1_Cohort, Table2_Balance, Table3_All, Table4_ByStrokeType, "
      "Table5_ByAge, Table6_CoxPH, Table7_Costs, Table8_KMSurvival")
