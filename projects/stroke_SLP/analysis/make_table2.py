"""
make_table2.py

Builds the comprehensive current-method Table 2 workbook.

This preserves the useful multi-sheet workbook shape from the old Table2.xlsx,
but updates the definitions to match the active pipeline:
  - Early SLP: days 8-35; Late SLP reference: days 36-90
  - Primary matched cohort: psm_matched_A = TRUE
  - Aspiration-related pneumonia: first pneumonia code J18 or J69
  - PEG/G-tube models exclude index PEG and pre-stroke tube
  - Time-to-event model uses time-varying Cox with person-time split at SLP day
"""

import numpy as np
import openpyxl
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from scipy.stats import chi2_contingency, mannwhitneyu

from stroke_slp_common import OUT_DIR, add_aspiration_related_pna, connect, run_tv_cox


OUT_PATH = OUT_DIR / "Table2.xlsx"
MAX_FOLLOW = 365
TIMEPOINTS = [90, 180, 365]
COMPARISONS = [("A", "Early", "Late", "psm_matched_A")]
TIMING_ORDER = ["Early", "Late"]


HEADER_FILL = PatternFill("solid", fgColor="70071c")
SUBHDR_FILL = PatternFill("solid", fgColor="ba0c2f")
ALT_FILL = PatternFill("solid", fgColor="fdf5f6")
WHITE_BOLD = Font(bold=True, color="FFFFFF")
BOLD = Font(bold=True)
ITALIC = Font(italic=True, color="666666")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
THIN = Side(style="thin", color="BFBFBF")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(cell, sub=False):
    cell.fill = SUBHDR_FILL if sub else HEADER_FILL
    cell.font = WHITE_BOLD
    cell.alignment = CENTER
    cell.border = THIN_BORDER


def style_data(cell, alt=False):
    cell.fill = ALT_FILL if alt else PatternFill()
    cell.alignment = LEFT
    cell.border = THIN_BORDER


def style_note(cell):
    cell.font = ITALIC
    cell.alignment = LEFT


def autofit(ws, min_width=8, max_width=48):
    for col in ws.columns:
        length = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(
            max(length + 2, min_width), max_width
        )


def bucket_drg(drg_cd):
    if pd.isna(drg_cd):
        return "Other"
    try:
        n = int(str(drg_cd).strip())
    except ValueError:
        return "Other"
    if 61 <= n <= 69:
        return "Medical_stroke"
    if 20 <= n <= 38:
        return "Neurosurgical"
    if 52 <= n <= 60:
        return "Spinal"
    if 70 <= n <= 74:
        return "TIA_headache"
    return "Other"


def preprocess_covariates(df):
    df = df.copy()
    for col in ["age_at_adm", "index_los", "van_walraven_score", "adm_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    for col in ["afib", "hypertension", "mech_vent", "prior_stroke", "dual_eligible"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if "drg_cd" in df.columns:
        df["drg_group"] = df["drg_cd"].apply(bucket_drg)
    for col in ["sex", "race", "stroke_type", "drg_group", "adm_source", "rucc_group"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


def load_full(con):
    df = con.execute("""
        SELECT
            p.DSYSRTKY,
            p.slp_timing_group,
            p.psm_matched_A,
            p.days_to_slp_outpt,
            p.age_at_adm,
            p.sex,
            p.race,
            p.stroke_type,
            p.adm_year,
            p.index_los,
            p.mech_vent,
            p.prior_stroke,
            p.van_walraven_score,
            p.afib,
            p.hypertension,
            p.dementia,
            p.dyslipid,
            p.smoking,
            p.dschg_group,
            p.rucc_group,
            p.dual_eligible,
            p.DRG_CD AS drg_cd,
            p.adm_source,
            c.index_pmt
        FROM stroke_propensity p
        JOIN stroke_cohort c ON c.DSYSRTKY = p.DSYSRTKY
        WHERE p.slp_timing_group IN ('Early', 'Late')
    """).df()
    return preprocess_covariates(df)


def load_matched(con, match_col):
    df = con.execute(f"""
        SELECT
            p.DSYSRTKY,
            p.slp_timing_group,
            p.psm_matched_A,
            p.days_to_slp_outpt,
            p.dschg_group,
            p.age_at_adm,
            p.sex,
            p.race,
            p.stroke_type,
            p.adm_year,
            p.index_los,
            p.mech_vent,
            p.prior_stroke,
            p.van_walraven_score,
            p.afib,
            p.hypertension,
            p.dementia,
            p.dyslipid,
            p.smoking,
            p.rucc_group,
            p.dual_eligible,
            p.DRG_CD AS drg_cd,
            p.adm_source,
            p.peg_placed,
            o.days_to_death,
            o.days_to_readmit,
            o.n_readmissions_365d,
            o.days_to_pneumonia,
            o.first_pneumonia_code,
            o.days_to_dysphagia,
            o.days_to_gtube,
            o.pre_stroke_tube,
            o.days_to_snf,
            o.snf_30d,
            o.hha_90d,
            o.total_pmt_365d,
            o.snf_pmt_365d,
            o.hha_pmt_365d,
            c.index_pmt
        FROM stroke_propensity p
        JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
        JOIN stroke_cohort c ON c.DSYSRTKY = p.DSYSRTKY
        WHERE p.{match_col} = TRUE
          AND p.slp_timing_group IN ('Early', 'Late')
    """).df()
    df = preprocess_covariates(df)
    return add_derived(add_aspiration_related_pna(df))


def add_derived(df):
    df = df.copy()
    df["treated"] = (df["slp_timing_group"] == "Early").astype(int)
    df["censor_days"] = np.where(
        df["days_to_death"].notna(),
        df["days_to_death"].clip(upper=MAX_FOLLOW),
        MAX_FOLLOW,
    )
    df["gtube_eligible"] = (
        (df["peg_placed"].fillna(0) == 0)
        & (df["pre_stroke_tube"].fillna(0) == 0)
    )
    for days, label in [(90, "90d"), (180, "180d"), (365, "365d")]:
        df[f"died_{label}"] = (
            df["days_to_death"].notna() & (df["days_to_death"] <= days)
        ).astype(int)
        df[f"readmit_{label}"] = (
            df["days_to_readmit"].notna() & (df["days_to_readmit"] <= days)
        ).astype(int)
        df[f"asp_related_{label}"] = (
            df["days_to_asp_related"].notna() & (df["days_to_asp_related"] <= days)
        ).astype(int)
        df[f"dysphagia_{label}"] = (
            df["days_to_dysphagia"].notna() & (df["days_to_dysphagia"] <= days)
        ).astype(int)
        df[f"gtube_{label}"] = (
            df["days_to_gtube"].notna()
            & (df["days_to_gtube"] <= days)
            & df["gtube_eligible"]
        ).astype(int)
    return df


def write_methods_sheet(wb):
    ws = wb.create_sheet("Methods")
    rows = [
        ("Workbook", "Comprehensive Table 2: current-method Early vs Late SLP results"),
        ("Cohort", "Primary PSM-matched cohort where psm_matched_A = TRUE"),
        ("Exposure", "Early SLP days 8-35 vs Late SLP days 36-90"),
        ("Primary pneumonia outcome", "Aspiration-related pneumonia: first pneumonia code J18 or J69"),
        ("PEG/G-tube", "Excludes index PEG and pre-stroke tube for PEG/G-tube outcome analyses"),
        ("Primary time-to-event model", "Time-varying Cox PH; person-time split at days_to_slp_outpt"),
        ("Landmark OR/KM sheets", "Descriptive/supporting analyses using current outcome definitions"),
        ("Readmission-before-SLP", "Retained in primary analysis; sensitivity-only flag in pipeline"),
    ]
    for ci, h in enumerate(["Item", "Definition"], 1):
        style_header(ws.cell(row=1, column=ci, value=h))
    for ri, row in enumerate(rows, 2):
        for ci, value in enumerate(row, 1):
            style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
    autofit(ws)


def count_pct(sub, col):
    n = len(sub)
    s = pd.to_numeric(sub[col], errors="coerce").fillna(0).sum()
    return f"{int(s):,} ({100 * s / n:.1f}%)" if n else "N/A"


def mean_sd(sub, col):
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    return f"{v.mean():.1f} ({v.std():.1f})" if len(v) else "N/A"


def cat_pct(sub, col, val):
    n = len(sub)
    s = (sub[col] == val).sum()
    return f"{s:,} ({100 * s / n:.1f}%)" if n else "N/A"


def write_cohort_sheet(wb, full_df):
    ws = wb.create_sheet("Table1_Cohort")
    headers = ["Variable"] + TIMING_ORDER
    for ci, h in enumerate(headers, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    subs = {g: full_df[full_df["slp_timing_group"] == g] for g in TIMING_ORDER}
    rows = [
        ("N", lambda s: f"{len(s):,}"),
        ("Age, mean (SD)", lambda s: mean_sd(s, "age_at_adm")),
        ("Female sex, n (%)", lambda s: cat_pct(s, "sex", "Female")),
        ("Male sex, n (%)", lambda s: cat_pct(s, "sex", "Male")),
        ("Race: White, n (%)", lambda s: cat_pct(s, "race", "White")),
        ("Race: Black, n (%)", lambda s: cat_pct(s, "race", "Black")),
        ("Race: Hispanic, n (%)", lambda s: cat_pct(s, "race", "Hispanic")),
        ("Stroke: Ischemic, n (%)", lambda s: cat_pct(s, "stroke_type", "Ischemic")),
        ("Stroke: ICH, n (%)", lambda s: cat_pct(s, "stroke_type", "ICH")),
        ("Stroke: SAH, n (%)", lambda s: cat_pct(s, "stroke_type", "SAH")),
        ("Index LOS, mean (SD)", lambda s: mean_sd(s, "index_los")),
        ("van Walraven score, mean (SD)", lambda s: mean_sd(s, "van_walraven_score")),
        ("Admission year, mean (SD)", lambda s: mean_sd(s, "adm_year")),
        ("Mechanical ventilation, n (%)", lambda s: count_pct(s, "mech_vent")),
        ("Prior stroke, n (%)", lambda s: count_pct(s, "prior_stroke")),
        ("Atrial fibrillation, n (%)", lambda s: count_pct(s, "afib")),
        ("Hypertension, n (%)", lambda s: count_pct(s, "hypertension")),
        ("Dementia, n (%)", lambda s: count_pct(s, "dementia")),
        ("Dyslipidemia, n (%)", lambda s: count_pct(s, "dyslipid")),
        ("Smoking, n (%)", lambda s: count_pct(s, "smoking")),
        ("Discharge home, n (%)", lambda s: cat_pct(s, "dschg_group", "Home")),
        ("Discharge home + HHA, n (%)", lambda s: cat_pct(s, "dschg_group", "Home+HHA")),
        ("Metro county, n (%)", lambda s: cat_pct(s, "rucc_group", "Metro")),
        ("Nonmetro county, n (%)", lambda s: cat_pct(s, "rucc_group", "Nonmetro")),
        ("Rural county, n (%)", lambda s: cat_pct(s, "rucc_group", "Rural")),
        ("Dual eligible, n (%)", lambda s: count_pct(s, "dual_eligible")),
    ]
    for ri, (label, fn) in enumerate(rows, 2):
        values = [label] + [fn(subs[g]) for g in TIMING_ORDER]
        for ci, value in enumerate(values, 1):
            style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
    ws.freeze_panes = "B2"
    autofit(ws)


def smd_values(x, treated):
    x = pd.Series(x).astype(float)
    treated = pd.Series(treated).astype(int)
    x1 = x[treated == 1].dropna()
    x0 = x[treated == 0].dropna()
    if not len(x1) or not len(x0):
        return np.nan
    pooled = np.sqrt((x1.std() ** 2 + x0.std() ** 2) / 2)
    return abs(x1.mean() - x0.mean()) / pooled if pooled > 0 else 0.0


def balance_matrix(df):
    cont = ["age_at_adm", "index_los", "van_walraven_score", "adm_year"]
    binary = ["afib", "hypertension", "mech_vent", "prior_stroke", "dual_eligible"]
    cats = ["sex", "race", "stroke_type", "drg_group", "adm_source", "rucc_group"]
    X = df[cont + binary].astype(float).copy()
    X = pd.concat([X, pd.get_dummies(df[cats], prefix=cats, drop_first=False)], axis=1)
    return X


def pretty_balance_label(col):
    replacements = {
        "age_at_adm": "Age at admission",
        "index_los": "Index LOS",
        "van_walraven_score": "van Walraven score",
        "adm_year": "Admission year",
        "afib": "Atrial fibrillation",
        "hypertension": "Hypertension",
        "mech_vent": "Mechanical ventilation",
        "prior_stroke": "Prior stroke",
        "dual_eligible": "Dual eligible",
    }
    if col in replacements:
        return replacements[col]
    return col.replace("_", ": ", 1).replace("_", " ")


def write_balance_sheet(wb, sheet_name, full_df, matched_df):
    ws = wb.create_sheet(sheet_name)
    headers = ["Variable", "Pre-match SMD", "Post-match SMD", "Balance OK (<0.1)"]
    for ci, h in enumerate(headers, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    pre = full_df[full_df["slp_timing_group"].isin(["Early", "Late"])].copy()
    post = matched_df.copy()
    X_pre = balance_matrix(pre)
    X_post = balance_matrix(post)
    all_cols = list(dict.fromkeys(list(X_pre.columns) + list(X_post.columns)))
    X_pre = X_pre.reindex(columns=all_cols, fill_value=0)
    X_post = X_post.reindex(columns=all_cols, fill_value=0)

    rows = []
    for col in all_cols:
        pre_smd = smd_values(X_pre[col], pre["slp_timing_group"].eq("Early"))
        post_smd = smd_values(X_post[col], post["slp_timing_group"].eq("Early"))
        rows.append((pretty_balance_label(col), pre_smd, post_smd))

    rows.sort(key=lambda row: (0 if row[2] >= 0.1 else 1, -row[2] if pd.notna(row[2]) else 0))
    for ri, (label, pre_smd, post_smd) in enumerate(rows, 2):
        ok = "Yes" if pd.notna(post_smd) and post_smd < 0.1 else "No"
        values = [
            label,
            "" if pd.isna(pre_smd) else round(float(pre_smd), 3),
            "" if pd.isna(post_smd) else round(float(post_smd), 3),
            ok,
        ]
        for ci, value in enumerate(values, 1):
            style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
    ws.freeze_panes = "A2"
    autofit(ws)


def compute_or(sub, ev_col, days_col=None, cutoff=None):
    if cutoff is None:
        elig = sub.copy()
        elig["ev"] = pd.to_numeric(elig[ev_col], errors="coerce").fillna(0).astype(int)
    else:
        ev = pd.to_numeric(sub[ev_col], errors="coerce").fillna(0).astype(bool)
        days = pd.to_numeric(sub[days_col], errors="coerce")
        mask = (sub["censor_days"] >= cutoff) | (ev & (days <= cutoff))
        elig = sub[mask].copy()
        elig_ev = pd.to_numeric(elig[ev_col], errors="coerce").fillna(0).astype(bool)
        elig_days = pd.to_numeric(elig[days_col], errors="coerce")
        elig["ev"] = (elig_ev & (elig_days <= cutoff)).astype(int)

    treated = elig[elig["treated"] == 1]
    control = elig[elig["treated"] == 0]
    nt, nc = len(treated), len(control)
    et, ec = int(treated["ev"].sum()), int(control["ev"].sum())
    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return {
            "n_treat": nt, "ev_treat": et, "pct_treat": "N/A",
            "n_ctrl": nc, "ev_ctrl": ec, "pct_ctrl": "N/A",
            "OR": "N/A", "CI": "N/A", "p": "N/A",
        }
    or_val = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = np.exp(np.log(or_val) - 1.96 * se), np.exp(np.log(or_val) + 1.96 * se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return {
        "n_treat": nt,
        "ev_treat": et,
        "pct_treat": f"{100 * et / nt:.1f}%",
        "n_ctrl": nc,
        "ev_ctrl": ec,
        "pct_ctrl": f"{100 * ec / nc:.1f}%",
        "OR": f"{or_val:.2f}",
        "CI": f"[{lo:.2f}, {hi:.2f}]",
        "p": "<0.001" if p < 0.001 else f"{p:.3f}",
    }


def landmark_outcomes():
    return [
        ("Mortality", "died_{tp}", "days_to_death", TIMEPOINTS, None),
        ("Readmission", "readmit_{tp}", "days_to_readmit", TIMEPOINTS, None),
        ("Aspiration-related PNA", "asp_related_{tp}", "days_to_asp_related", TIMEPOINTS, None),
        ("Dysphagia Dx", "dysphagia_{tp}", "days_to_dysphagia", TIMEPOINTS, None),
        ("PEG/G-tube", "gtube_{tp}", "days_to_gtube", TIMEPOINTS, "gtube_eligible"),
        ("SNF 30d", "snf_30d", None, [None], None),
        ("HHA 90d", "hha_90d", None, [None], None),
    ]


def build_or_rows(sub, strat_label):
    rows = []
    for label, ev_template, days_col, cutoffs, filter_col in landmark_outcomes():
        data = sub[sub[filter_col]].copy() if filter_col else sub.copy()
        for cutoff in cutoffs:
            if "{tp}" in ev_template:
                tp = f"{cutoff}d"
                ev_col = ev_template.replace("{tp}", tp)
                display = f"{label} {tp}"
            else:
                ev_col = ev_template
                display = label
            rows.append({"Stratum": strat_label, "Outcome": display, **compute_or(data, ev_col, days_col, cutoff)})
    return rows


def write_or_sheet(wb, sheet_name, rows, treat_grp, ctrl_grp):
    ws = wb.create_sheet(sheet_name)
    col_display = [
        "Stratum", "Outcome",
        f"n ({treat_grp})", f"Events ({treat_grp})", f"% ({treat_grp})",
        f"n ({ctrl_grp})", f"Events ({ctrl_grp})", f"% ({ctrl_grp})",
        "OR", "95% CI", "p",
    ]
    col_keys = [
        "Stratum", "Outcome", "n_treat", "ev_treat", "pct_treat",
        "n_ctrl", "ev_ctrl", "pct_ctrl", "OR", "CI", "p",
    ]
    for ci, h in enumerate(col_display, 1):
        style_header(ws.cell(row=1, column=ci, value=h))
    for ri, row in enumerate(rows, 2):
        for ci, key in enumerate(col_keys, 1):
            style_data(ws.cell(row=ri, column=ci, value=row.get(key, "")), ri % 2 == 0)
    ws.freeze_panes = "A2"
    autofit(ws)


def write_tv_cox_sheet(wb, sheet_name, matched_df):
    ws = wb.create_sheet(sheet_name)
    headers = ["Outcome", "N", "Events", "Event %", "HR", "95% CI Lower", "95% CI Upper", "p", "Note"]
    for ci, h in enumerate(headers, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    outcomes = [
        ("Aspiration-related PNA", "days_to_asp_related", "days_to_death", None, "J18/J69; death as competing event"),
        ("PEG/G-tube", "days_to_gtube", "days_to_death", "gtube_eligible", "Excludes index PEG and pre-stroke tube; death as competing event"),
        ("Dysphagia Dx", "days_to_dysphagia", "days_to_death", None, "Secondary descriptive TV Cox; death as competing event"),
        ("Mortality", "days_to_death", None, None, "All-cause mortality"),
    ]
    for ri, (label, event_col, comp_col, filter_col, note) in enumerate(outcomes, 2):
        sub = matched_df[matched_df[filter_col]].copy() if filter_col else matched_df.copy()
        result, n, events = run_tv_cox(sub, event_col, comp_col, "Early")
        event_pct = round(100 * events / n, 1) if n else 0
        if result:
            hr, lo, hi, p = result
            values = [
                label, n, events, f"{event_pct:.1f}%",
                round(float(hr), 2), round(float(lo), 2), round(float(hi), 2),
                "<0.0001" if p < 0.0001 else round(float(p), 4),
                note,
            ]
        else:
            values = [label, n, events, f"{event_pct:.1f}%", "N/A", "", "", "N/A", note]
        for ci, value in enumerate(values, 1):
            style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
    ws.freeze_panes = "A2"
    autofit(ws)


def write_cost_sheet(wb, sheet_name, matched_df, treat_grp, ctrl_grp):
    ws = wb.create_sheet(sheet_name)
    headers = ["Group", "N", "Mean 1yr Cost ($)", "Median 1yr Cost ($)", "Mean Index Cost ($)", "Median Index Cost ($)", "Mann-Whitney p"]
    for ci, h in enumerate(headers, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    groups = [(treat_grp, matched_df[matched_df["treated"] == 1]), (ctrl_grp, matched_df[matched_df["treated"] == 0])]
    c1 = groups[0][1]["total_pmt_365d"].dropna()
    c0 = groups[1][1]["total_pmt_365d"].dropna()
    if len(c1) >= 10 and len(c0) >= 10:
        _, mw_p = mannwhitneyu(c1, c0, alternative="two-sided")
        p_str = "<0.001" if mw_p < 0.001 else f"{mw_p:.3f}"
    else:
        p_str = "N/A"

    for ri, (label, sub) in enumerate(groups, 2):
        costs = sub["total_pmt_365d"].dropna()
        index_costs = sub["index_pmt"].dropna()
        values = [
            label,
            len(costs),
            round(costs.mean(), 0) if len(costs) else "N/A",
            round(costs.median(), 0) if len(costs) else "N/A",
            round(index_costs.mean(), 0) if len(index_costs) else "N/A",
            round(index_costs.median(), 0) if len(index_costs) else "N/A",
            p_str if ri == 2 else "",
        ]
        for ci, value in enumerate(values, 1):
            style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
    autofit(ws)


def km_at(kmf, t):
    survival = float(kmf.predict(t))
    ci = kmf.confidence_interval_survival_function_
    idx = max(min(ci.index.searchsorted(t, side="right") - 1, len(ci) - 1), 0)
    return survival, float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])


def write_km_sheet(wb, sheet_name, matched_df, treat_grp, ctrl_grp):
    ws = wb.create_sheet(sheet_name)
    headers = [
        "Outcome", "Metric", "Time (days)",
        f"% ({treat_grp})", f"CI Lo ({treat_grp})", f"CI Hi ({treat_grp})",
        f"% ({ctrl_grp})", f"CI Lo ({ctrl_grp})", f"CI Hi ({ctrl_grp})",
        "Log-rank p",
    ]
    for ci, h in enumerate(headers, 1):
        style_header(ws.cell(row=1, column=ci, value=h))

    outcomes = [
        ("All-cause mortality", "died_365d", "days_to_death", True, None),
        ("Aspiration-related PNA", "asp_related_365d", "days_to_asp_related", False, None),
        ("Dysphagia Dx", "dysphagia_365d", "days_to_dysphagia", False, None),
        ("PEG/G-tube", "gtube_365d", "days_to_gtube", False, "gtube_eligible"),
    ]
    ri = 2
    for label, ev_col, duration_col, show_survival, filter_col in outcomes:
        sub = matched_df[matched_df[filter_col]].copy() if filter_col else matched_df.copy()
        sub["_duration"] = sub[duration_col].fillna(sub["censor_days"]).astype(float).clip(lower=0.5)
        sub["_event"] = sub[ev_col].astype(int)
        treated = sub[sub["treated"] == 1]
        control = sub[sub["treated"] == 0]
        km_t = KaplanMeierFitter()
        km_c = KaplanMeierFitter()
        km_t.fit(treated["_duration"], treated["_event"])
        km_c.fit(control["_duration"], control["_event"])
        lr = logrank_test(
            treated["_duration"], control["_duration"],
            event_observed_A=treated["_event"], event_observed_B=control["_event"],
        )
        p_str = "<0.001" if lr.p_value < 0.001 else f"{lr.p_value:.4f}"
        metric = "Survival %" if show_survival else "Cumulative incidence %"
        for t in TIMEPOINTS:
            s_t, lo_t, hi_t = km_at(km_t, t)
            s_c, lo_c, hi_c = km_at(km_c, t)
            if show_survival:
                vals_t = (round(100 * s_t, 1), round(100 * lo_t, 1), round(100 * hi_t, 1))
                vals_c = (round(100 * s_c, 1), round(100 * lo_c, 1), round(100 * hi_c, 1))
            else:
                vals_t = (round(100 * (1 - s_t), 1), round(100 * (1 - hi_t), 1), round(100 * (1 - lo_t), 1))
                vals_c = (round(100 * (1 - s_c), 1), round(100 * (1 - hi_c), 1), round(100 * (1 - lo_c), 1))
            values = [label, metric, t, *vals_t, *vals_c, p_str if t == TIMEPOINTS[0] else ""]
            for ci, value in enumerate(values, 1):
                style_data(ws.cell(row=ri, column=ci, value=value), ri % 2 == 0)
            ri += 1
    ws.freeze_panes = "A2"
    autofit(ws)


def main():
    print("Loading current stroke SLP data...")
    con = connect(read_only=True)
    full_df = load_full(con)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_methods_sheet(wb)
    write_cohort_sheet(wb, full_df)

    for comp_label, treat_grp, ctrl_grp, match_col in COMPARISONS:
        print(f"Building comprehensive Table 2 ({comp_label}: {treat_grp} vs {ctrl_grp})...")
        matched_df = load_matched(con, match_col)
        print(f"  Matched rows: {len(matched_df):,}")
        write_balance_sheet(wb, f"Table2_Balance_{comp_label}", full_df, matched_df)
        write_or_sheet(wb, f"Table3_OR_{comp_label}", build_or_rows(matched_df, "All matched"), treat_grp, ctrl_grp)

        rows_stroke = []
        for label in ["Ischemic", "ICH", "SAH"]:
            sub = matched_df[matched_df["stroke_type"] == label]
            if len(sub) >= 20:
                rows_stroke.extend(build_or_rows(sub, label))
        write_or_sheet(wb, f"Table4_ByStrokeType_{comp_label}", rows_stroke, treat_grp, ctrl_grp)

        age_rows = []
        for label, sub in [
            ("Age <75", matched_df[matched_df["age_at_adm"] < 75]),
            ("Age 75-84", matched_df[(matched_df["age_at_adm"] >= 75) & (matched_df["age_at_adm"] < 85)]),
            ("Age 85+", matched_df[matched_df["age_at_adm"] >= 85]),
        ]:
            age_rows.extend(build_or_rows(sub, label))
        write_or_sheet(wb, f"Table5_ByAge_{comp_label}", age_rows, treat_grp, ctrl_grp)

        q1 = matched_df["van_walraven_score"].quantile(1 / 3)
        q2 = matched_df["van_walraven_score"].quantile(2 / 3)
        vw_rows = []
        for label, sub in [
            (f"Low (VWS <= {q1:.0f})", matched_df[matched_df["van_walraven_score"] <= q1]),
            (
                f"Mid (VWS {q1:.0f}-{q2:.0f})",
                matched_df[
                    (matched_df["van_walraven_score"] > q1)
                    & (matched_df["van_walraven_score"] <= q2)
                ],
            ),
            (f"High (VWS > {q2:.0f})", matched_df[matched_df["van_walraven_score"] > q2]),
        ]:
            vw_rows.extend(build_or_rows(sub, label))
        write_or_sheet(wb, f"Table5b_ByVW_{comp_label}", vw_rows, treat_grp, ctrl_grp)

        write_tv_cox_sheet(wb, f"Table6_TV_Cox_{comp_label}", matched_df)
        write_cost_sheet(wb, f"Table7_Costs_{comp_label}", matched_df, treat_grp, ctrl_grp)
        write_km_sheet(wb, f"Table8_KM_{comp_label}", matched_df, treat_grp, ctrl_grp)

    con.close()
    wb.save(str(OUT_PATH))
    print(f"Saved: {OUT_PATH}")
    print("Sheets: " + ", ".join(ws.title for ws in wb.worksheets))


if __name__ == "__main__":
    main()
