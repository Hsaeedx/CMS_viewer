"""
make_outcomes_table.py
Outcomes Excel workbook for PowerPoint presentation.

One sheet per comparison (A, B, C). Each sheet has three sections:
  Section 1  — Multi-interval cumulative incidence: G-tube + SLP
               Rows = intervals (14d, 30d, 90d, 180d, 1-yr, 3-yr)
               Cols = N/% per arm (uncond + at-risk-cond) + OR (95% CI) + p
  Section 2  — Binary outcomes OR matrix (Dysphagia)
               Rows = outcome × stratum (5 strata)
               Cols = 6-mo / 1-yr / 3-yr / 5-yr / Anytime
  Section 3  — Baseline descriptors (dx->tx delay, pre-tx event %)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
DB_PATH  = os.getenv("duckdb_database",    r"F:\CMS\cms_data.duckdb")
OUT_PATH = str(Path(os.getenv("analysis_directory", r"C:\Users\hsaee\Desktop\CMS_viewer")) / "projects" / "opscc" / "figures" / "outcomes_tables.xlsx")

COMPARISONS = [
    ('A', 'psm_matched_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'TORS + RT',  'CRT'),
    ('C', 'psm_matched_C', 'TORS + CRT', 'CRT'),
]

GTUBE_INTERVALS = [
    ('14d',   14,   'placement_by_14d'),
    ('30d',   30,   'placement_by_30d'),
    ('90d',   90,   'placement_by_90d'),
    ('180d',  180,  'placement_by_180d'),
    ('1-yr',  365,  'placement_by_365d'),
    ('3-yr',  1095, 'placement_by_1095d'),
]
SLP_INTERVALS = [
    ('14d',   14,   'event_by_14d'),
    ('30d',   30,   'event_by_30d'),
    ('90d',   90,   'event_by_90d'),
    ('180d',  180,  'event_by_180d'),
    ('1-yr',  365,  'event_by_365d'),
    ('3-yr',  1095, 'event_by_1095d'),
]

OUTCOMES_BINARY = [
    ('Dysphagia', 'has_dysphagia', 'days_dys'),
]

TIMEPOINTS = [
    ('6-mo',    182),
    ('1-yr',    365),
    ('3-yr',   1095),
    ('5-yr',   1825),
    ('Anytime', None),
]

STRATA_LABELS = [
    'All matched',
    'Age < 75',
    'Age ≥ 75',
    'Low comorbidity (VW≤0)',
    'High comorbidity (VW >0)',
]


def or_p(et, nt, ec, nc):
    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return np.nan, np.nan, np.nan, np.nan
    or_v   = (a*d) / (b*c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return or_v, lo, hi, p


def irr_p(et, pdt, ec, pdc):
    """Poisson-based incidence rate ratio + 95% CI + Wald p-value."""
    from scipy.stats import norm
    if et == 0 or ec == 0 or pdt == 0 or pdc == 0:
        return np.nan, np.nan, np.nan, np.nan
    irr   = (et / pdt) / (ec / pdc)
    se    = np.sqrt(1/et + 1/ec)
    lo, hi = np.exp(np.log(irr) - 1.96*se), np.exp(np.log(irr) + 1.96*se)
    z     = np.log(irr) / se
    p     = 2 * (1 - norm.cdf(abs(z)))
    return irr, lo, hi, p


def compute_or(sub, has_col, days_col, cutoff):
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
    t_ = elig[elig['tors'] == 1]
    c_ = elig[elig['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    return or_p(et, nt, ec, nc)


# Delayed toxicity after treatment completion (>=90d washout), anchored on
# last RT/chemo date. Marker for delayed swallowing failure / G-tube dependence.
DELAYED_LANDMARKS = [('6-mo', 180), ('1-yr', 365), ('2-yr', 730), ('3-yr', 1095)]
DELAYED_OUTCOMES = [
    ('G-tube placement', 'g_delayed_by_',   'g_through_completion90'),
    ('SLP visit',        'slp_delayed_by_', 'slp_through_completion90'),
    ('Dysphagia',        'dys_delayed_by_', 'dys_through_completion90'),
]


def delayed_or(sub, flag_prefix, through_col, L, incident_only):
    """At-risk delayed-event cumulative incidence by L days after completion.
    Eligible = followed past L after completion OR had the event by L.
    Incident view restricts to patients reaching completion+90d event-free.
    Returns (nt, et, nc, ec, or_v, lo, hi, p)."""
    flag_col = f"{flag_prefix}{L}d"
    d = sub[sub[flag_col].notna()].copy()
    if incident_only:
        d = d[d[through_col].fillna(True) == False]
    flag = d[flag_col].astype(bool)
    elig = (d['foll_after_comp'] >= L) | flag
    e = d[elig]
    t_, c_ = e[e['tors'] == 1], e[e['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et = int(t_[flag_col].astype(bool).sum())
    ec = int(c_[flag_col].astype(bool).sum())
    or_v, lo, hi, p = or_p(et, nt, ec, nc)
    return nt, et, nc, ec, or_v, lo, hi, p


def side(style='thin'):  return Side(style=style)
def border_all(s='thin'): return Border(left=side(s), right=side(s), top=side(s), bottom=side(s))
def fill(hex_color):      return PatternFill(fill_type='solid', fgColor=hex_color)
def font(bold=False, size=10, color='000000', name='Calibri'):
    return Font(bold=bold, size=size, color=color, name=name)
def align(h='center', v='center', wrap=True):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

HDR_FILL   = fill('BA0C2F')   # OSU Scarlet
OUT_FILL   = fill('70071C')   # Scarlet Dark
ALT_FILL   = fill('F2F2F2')
SIG_FILL   = fill('E2EFDA')
WHITE_FILL = fill('FFFFFF')

N_TP = len(TIMEPOINTS)

con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

wb = openpyxl.Workbook()
wb.remove(wb.active)

for comp, match_col, tors_label, ctrl_label in COMPARISONS:

    print(f"Loading data for Comparison {comp}: {tors_label} vs {ctrl_label}...")

    df = con.execute(f"""
        SELECT
            s.DSYSRTKY, s.tx_group, s.first_tx_date,
            s.age_at_dx, s.van_walraven_score,
            f.ffs_censor_date,
            DATEDIFF('day', s.first_tx_date, f.ffs_censor_date) AS follow_up_days,
            o.has_dysphagia,
            DATEDIFF('day', s.first_tx_date, o.first_dysphagia_date) AS days_dys,
            DATEDIFF('day', s.first_tx_date, co.first_chemo_date)    AS days_tx_to_chemo,
            p.psm_match_id_C    AS pair_id_C,
            g.first_post_placement_day AS g_first_post_placement_day,
            slp.first_post_day_from_tx AS slp_first_post_day_from_tx,
            g.days_dx_to_tx           AS g_days_dx_to_tx,
            g.pre_placement_any       AS g_pre_placement_any,
            g.placement_by_14d        AS g_placement_by_14d,
            g.placement_by_30d        AS g_placement_by_30d,
            g.placement_by_90d        AS g_placement_by_90d,
            g.placement_by_180d       AS g_placement_by_180d,
            g.placement_by_365d       AS g_placement_by_365d,
            g.placement_by_1095d      AS g_placement_by_1095d,
            slp.pre_event_any      AS slp_pre_event_any,
            slp.event_by_14d       AS slp_event_by_14d,
            slp.event_by_30d       AS slp_event_by_30d,
            slp.event_by_90d       AS slp_event_by_90d,
            slp.event_by_180d      AS slp_event_by_180d,
            slp.event_by_365d      AS slp_event_by_365d,
            slp.event_by_1095d     AS slp_event_by_1095d,
            -- Delayed toxicity after treatment completion (90d washout)
            g.days_tx_to_completion           AS days_tx_to_completion,
            g.placement_through_completion90  AS g_through_completion90,
            g.delayed_placement_by_180d       AS g_delayed_by_180d,
            g.delayed_placement_by_365d       AS g_delayed_by_365d,
            g.delayed_placement_by_730d       AS g_delayed_by_730d,
            g.delayed_placement_by_1095d      AS g_delayed_by_1095d,
            slp.slp_through_completion90      AS slp_through_completion90,
            slp.delayed_slp_by_180d           AS slp_delayed_by_180d,
            slp.delayed_slp_by_365d           AS slp_delayed_by_365d,
            slp.delayed_slp_by_730d           AS slp_delayed_by_730d,
            slp.delayed_slp_by_1095d          AS slp_delayed_by_1095d,
            o.dysphagia_through_completion90  AS dys_through_completion90,
            o.delayed_dysphagia_by_180d       AS dys_delayed_by_180d,
            o.delayed_dysphagia_by_365d       AS dys_delayed_by_365d,
            o.delayed_dysphagia_by_730d       AS dys_delayed_by_730d,
            o.delayed_dysphagia_by_1095d      AS dys_delayed_by_1095d
        FROM opscc_survival s
        JOIN opscc_propensity p USING (DSYSRTKY)
        JOIN opscc_ffs_dates  f USING (DSYSRTKY)
        JOIN opscc_outcomes   o USING (DSYSRTKY)
        LEFT JOIN opscc_cohort co  ON co.DSYSRTKY = s.DSYSRTKY
        LEFT JOIN opscc_slp   slp USING (DSYSRTKY)
        LEFT JOIN opscc_gtube_dependence g USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
    """).df()

    if len(df) == 0:
        print(f"  No matched patients for Comparison {comp}. Skipping.")
        continue

    df['tors']     = (df['tx_group'] == tors_label).astype(int)
    df['elix_grp'] = (df['van_walraven_score'] > 0).map({False: 'Low', True: 'High'})

    # Follow-up from treatment completion (delayed-toxicity at-risk sets).
    # Computed from the original first_tx-anchored follow_up_days BEFORE the
    # Comp C chemo re-anchoring below mutates follow_up_days.
    df['foll_after_comp'] = df['follow_up_days'] - df['days_tx_to_completion']

    # ── Comp C only: build (1) TORS+CRT post-op descriptive (no comparison)
    #                       (2) Post-chemo / Overall head-to-head comparison
    # Computed BEFORE the chemo re-anchoring below so we have the original
    # event-day columns (days_dys, g_first_post_placement_day, etc.) intact.
    post_op_descriptive = None  # list of (outcome, n_events, n_total, pct_str, rate_per_1k_pdays)
    comparison_rows = None      # list of (outcome, view, T_str, C_str, OR_stats)
    if comp == 'C':
        T = df[df['tors'] == 1].copy()
        C = df[df['tors'] == 0].copy()
        T_valid = T[T['days_tx_to_chemo'].fillna(-1) >= 0]
        pdt = T_valid['days_tx_to_chemo'].clip(lower=0).sum()

        post_op_descriptive = []
        for label, col_name in [
            ('G-tube',    'g_first_post_placement_day'),
            ('Dysphagia', 'days_dys'),
            ('SLP',       'slp_first_post_day_from_tx'),
        ]:
            et = int((T_valid[col_name].notna() &
                      (T_valid[col_name] < T_valid['days_tx_to_chemo'])).sum())
            rate = 1000 * et / pdt if pdt else float('nan')
            post_op_descriptive.append((label,
                                        f"{et}/{len(T_valid)} ({100*et/len(T_valid):.1f}%)",
                                        f"{rate:.2f}"))

        comparison_rows = []
        for label, col_name in [
            ('G-tube',    'g_first_post_placement_day'),
            ('Dysphagia', 'days_dys'),
            ('SLP',       'slp_first_post_day_from_tx'),
        ]:
            # Post-chemo (chemo-anchored 3yr)
            d_T = T['days_tx_to_chemo'].fillna(0)
            d_C = C['days_tx_to_chemo'].fillna(0)
            et_post = int((T[col_name].notna() &
                           ((T[col_name] - d_T) >= 0) &
                           ((T[col_name] - d_T) <= 1095)).sum())
            ec_post = int((C[col_name].notna() &
                           ((C[col_name] - d_C) >= 0) &
                           ((C[col_name] - d_C) <= 1095)).sum())
            or_post = or_p(et_post, len(T), ec_post, len(C))

            # Overall (first_tx-anchored 3yr)
            et_all = int((T[col_name].notna() & T[col_name].between(0, 1095)).sum())
            ec_all = int((C[col_name].notna() & C[col_name].between(0, 1095)).sum())
            or_all = or_p(et_all, len(T), ec_all, len(C))

            comparison_rows.append((label, 'Post-chemo (3-yr from chemo)',
                                    f"{et_post}/{len(T)} ({100*et_post/len(T):.1f}%)",
                                    f"{ec_post}/{len(C)} ({100*ec_post/len(C):.1f}%)",
                                    or_post))
            comparison_rows.append((label, 'Overall (3-yr from first tx)',
                                    f"{et_all}/{len(T)} ({100*et_all/len(T):.1f}%)",
                                    f"{ec_all}/{len(C)} ({100*ec_all/len(C):.1f}%)",
                                    or_all))

        # Save total post-op window stats for the descriptive header
        df.attrs['postop_window'] = {
            'n': len(T_valid),
            'mean': float(T_valid['days_tx_to_chemo'].mean()) if len(T_valid) else 0.0,
            'sd':   float(T_valid['days_tx_to_chemo'].std()) if len(T_valid) else 0.0,
            'median': float(T_valid['days_tx_to_chemo'].median()) if len(T_valid) else 0.0,
            'p25': float(T_valid['days_tx_to_chemo'].quantile(0.25)) if len(T_valid) else 0.0,
            'p75': float(T_valid['days_tx_to_chemo'].quantile(0.75)) if len(T_valid) else 0.0,
            'max': float(T_valid['days_tx_to_chemo'].max()) if len(T_valid) else 0.0,
            'pdays': int(pdt),
        }

    # ── Comp C only: re-anchor flags + days_dys + follow_up on first_chemo_date
    # so the table matches the chemo-aligned figures.
    if comp == 'C':
        dchemo = df['days_tx_to_chemo'].fillna(0)
        df['follow_up_days'] = df['follow_up_days'] - dchemo

        # Re-anchor dysphagia: events before chemo are excluded from the
        # chemo-anchored cumulative incidence (they live in the pre-CRT
        # surgical era).
        df['days_dys'] = df['days_dys'] - dchemo
        pre_dys = df['days_dys'] < 0
        df.loc[pre_dys, 'has_dysphagia'] = False
        df.loc[pre_dys, 'days_dys'] = np.nan

        # Re-build G-tube placement_by_T flags from chemo start
        gd = df['g_first_post_placement_day'] - dchemo
        for T, lbl in [(14, '14d'), (30, '30d'), (90, '90d'),
                       (180, '180d'), (365, '365d'), (1095, '1095d')]:
            df[f'g_placement_by_{lbl}'] = (gd.notna()) & (gd >= 0) & (gd <= T)

        # Re-build SLP event_by_T flags from chemo start
        sd = df['slp_first_post_day_from_tx'] - dchemo
        for T, lbl in [(14, '14d'), (30, '30d'), (90, '90d'),
                       (180, '180d'), (365, '365d'), (1095, '1095d')]:
            df[f'slp_event_by_{lbl}'] = (sd.notna()) & (sd >= 0) & (sd <= T)

    strata_dfs = [
        df,
        df[df['age_at_dx'] < 75],
        df[df['age_at_dx'] >= 75],
        df[df['elix_grp'] == 'Low'],
        df[df['elix_grp'] == 'High'],
    ]

    ws = wb.create_sheet(title=f"Comparison {comp}")

    # ── Title ─────────────────────────────────────────────────────────────────
    anchor_note = "  —  anchored on chemoradiation start" if comp == 'C' else ""
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=10)
    c = ws.cell(row=1, column=1,
                value=f"Comparison {comp}: {tors_label} vs {ctrl_label}"
                      f"  —  OR < 1 favors {tors_label}  —  Green = p < 0.05"
                      f"{anchor_note}")
    c.font = font(bold=True, size=12); c.alignment = align(h='left', wrap=False)
    ws.row_dimensions[1].height = 20
    row = 2

    # ── Section 1: Baseline descriptors ──────────────────────────────────────
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
    c = ws.cell(row=row, column=1, value="Baseline descriptors")
    c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
    c.alignment = align(h='left', wrap=False); c.border = border_all()
    row += 1

    hdrs = ['Arm', 'N', 'dx→tx median (IQR)',
            'G-tube placement pre-tx %', 'SLP pre-tx %']
    for i, h in enumerate(hdrs):
        c = ws.cell(row=row, column=1+i, value=h)
        c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
        c.alignment = align(); c.border = border_all()
    row += 1
    for arm, label in [(1, tors_label), (0, ctrl_label)]:
        s = df[df['tors'] == arm]
        dx_med = s['g_days_dx_to_tx'].median()
        dx_p25 = s['g_days_dx_to_tx'].quantile(0.25)
        dx_p75 = s['g_days_dx_to_tx'].quantile(0.75)
        g_pre  = int(s['g_pre_placement_any'].fillna(False).sum())
        slp_pre = int(s['slp_pre_event_any'].fillna(False).sum())
        values = [label, len(s), f"{dx_med:.0f} ({dx_p25:.0f}–{dx_p75:.0f})",
                  f"{g_pre} ({100*g_pre/len(s):.1f}%)",
                  f"{slp_pre} ({100*slp_pre/len(s):.1f}%)"]
        for i, v in enumerate(values):
            c = ws.cell(row=row, column=1+i, value=v)
            c.font = font(size=10); c.alignment = align(); c.border = border_all()
        row += 1
    row += 1

    # ── Section 2: G-tube + SLP cumulative incidence ─────────────────────────
    # G-tube uses placement-only flags (Z93.1 excluded); SLP uses any-event flags
    for outcome_label, prefix, intervals in [
        ('G-tube placement cumulative incidence', 'g_',   GTUBE_INTERVALS),
        ('SLP cumulative incidence',              'slp_', SLP_INTERVALS),
    ]:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
        c = ws.cell(row=row, column=1, value=outcome_label)
        c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
        c.alignment = align(h='left', wrap=False); c.border = border_all()
        row += 1

        hdrs = ['Interval',
                f'{tors_label} N/% (uncond)', f'{tors_label} % (at-risk)',
                f'{ctrl_label} N/% (uncond)', f'{ctrl_label} % (at-risk)',
                'OR (95% CI)', 'p']
        for i, h in enumerate(hdrs):
            c = ws.cell(row=row, column=1+i, value=h)
            c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
            c.alignment = align(); c.border = border_all()
        row += 1

        for int_lbl, cutoff_d, flag_suffix in intervals:
            flag_col = prefix + flag_suffix
            t_ = df[df['tors'] == 1]; c_ = df[df['tors'] == 0]
            nt, nc = len(t_), len(c_)
            et = int(t_[flag_col].fillna(False).sum())
            ec = int(c_[flag_col].fillna(False).sum())
            pt_u = f"{et}/{nt} ({100*et/nt:.1f}%)" if nt else "N/A"
            pc_u = f"{ec}/{nc} ({100*ec/nc:.1f}%)" if nc else "N/A"

            t_cond = t_[t_['follow_up_days'] >= cutoff_d]
            c_cond = c_[c_['follow_up_days'] >= cutoff_d]
            et_c = int(t_cond[flag_col].fillna(False).sum())
            ec_c = int(c_cond[flag_col].fillna(False).sum())
            pt_c = f"{100*et_c/len(t_cond):.1f}%" if len(t_cond) else "N/A"
            pc_c = f"{100*ec_c/len(c_cond):.1f}%" if len(c_cond) else "N/A"

            or_v, lo, hi, p = or_p(et, nt, ec, nc)
            sig = (not np.isnan(or_v)) and (p < 0.05)
            if np.isnan(or_v):
                or_str, p_str = 'N/A', 'N/A'
            else:
                or_str = f"{or_v:.2f} ({lo:.2f}–{hi:.2f})"
                p_str  = "<0.0001" if p < 0.0001 else f"{p:.4f}"

            row_fill = SIG_FILL if sig else WHITE_FILL
            vals = [int_lbl, pt_u, pt_c, pc_u, pc_c, or_str, p_str]
            for i, v in enumerate(vals):
                c = ws.cell(row=row, column=1+i, value=v)
                c.font = font(bold=sig, size=10); c.fill = row_fill
                c.alignment = align(); c.border = border_all()
            row += 1
        row += 1

    # ── Section 3: Binary outcomes (Dysphagia) ───────────────────────────────
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=1+N_TP)
    c = ws.cell(row=row, column=1, value="Binary outcomes — OR (95% CI) by stratum")
    c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
    c.alignment = align(h='left', wrap=False); c.border = border_all()
    row += 1

    c = ws.cell(row=row, column=1, value='Subgroup')
    c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
    c.alignment = align(); c.border = border_all()
    for tp_idx, (tp_lbl, _) in enumerate(TIMEPOINTS):
        c = ws.cell(row=row, column=2+tp_idx, value=tp_lbl)
        c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
        c.alignment = align(); c.border = border_all()
    row += 1

    for out_lbl, has_col, days_col in OUTCOMES_BINARY:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=1+N_TP)
        c = ws.cell(row=row, column=1, value=out_lbl)
        c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = OUT_FILL
        c.alignment = align(h='left', wrap=False); c.border = border_all()
        row += 1
        for s_idx, (s_lbl, sub) in enumerate(zip(STRATA_LABELS, strata_dfs)):
            row_fill = ALT_FILL if s_idx % 2 == 1 else WHITE_FILL
            c = ws.cell(row=row, column=1, value=f"  {s_lbl}")
            c.font = font(bold=(s_idx == 0), size=10); c.fill = row_fill
            c.alignment = align(h='left', wrap=False); c.border = border_all()
            for tp_idx, (_, cutoff) in enumerate(TIMEPOINTS):
                or_v, lo, hi, p = compute_or(sub, has_col, days_col, cutoff)
                sig = (not np.isnan(or_v)) and (p < 0.05)
                if np.isnan(or_v):
                    cell_val = 'N/A'
                else:
                    star = '*' if sig else ''
                    cell_val = f"{or_v:.2f} ({lo:.2f}–{hi:.2f}){star}"
                c = ws.cell(row=row, column=2+tp_idx, value=cell_val)
                c.font = font(bold=sig, size=10)
                c.fill = SIG_FILL if sig else row_fill
                c.alignment = align(); c.border = border_all()
            row += 1
        row += 1

    # ── Section 4a (Comp C only): TORS+CRT post-op descriptive (no comparator)
    if post_op_descriptive:
        window = df.attrs.get('postop_window', {})
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        c = ws.cell(row=row, column=1,
                    value=f"Comp C — TORS+CRT post-surgical phase (descriptive only, no CRT comparator). "
                          f"N={window.get('n', 0)}; "
                          f"TORS-to-chemo days: mean {window.get('mean', 0):.1f} "
                          f"(SD {window.get('sd', 0):.1f}), "
                          f"median {window.get('median', 0):.0f} "
                          f"(IQR {window.get('p25', 0):.0f}–{window.get('p75', 0):.0f}), "
                          f"range 0–{window.get('max', 0):.0f}. "
                          f"Total {window.get('pdays', 0):,} person-days.")
        c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
        c.alignment = align(h='left', wrap=False); c.border = border_all()
        row += 1
        hdrs = ['Outcome', 'Events (N events / N patients)', 'Rate per 1000 person-days']
        for i, h in enumerate(hdrs):
            c = ws.cell(row=row, column=1+i, value=h)
            c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
            c.alignment = align(); c.border = border_all()
        row += 1
        for outcome, events_str, rate_str in post_op_descriptive:
            vals = [outcome, events_str, rate_str]
            for i, v in enumerate(vals):
                c = ws.cell(row=row, column=1+i, value=v)
                c.font = font(bold=(i == 0), size=10); c.fill = WHITE_FILL
                c.alignment = align(h='left' if i <= 1 else 'center')
                c.border = border_all()
            row += 1
        row += 1

    # ── Section 4b (Comp C only): Post-chemo / Overall head-to-head ─────────
    if comparison_rows:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        c = ws.cell(row=row, column=1,
                    value="Comp C — head-to-head comparison (TORS+CRT vs CRT). "
                          "Pre-chemo comparison omitted because CRT first_rt_date in claims "
                          "captures planning, not delivery; comparison begins post-chemo.")
        c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
        c.alignment = align(h='left', wrap=False); c.border = border_all()
        row += 1
        hdrs = ['Outcome', 'View', 'TORS + CRT', 'CRT', 'OR (95% CI)', 'p']
        for i, h in enumerate(hdrs):
            c = ws.cell(row=row, column=1+i, value=h)
            c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
            c.alignment = align(); c.border = border_all()
        row += 1

        last_outcome = None
        for (outcome, view, t_str, c_str, stats) in comparison_rows:
            or_v, lo, hi, p = stats
            if np.isnan(or_v):
                or_str, p_str = 'N/A', 'N/A'
            else:
                or_str = f"{or_v:.2f} ({lo:.2f}–{hi:.2f})"
                p_str  = "<0.0001" if p < 0.0001 else f"{p:.4f}"
            sig = (not np.isnan(or_v)) and (p < 0.05)
            outcome_cell_val = outcome if outcome != last_outcome else ''
            last_outcome = outcome
            row_fill = SIG_FILL if sig else WHITE_FILL
            vals = [outcome_cell_val, view, t_str, c_str, or_str, p_str]
            for i, v in enumerate(vals):
                c = ws.cell(row=row, column=1+i, value=v)
                c.font = font(bold=(i == 0 and outcome_cell_val) or sig, size=10)
                c.fill = row_fill
                c.alignment = align(h='left' if i <= 1 else 'center')
                c.border = border_all()
            row += 1
        row += 1

    # ── Section 5: Delayed toxicity after treatment completion (≥90d washout) ─
    # Anchored on last RT/chemo date. All-comers + incident (finished event-free).
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    dur_t = df[df['tors'] == 1]['days_tx_to_completion']
    dur_c = df[df['tors'] == 0]['days_tx_to_completion']
    c = ws.cell(row=row, column=1,
                value=f"Delayed toxicity after treatment completion (≥90d washout; anchor = last RT/chemo date). "
                      f"Treatment duration median: {tors_label} {dur_t.median():.0f}d, "
                      f"{ctrl_label} {dur_c.median():.0f}d.")
    c.font = font(bold=True, size=11, color='FFFFFF'); c.fill = OUT_FILL
    c.alignment = align(h='left', wrap=False); c.border = border_all()
    row += 1

    for out_lbl, prefix, through_col in DELAYED_OUTCOMES:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        c = ws.cell(row=row, column=1, value=f"Delayed {out_lbl} — cumulative incidence after completion")
        c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = OUT_FILL
        c.alignment = align(h='left', wrap=False); c.border = border_all()
        row += 1

        hdrs = ['Landmark', 'View', tors_label, ctrl_label, 'OR (95% CI)', 'p']
        for i, h in enumerate(hdrs):
            c = ws.cell(row=row, column=1+i, value=h)
            c.font = font(bold=True, size=10, color='FFFFFF'); c.fill = HDR_FILL
            c.alignment = align(); c.border = border_all()
        row += 1

        for view_lbl, incident in [('All-comers', False), ('Incident', True)]:
            for lm_lbl, L in DELAYED_LANDMARKS:
                nt, et, nc, ec, or_v, lo, hi, p = delayed_or(df, prefix, through_col, L, incident)
                t_str = f"{et}/{nt} ({100*et/nt:.1f}%)" if nt else "N/A"
                c_str = f"{ec}/{nc} ({100*ec/nc:.1f}%)" if nc else "N/A"
                sig = (not np.isnan(or_v)) and (p < 0.05)
                if np.isnan(or_v):
                    or_str, p_str = 'N/A', 'N/A'
                else:
                    or_str = f"{or_v:.2f} ({lo:.2f}–{hi:.2f})"
                    p_str  = "<0.0001" if p < 0.0001 else f"{p:.4f}"
                row_fill = SIG_FILL if sig else WHITE_FILL
                view_cell = view_lbl if lm_lbl == DELAYED_LANDMARKS[0][0] else ''
                vals = [lm_lbl, view_cell, t_str, c_str, or_str, p_str]
                for i, v in enumerate(vals):
                    c = ws.cell(row=row, column=1+i, value=v)
                    c.font = font(bold=sig, size=10); c.fill = row_fill
                    c.alignment = align(h='left' if i == 1 else 'center'); c.border = border_all()
                row += 1
        row += 1

    # ── Column widths ────────────────────────────────────────────────────────
    ws.column_dimensions['A'].width = 28
    for i in range(2, 11):
        ws.column_dimensions[get_column_letter(i)].width = 18
    ws.freeze_panes = 'B3'

con.close()
wb.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
