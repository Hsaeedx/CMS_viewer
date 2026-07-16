"""
outcomes_analysis.py

Multi-interval functional outcomes analysis for OPSCC PSM cohort.

For G-tube and SLP: cumulative-incidence rates at 14d, 30d, 90d, 180d, 1yr, 3yr
with unconditional and at-risk-conditional rates + 95% Wilson CIs.
ORs reported at 90d as the canonical comparison timepoint (forest plot).

For dysphagia: ORs at 6mo / 1yr / 3yr / 5yr / anytime
via the existing compute_or helper.

Three comparisons:
  A: TORS alone vs RT alone   (C77/nodal patients excluded at PSM stage)
  B: TORS + RT  vs CRT
  C: TORS + CRT vs CRT

Reads from pre-built SQL tables:
  opscc_survival          — death date + Dec-31 censor
  opscc_ffs_dates         — FFS dropout date per patient
  opscc_propensity        — PSM match flags
  opscc_outcomes          — dysphagia flags
  opscc_slp               — multi-interval SLP cumulative-incidence flags
  opscc_gtube_dependence  — multi-interval G-tube cumulative-incidence flags
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm

DB_PATH = r"F:\CMS\cms_data.duckdb"

COMPARISONS = [
    ('A', 'psm_matched_A', 'TORS alone', 'RT alone'),
    ('B', 'psm_matched_B', 'TORS + RT',  'CRT'),
    ('C', 'psm_matched_C', 'TORS + CRT', 'CRT'),
]

# For G-tube: PRIMARY = placement-only; SECONDARY = any-event (incl. Z93.1)
# For SLP: only one flag set exists.
GTUBE_INTERVALS = [
    ('14d',   14,   'placement_by_14d'),
    ('30d',   30,   'placement_by_30d'),
    ('90d',   90,   'placement_by_90d'),
    ('180d',  180,  'placement_by_180d'),
    ('1-yr',  365,  'placement_by_365d'),
    ('3-yr',  1095, 'placement_by_1095d'),
]
GTUBE_SENS_INTERVALS = [
    ('14d',   14,   'event_by_14d'),
    ('30d',   30,   'event_by_30d'),
    ('90d',   90,   'event_by_90d'),
    ('180d',  180,  'event_by_180d'),
    ('1-yr',  365,  'event_by_365d'),
    ('3-yr',  1095, 'event_by_1095d'),
]
SLP_INTERVALS = [
    ('14d',   14,   'event_by_14d'),
    ('30d',   30,   'event_by_30d'),
    ('90d',   90,   'event_by_90d'),
    ('180d',  180,  'event_by_180d'),
    ('1-yr',  365,  'event_by_365d'),
    ('3-yr',  1095, 'event_by_1095d'),
]

CANONICAL_FLAG = 'placement_by_90d'

OUTCOMES_BINARY = [
    ('Dysphagia', 'has_dysphagia', 'days_dys'),
]

TIMEPOINTS = [
    ('6-month',  182),
    ('1-year',   365),
    ('3-year',  1095),
    ('5-year',  1825),
    ('Anytime',  None),
]


def wilson_ci(k, n):
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (float('nan'), float('nan'))
    z = 1.96
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half


def or_p(et, nt, ec, nc):
    """Plain OR + 95% CI + chi-square p."""
    a, b, c, d = et, nt - et, ec, nc - ec
    if 0 in (a, b, c, d) or nt < 10 or nc < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')
    or_v   = (a*d) / (b*c)
    log_or = np.log(or_v)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
    _, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return or_v, lo, hi, p


def irr_p(et, pdt, ec, pdc):
    """Poisson-based incidence rate ratio + 95% CI + Wald p-value."""
    from scipy.stats import norm as _norm
    if et == 0 or ec == 0 or pdt == 0 or pdc == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    irr   = (et / pdt) / (ec / pdc)
    se    = np.sqrt(1/et + 1/ec)
    lo, hi = np.exp(np.log(irr) - 1.96*se), np.exp(np.log(irr) + 1.96*se)
    z     = np.log(irr) / se
    p     = 2 * (1 - _norm.cdf(abs(z)))
    return irr, lo, hi, p


def mcnemar(b, c):
    """McNemar's test on discordant pair counts b and c.
    Returns discordant OR = b/c, exact binomial 95% CI, and p-value
    (continuity-corrected χ²)."""
    from scipy.stats import binomtest
    n_disc = b + c
    if n_disc == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    if c == 0:
        or_v = float('inf')
        lo, hi = float('nan'), float('nan')
    else:
        or_v = b / c
        # Exact binomial CI on b/(b+c), then convert to OR scale
        ci = binomtest(b, n_disc, p=0.5).proportion_ci(method='exact')
        p_lo, p_hi = ci.low, ci.high
        lo = p_lo / (1 - p_lo) if p_lo < 1 else float('nan')
        hi = p_hi / (1 - p_hi) if p_hi < 1 else float('inf')
    # McNemar χ² with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n_disc if n_disc > 0 else 0
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2, df=1)
    return or_v, lo, hi, p_val


def compute_or(sub, has_col, days_col, cutoff):
    """Legacy follow-up-eligibility OR for binary outcomes (dysphagia)."""
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

    t_, c_ = elig[elig['tors'] == 1], elig[elig['tors'] == 0]
    nt, nc = len(t_), len(c_)
    et, ec = int(t_['ev'].sum()), int(c_['ev'].sum())
    pt = f"{100*et/nt:.1f}" if nt > 0 else "N/A"
    pc = f"{100*ec/nc:.1f}" if nc > 0 else "N/A"
    or_v, lo, hi, p = or_p(et, nt, ec, nc)
    return nt, et, pt, nc, ec, pc, or_v, lo, hi, p


def cuminc_row(sub, flag_col, cutoff_days):
    """
    Cumulative-incidence rate at cutoff_days post-tx, both unconditional
    (treat death/disenroll as no-event) and at-risk-conditional
    (denominator = alive AND FFS-enrolled at first_tx_date + cutoff_days).
    """
    out = {}
    for arm, label in [(1, 'tors'), (0, 'ctrl')]:
        s = sub[sub['tors'] == arm]
        n_uncond = len(s)
        k_uncond = int(s[flag_col].fillna(False).astype(int).sum())
        out[f'n_uncond_{label}'] = n_uncond
        out[f'k_uncond_{label}'] = k_uncond
        out[f'p_uncond_{label}'] = 100 * k_uncond / n_uncond if n_uncond else float('nan')

        at_risk = s[s['follow_up_days'] >= cutoff_days]
        n_cond = len(at_risk)
        k_cond = int(at_risk[flag_col].fillna(False).astype(int).sum())
        out[f'n_cond_{label}']  = n_cond
        out[f'k_cond_{label}']  = k_cond
        out[f'p_cond_{label}']  = 100 * k_cond / n_cond if n_cond else float('nan')

    out['or_v'], out['or_lo'], out['or_hi'], out['or_p'] = or_p(
        out['k_uncond_tors'], out['n_uncond_tors'],
        out['k_uncond_ctrl'], out['n_uncond_ctrl'])
    return out


# ── Delayed toxicity after treatment completion (≥90d washout) ─────────────────
# Anchored on the last RT/chemo date (course completion), not first_tx_date.
# Marker for delayed swallowing failure / G-tube dependence.
DELAYED_LANDMARKS = [('6-mo', 180), ('1-yr', 365), ('2-yr', 730), ('3-yr', 1095)]
DELAYED_OUTCOMES = [
    ('G-tube placement', 'g_delayed_by_',   'g_through_completion90'),
    ('SLP visit',        'slp_delayed_by_', 'slp_through_completion90'),
    ('Dysphagia',        'dys_delayed_by_', 'dys_through_completion90'),
]


def delayed_row(sub, flag_prefix, through_col, L, incident_only):
    """At-risk cumulative incidence of a delayed event by L days after completion.

    Eligible = followed past landmark L after completion OR had the event by L.
    Incident view further restricts to patients who reached completion+90d
    event-free (`through_col` False). Rows with a NULL delayed flag (e.g.
    pre-existing dysphagia) are dropped.
    """
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


# ── Main loop ─────────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12; SET temp_directory='F:\\CMS\\duckdb_temp';")

for comp, match_col, tors_label, ctrl_label in COMPARISONS:

    print(f"\n{'#'*78}")
    print(f"  COMPARISON {comp}: {tors_label}  vs  {ctrl_label}")
    print(f"{'#'*78}")

    df = con.execute(f"""
        SELECT
            s.DSYSRTKY, s.tx_group, s.first_tx_date,
            s.age_at_dx, s.van_walraven_score,
            f.ffs_censor_date,
            DATEDIFF('day', s.first_tx_date, f.ffs_censor_date) AS follow_up_days,
            o.has_dysphagia,
            DATEDIFF('day', s.first_tx_date, o.first_dysphagia_date) AS days_dys,
            co.first_chemo_date AS first_chemo_date,
            DATEDIFF('day', s.first_tx_date, co.first_chemo_date) AS days_tx_to_chemo,
            p.psm_match_id_C AS pair_id_C,
            g.first_post_placement_day AS g_first_post_placement_day,
            slp.first_post_day_from_tx AS slp_first_post_day_from_tx,
            g.days_dx_to_tx           AS g_days_dx_to_tx,
            g.pre_placement_any       AS g_pre_placement_any,
            g.pre_event_any           AS g_pre_event_any,
            g.placement_by_14d        AS g_placement_by_14d,
            g.placement_by_30d        AS g_placement_by_30d,
            g.placement_by_90d        AS g_placement_by_90d,
            g.placement_by_180d       AS g_placement_by_180d,
            g.placement_by_365d       AS g_placement_by_365d,
            g.placement_by_1095d      AS g_placement_by_1095d,
            g.event_by_14d            AS g_event_by_14d,
            g.event_by_30d            AS g_event_by_30d,
            g.event_by_90d            AS g_event_by_90d,
            g.event_by_180d           AS g_event_by_180d,
            g.event_by_365d           AS g_event_by_365d,
            g.event_by_1095d          AS g_event_by_1095d,
            g.total_post_placements   AS g_total_placements,
            g.total_post_events       AS g_total_events,
            slp.pre_event_any      AS slp_pre_event_any,
            slp.event_by_14d       AS slp_event_by_14d,
            slp.event_by_30d       AS slp_event_by_30d,
            slp.event_by_90d       AS slp_event_by_90d,
            slp.event_by_180d      AS slp_event_by_180d,
            slp.event_by_365d      AS slp_event_by_365d,
            slp.event_by_1095d     AS slp_event_by_1095d,
            slp.total_post_events  AS slp_total_events,
            -- Delayed toxicity after treatment completion (90d washout)
            g.tx_completion_date              AS tx_completion_date,
            g.days_tx_to_completion           AS days_tx_to_completion,
            g.placement_through_completion90  AS g_through_completion90,
            g.first_delayed_placement_day     AS g_first_delayed_placement_day,
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
        JOIN opscc_propensity p       USING (DSYSRTKY)
        JOIN opscc_ffs_dates  f       USING (DSYSRTKY)
        JOIN opscc_outcomes   o       USING (DSYSRTKY)
        LEFT JOIN opscc_cohort co     ON co.DSYSRTKY = s.DSYSRTKY
        LEFT JOIN opscc_slp   slp     USING (DSYSRTKY)
        LEFT JOIN opscc_gtube_dependence g USING (DSYSRTKY)
        WHERE p.{match_col} = TRUE
          AND s.tx_group IN ('{tors_label}', '{ctrl_label}')
    """).df()

    if len(df) == 0:
        print(f"  No matched patients for Comparison {comp}. Skipping.")
        continue

    df['tors'] = (df['tx_group'] == tors_label).astype(int)
    n_tors = int(df['tors'].sum())
    n_ctrl = int((df['tors'] == 0).sum())
    print(f"N: {len(df):,}  |  {tors_label}: {n_tors:,}   {ctrl_label}: {n_ctrl:,}")

    # Follow-up measured from treatment completion (for delayed-toxicity at-risk sets)
    df['foll_after_comp'] = df['follow_up_days'] - df['days_tx_to_completion']

    # ── Baseline descriptors (dx→tx + pre-tx placements) ────────────────────
    print(f"\n{'-'*78}\n  BASELINE DESCRIPTORS\n{'-'*78}")
    for arm, label in [(1, tors_label), (0, ctrl_label)]:
        s = df[df['tors'] == arm]
        dx_med = s['g_days_dx_to_tx'].median()
        dx_p25 = s['g_days_dx_to_tx'].quantile(0.25)
        dx_p75 = s['g_days_dx_to_tx'].quantile(0.75)
        g_pre  = s['g_pre_placement_any'].fillna(False).sum()
        slp_pre = s['slp_pre_event_any'].fillna(False).sum()
        print(f"  {label:<14}  dx→tx median (IQR): {dx_med:.0f} ({dx_p25:.0f}-{dx_p75:.0f}) days"
              f"   G-tube placement pre-tx: {g_pre}/{len(s)} ({100*g_pre/len(s):.1f}%)"
              f"   SLP pre-tx: {slp_pre}/{len(s)} ({100*slp_pre/len(s):.1f}%)")

    # ── Multi-interval cumulative incidence (G-tube + SLP) ─────────────────
    # G-tube uses placement-only flags; SLP uses any-event flags.
    for label, src_prefix, intervals in [
        ('G-TUBE PLACEMENT', 'g_',   GTUBE_INTERVALS),
        ('SLP VISIT',        'slp_', SLP_INTERVALS),
    ]:
        print(f"\n{'-'*78}\n  CUMULATIVE INCIDENCE: {label}   (OR<1 favors {tors_label})\n{'-'*78}")
        print(f"  {'Interval':<8}"
              f"  {tors_label[:10]:<10} {'(uncond)':>9} {'(cond)':>8}"
              f"  {ctrl_label[:10]:<10} {'(uncond)':>9} {'(cond)':>8}"
              f"  {'OR (95% CI)':<22} {'p':>9}")
        print(f"  {'-'*116}")
        for int_lbl, cutoff_d, _flag in intervals:
            flag_col = src_prefix + _flag
            row = cuminc_row(df, flag_col, cutoff_d)
            or_v, lo, hi, p = row['or_v'], row['or_lo'], row['or_hi'], row['or_p']
            or_str = f"{or_v:.2f} ({lo:.2f}-{hi:.2f})" if not np.isnan(or_v) else "N/A"
            p_str  = "<0.0001" if (not np.isnan(p) and p < 0.0001) else (f"{p:.4f}" if not np.isnan(p) else "N/A")
            print(f"  {int_lbl:<8}"
                  f"  {row['k_uncond_tors']:>4}/{row['n_uncond_tors']:<5}"
                  f" {row['p_uncond_tors']:>8.1f}%  {row['p_cond_tors']:>6.1f}% "
                  f"  {row['k_uncond_ctrl']:>4}/{row['n_uncond_ctrl']:<5}"
                  f" {row['p_uncond_ctrl']:>8.1f}%  {row['p_cond_ctrl']:>6.1f}% "
                  f"  {or_str:<22} {p_str:>9}")

    # ── Binary outcome (dysphagia) — legacy compute_or ─────────────────────
    df['elix_grp'] = (df['van_walraven_score'] > 0).map({False: 'Low', True: 'High'})
    strata = [
        ('All matched',         df),
        ('Age < 75',            df[df['age_at_dx'] < 75]),
        ('Age >= 75',           df[df['age_at_dx'] >= 75]),
        ('Elix Low  (VW<=0)',   df[df['elix_grp'] == 'Low']),
        ('Elix High (VW>0)',    df[df['elix_grp'] == 'High']),
    ]

    W = 118
    for out_lbl, has_col, days_col in OUTCOMES_BINARY:
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
                    ci_str = f"({lo:.2f}-{hi:.2f})"
                    p_str  = "<0.0001" if p < 0.0001 else f"{p:.4f}"
                print(f"  {sl:<26} {tp_lbl:<9}"
                      f" {nt:>6,} {et:>6} {pt:>5}%"
                      f" {nc:>6,} {ec:>6} {pc:>5}%"
                      f"  {or_str:>6}  {ci_str:<16}  {p_str:>8}")
            print(f"  {'-'*W}")

    # ── Delayed toxicity after treatment completion (≥90d washout) ─────────────
    # Anchored on last RT/chemo date. All-comers + incident (finished tube-free).
    print(f"\n{'#'*78}")
    print(f"  DELAYED TOXICITY AFTER TREATMENT COMPLETION  (>=90d washout)")
    print(f"  Anchor = last RT/chemo date (course completion); OR<1 favors {tors_label}")
    print(f"{'#'*78}")

    print(f"\n  Treatment duration (first tx -> last RT/chemo) & clean-completion counts:")
    for arm, label in [(1, tors_label), (0, ctrl_label)]:
        s = df[df['tors'] == arm]
        dur = s['days_tx_to_completion']
        med = dur.median(); p25 = dur.quantile(0.25); p75 = dur.quantile(0.75)
        clean = int((s['g_through_completion90'].fillna(True) == False).sum())
        print(f"  {label:<14}  duration median (IQR): {med:.0f} ({p25:.0f}-{p75:.0f}) days"
              f"   finished G-tube-free: {clean}/{len(s)} ({100*clean/len(s):.1f}%)")

    for out_lbl, prefix, through_col in DELAYED_OUTCOMES:
        print(f"\n{'-'*92}\n  DELAYED {out_lbl.upper()}  (cumulative incidence after completion)\n{'-'*92}")
        for view_lbl, incident in [('All-comers', False), ('Incident (finished clean)', True)]:
            print(f"  {view_lbl}")
            print(f"  {'Landmark':<8}  {tors_label[:12]:<12} {'%':>7}  {ctrl_label[:12]:<12} {'%':>7}"
                  f"  {'OR (95% CI)':<22} {'p':>9}")
            for lm_lbl, L in DELAYED_LANDMARKS:
                nt, et, nc, ec, or_v, lo, hi, p = delayed_row(df, prefix, through_col, L, incident)
                pt = 100*et/nt if nt else float('nan')
                pc = 100*ec/nc if nc else float('nan')
                or_str = f"{or_v:.2f} ({lo:.2f}-{hi:.2f})" if not np.isnan(or_v) else "N/A"
                p_str  = "<0.0001" if (not np.isnan(p) and p < 0.0001) else (f"{p:.4f}" if not np.isnan(p) else "N/A")
                print(f"  {lm_lbl:<8}  {et:>4}/{nt:<5} {pt:>6.1f}%  {ec:>4}/{nc:<5} {pc:>6.1f}%"
                      f"  {or_str:<22} {p_str:>9}")

    # ── Comp C only: chemo-aligned analysis ────────────────────────────────────
    # Re-anchor both arms on first_chemo_date (start of chemoradiation) so the
    # comparison is apples-to-apples by treatment phase. For TORS+CRT patients,
    # any G-tube/SLP/dysphagia events before chemo started are reported as
    # "pre-CRT surgical-era cost" descriptively. Then both arms are compared on
    # cumulative incidence from chemo start at standard intervals.
    if comp != 'C':
        continue

    print(f"\n{'#'*78}")
    print(f"  COMP C CHEMO-ALIGNED ANALYSIS")
    print(f"  Both arms anchored on first_chemo_date")
    print(f"{'#'*78}")

    # Days from chemo start to first event (negative = before chemo started)
    df = df.copy()
    df['gtube_days_from_chemo'] = df['g_first_post_placement_day']  - df['days_tx_to_chemo']
    df['slp_days_from_chemo']   = df['slp_first_post_day_from_tx']  - df['days_tx_to_chemo']
    df['dys_days_from_chemo']   = df['days_dys']                    - df['days_tx_to_chemo']

    tors_arm = df[df['tors'] == 1]
    ctrl_arm = df[df['tors'] == 0]

    # ── Section A: Pre-CRT surgical-era cost (TORS+CRT only) ──────────────────
    print(f"\n  PRE-CRT SURGICAL ERA (TORS+CRT only — events before chemo started)")
    print(f"  CRT arm has no analog: they hadn't begun any treatment at this point.")
    print(f"  {'-'*76}")
    for outcome_label, event_col, days_col in [
        ('G-tube',    'g_first_post_placement_day', 'days_tx_to_chemo'),
        ('Dysphagia', 'days_dys',                   'days_tx_to_chemo'),
        ('SLP',       'slp_first_post_day_from_tx', 'days_tx_to_chemo'),
    ]:
        # Restrict to TORS+CRT patients with a valid pre-CRT window (days_tx_to_chemo >= 0)
        sub = tors_arm[tors_arm['days_tx_to_chemo'].notna() & (tors_arm['days_tx_to_chemo'] >= 0)]
        n = len(sub)
        pre = sub[event_col].notna() & (sub[event_col] < sub['days_tx_to_chemo'])
        n_pre = int(pre.sum())
        print(f"  {outcome_label:<10}  {n_pre}/{n} ({100*n_pre/n:.1f}%) had event before chemo started")
    n_pre_tors_chemo = int((tors_arm['days_tx_to_chemo'] < 0).sum())
    print(f"\n  Note: {n_pre_tors_chemo} TORS+CRT patients had chemo before TORS — excluded above.")

    # ── Section B: Post-chemo-start cumulative incidence (both arms) ──────────
    print(f"\n  POST-CHEMO-START CUMULATIVE INCIDENCE (anchored on first_chemo_date)")
    print(f"  Both arms aligned on day 0 = chemoradiation start.")
    print(f"  {'-'*112}")

    def _post_chemo_flag(event_days_from_chemo, T):
        """True if event occurred between 0 and T days after chemo start."""
        if pd.isna(event_days_from_chemo):
            return False
        return 0 <= event_days_from_chemo <= T

    INTERVALS_POST = [(14, '14d'), (30, '30d'), (90, '90d'),
                      (180, '180d'), (365, '1-yr'), (1095, '3-yr')]

    for outcome_label, event_col_chemo in [
        ('G-tube',    'gtube_days_from_chemo'),
        ('SLP',       'slp_days_from_chemo'),
        ('Dysphagia', 'dys_days_from_chemo'),
    ]:
        print(f"\n  {outcome_label.upper()} — events from chemo start")
        print(f"  {'Interval':<8}  {'TORS+CRT':<14}  {'CRT':<14}  "
              f"{'OR (95% CI)':<22}  {'p':>9}")
        print(f"  {'-'*80}")
        for T, lbl in INTERVALS_POST:
            tors_flag = tors_arm[event_col_chemo].apply(lambda d: _post_chemo_flag(d, T))
            ctrl_flag = ctrl_arm[event_col_chemo].apply(lambda d: _post_chemo_flag(d, T))
            et, nt = int(tors_flag.sum()), len(tors_arm)
            ec, nc = int(ctrl_flag.sum()), len(ctrl_arm)
            or_v, lo_, hi_, pv = or_p(et, nt, ec, nc)
            if np.isnan(or_v):
                or_str, p_str = 'N/A', 'N/A'
            else:
                or_str = f"{or_v:.2f} ({lo_:.2f}-{hi_:.2f})"
                p_str  = "<0.0001" if pv < 0.0001 else f"{pv:.4f}"
            print(f"  {lbl:<8}  {et:>3}/{nt:<3} ({100*et/nt:>5.1f}%)  "
                  f"{ec:>3}/{nc:<3} ({100*ec/nc:>5.1f}%)  "
                  f"{or_str:<22}  {p_str:>9}")

    # ── Section C: Total cumulative burden through 3 yr from chemo ───────────
    print(f"\n  TOTAL CUMULATIVE BURDEN BY 3 YEARS FROM CHEMO START")
    print(f"  TORS+CRT total = pre-CRT events + post-chemo events through 3 yr")
    print(f"  {'-'*78}")
    for outcome_label, event_col, event_col_chemo in [
        ('G-tube',    'g_first_post_placement_day', 'gtube_days_from_chemo'),
        ('Dysphagia', 'days_dys',                   'dys_days_from_chemo'),
        ('SLP',       'slp_first_post_day_from_tx', 'slp_days_from_chemo'),
    ]:
        # TORS+CRT total: ANY event from first_tx_date through chemo + 3yr (= days_tx_to_chemo + 1095)
        tors_total = tors_arm.apply(
            lambda r: pd.notna(r[event_col]) and r[event_col] <= (r['days_tx_to_chemo'] + 1095),
            axis=1)
        et = int(tors_total.sum()); nt = len(tors_arm)
        # CRT: any event in [0, 1095] from chemo (= first_post_placement_day in [days_tx_to_chemo,
        # days_tx_to_chemo + 1095]) — for CRT, days_tx_to_chemo ≈ 0 so it's effectively [0, 1095]
        ctrl_total = ctrl_arm[event_col_chemo].apply(lambda d: _post_chemo_flag(d, 1095))
        ec = int(ctrl_total.sum()); nc = len(ctrl_arm)
        or_v, lo_, hi_, pv = or_p(et, nt, ec, nc)
        or_str = f"{or_v:.2f} ({lo_:.2f}-{hi_:.2f})" if not np.isnan(or_v) else "N/A"
        p_str  = ("<0.0001" if (not np.isnan(pv) and pv < 0.0001)
                  else (f"{pv:.4f}" if not np.isnan(pv) else "N/A"))
        print(f"  {outcome_label:<10}  TORS+CRT {et}/{nt} ({100*et/nt:.1f}%)   "
              f"CRT {ec}/{nc} ({100*ec/nc:.1f}%)   {or_str}   p={p_str}")

    # ── Section D: TORS+CRT post-surgical descriptive (NON-COMPARATIVE) ───────
    # Pre-chemo events for TORS+CRT reflect post-surgical recovery from a real
    # intervention. The CRT arm has no comparable "pre-chemo" period — their
    # first_rt_date is the planning visit, not actual radiation delivery, so
    # their pre-chemo "events" are baseline disease severity, not treatment
    # exposure. We therefore report TORS+CRT post-op rates descriptively and
    # do not compare to CRT.
    print(f"\n{'#'*78}")
    print(f"  COMP C — TORS+CRT POST-SURGICAL DESCRIPTIVE (no comparator)")
    print(f"  Events during the post-op recovery period before chemo started")
    print(f"{'#'*78}")
    T_arm = df[df['tors'] == 1].copy()
    C_arm = df[df['tors'] == 0].copy()
    T_valid = T_arm[T_arm['days_tx_to_chemo'].fillna(-1) >= 0]
    pdt = T_valid['days_tx_to_chemo'].clip(lower=0).sum()
    _mean_d = T_valid['days_tx_to_chemo'].mean()
    _sd_d   = T_valid['days_tx_to_chemo'].std()
    _med    = T_valid['days_tx_to_chemo'].median()
    _p25    = T_valid['days_tx_to_chemo'].quantile(0.25)
    _p75    = T_valid['days_tx_to_chemo'].quantile(0.75)
    _max    = T_valid['days_tx_to_chemo'].max()
    print(f"  N with valid post-op window: {len(T_valid)}")
    print(f"  TORS-to-chemo days: mean {_mean_d:.1f} (SD {_sd_d:.1f}); "
          f"median {_med:.0f} (IQR {_p25:.0f}–{_p75:.0f}); "
          f"range 0–{_max:.0f}")
    print(f"  Total person-days: {int(pdt):,}")
    print(f"\n  {'Outcome':<10}  {'Events':<14}  {'Rate per 1000 person-days':<27}")
    print(f"  {'-'*55}")
    for outcome_label, col_name in [
        ('G-tube',    'g_first_post_placement_day'),
        ('Dysphagia', 'days_dys'),
        ('SLP',       'slp_first_post_day_from_tx'),
    ]:
        et = int((T_valid[col_name].notna() &
                  (T_valid[col_name] < T_valid['days_tx_to_chemo'])).sum())
        rate = 1000 * et / pdt if pdt else float('nan')
        print(f"  {outcome_label:<10}  {et}/{len(T_valid)} ({100*et/len(T_valid):.1f}%)  {rate:>10.2f}")

    # ── Section E: head-to-head comparison summary (Post-chemo / Overall) ────
    print(f"\n{'#'*78}")
    print(f"  COMP C — HEAD-TO-HEAD COMPARISON (anchored on chemoradiation start)")
    print(f"  Both arms aligned on chemo start; comparisons begin from this point.")
    print(f"{'#'*78}")
    print(f"  {'Outcome':<10}  {'View':<32}  {'TORS+CRT':<18}  {'CRT':<18}  "
          f"{'OR (95% CI)':<22}  {'p':>9}")
    print(f"  {'-'*120}")

    def _fmt_or(stats):
        or_v, lo, hi, p = stats
        if np.isnan(or_v):
            return 'N/A', 'N/A'
        return f"{or_v:.2f} ({lo:.2f}-{hi:.2f})", \
               ('<0.0001' if p < 0.0001 else f"{p:.4f}")

    for outcome_label, col_name in [
        ('G-tube',    'g_first_post_placement_day'),
        ('Dysphagia', 'days_dys'),
        ('SLP',       'slp_first_post_day_from_tx'),
    ]:
        d_T = T_arm['days_tx_to_chemo'].fillna(0)
        d_C = C_arm['days_tx_to_chemo'].fillna(0)
        et_post = int((T_arm[col_name].notna() &
                       ((T_arm[col_name] - d_T) >= 0) &
                       ((T_arm[col_name] - d_T) <= 1095)).sum())
        ec_post = int((C_arm[col_name].notna() &
                       ((C_arm[col_name] - d_C) >= 0) &
                       ((C_arm[col_name] - d_C) <= 1095)).sum())
        or_post = or_p(et_post, len(T_arm), ec_post, len(C_arm))

        et_all = int((T_arm[col_name].notna() & T_arm[col_name].between(0, 1095)).sum())
        ec_all = int((C_arm[col_name].notna() & C_arm[col_name].between(0, 1095)).sum())
        or_all = or_p(et_all, len(T_arm), ec_all, len(C_arm))

        for view, t_n, t_N, c_n, c_N, stats in [
            ('Post-chemo (3-yr from chemo)',
             et_post, len(T_arm), ec_post, len(C_arm), or_post),
            ('Overall (3-yr from first tx)',
             et_all, len(T_arm), ec_all, len(C_arm), or_all),
        ]:
            or_str, p_str = _fmt_or(stats)
            label = outcome_label if view.startswith('Post') else ''
            print(f"  {label:<10}  {view:<32}  "
                  f"{t_n}/{t_N} ({100*t_n/t_N:.1f}%)".ljust(20) +
                  f"{c_n}/{c_N} ({100*c_n/c_N:.1f}%)".ljust(20) +
                  f"{or_str:<22}  {p_str:>9}")
        print(f"  {'-'*120}")

con.close()
