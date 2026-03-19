"""
stroke_analysis.py

Analysis of stroke + SLP timing outcomes on PSM-matched cohort.

Two pairwise comparisons (each run on its own matched set):
  Comparison A: 0-14d  vs 31-90d  (psm_matched_A = TRUE)
  Comparison B: 15-30d vs 31-90d  (psm_matched_B = TRUE)

Outputs per comparison (printed):
  1. OR tables for binary outcomes at 90d / 180d / 365d
     (measured from discharge — descriptive; subject to immortal time)
  2. Time-varying Cox PH — PRIMARY analysis
     Person-time split at days_to_slp_outpt:
       pre-SLP segment  [0, slp_day)       trt=0 for both groups
       post-SLP segment [slp_day, censor]  trt=1 (treated) or 0 (reference)
     Eliminates immortal time bias; death treated as competing event
     for non-mortality outcomes.
  3. KM / cumulative-incidence estimates at each time point
  4. Medicare cost analysis (Mann-Whitney, 1-year post-discharge)

Run after:
  stroke_propensity.sql  (builds stroke_propensity)
  stroke_psm.py          (populates psm_matched_A/B in stroke_propensity)
"""

import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

DB_PATH    = r"F:\CMS\cms_data.duckdb"
MAX_FOLLOW = 365
TIMEPOINTS = [90, 180, 365]

COMPARISONS = [
    ('A', '0-14d',  '31-90d', 'psm_matched_A'),
    ('B', '15-30d', '31-90d', 'psm_matched_B'),
]


# ── Load data ──────────────────────────────────────────────────────────────────

def load_cohort(con, match_col):
    return con.execute(f"""
        SELECT
            p.DSYSRTKY,
            p.slp_timing_group,
            p.dschg_group,
            p.age_at_adm,
            p.sex,
            p.race,
            p.stroke_type,
            p.drg_cd,
            p.adm_year,
            p.index_los,
            p.dysphagia_poa,
            p.aspiration_poa,
            p.mech_vent,
            p.peg_placed,
            p.trach_placed,
            p.van_walraven_score,
            p.afib,
            p.hypertension,
            p.days_to_slp_outpt,
            o.death_date,
            o.days_to_death,
            o.days_to_readmit,
            o.n_readmissions_365d,
            o.days_to_recur_stroke,
            o.days_to_aspiration,
            o.days_to_dysphagia,
            o.days_to_gtube,
            o.days_to_snf,
            o.snf_30d,
            o.hha_90d,
            o.total_pmt_365d,
            o.snf_pmt_365d,
            o.hha_pmt_365d,
            c.index_pmt,
            c.index_dschg_date
        FROM stroke_propensity p
        JOIN stroke_outcomes   o ON o.DSYSRTKY = p.DSYSRTKY
        JOIN stroke_cohort     c ON c.DSYSRTKY = p.DSYSRTKY
        WHERE p.{match_col} = TRUE
    """).df()


# ── Derived columns ────────────────────────────────────────────────────────────

def add_derived(df, treat_grp):
    df = df.copy()
    df['treated'] = (df['slp_timing_group'] == treat_grp).astype(int)

    df['censor_days'] = np.where(
        df['days_to_death'].notna(),
        df['days_to_death'].clip(upper=MAX_FOLLOW),
        MAX_FOLLOW
    )

    for days, label in [(90,'90d'), (180,'180d'), (365,'365d')]:
        df[f'died_{label}']      = (df['days_to_death'].notna()        & (df['days_to_death']        <= days)).astype(int)
        df[f'readmit_{label}']   = (df['days_to_readmit'].notna()      & (df['days_to_readmit']      <= days)).astype(int)
        df[f'recur_{label}']     = (df['days_to_recur_stroke'].notna() & (df['days_to_recur_stroke'] <= days)).astype(int)
        df[f'asp_{label}']       = (df['days_to_aspiration'].notna()   & (df['days_to_aspiration']   <= days)).astype(int)
        df[f'dysphagia_{label}'] = (df['days_to_dysphagia'].notna()    & (df['days_to_dysphagia']    <= days)).astype(int)
        df[f'gtube_{label}']     = (df['days_to_gtube'].notna()        & (df['days_to_gtube']        <= days)).astype(int)

    q1 = df['van_walraven_score'].quantile(1/3)
    q2 = df['van_walraven_score'].quantile(2/3)
    df['elix_grp'] = pd.cut(df['van_walraven_score'],
                             bins=[-np.inf, q1, q2, np.inf],
                             labels=['Low', 'Mid', 'High'])
    return df, q1, q2


# ── OR computation ─────────────────────────────────────────────────────────────

def compute_or(sub, ev_col, days_col, cutoff):
    valid = sub.copy()
    if cutoff is None:
        elig     = valid.copy()
        elig['ev'] = elig[ev_col].astype(int)
    else:
        mask = (
            (valid['censor_days'] >= cutoff) |
            (valid[ev_col].astype(bool) & (valid[days_col] <= cutoff))
        )
        elig = valid[mask].copy()
        elig['ev'] = (elig[ev_col].astype(bool) & (elig[days_col] <= cutoff)).astype(int)

    t_ = elig[elig['treated'] == 1]
    c_ = elig[elig['treated'] == 0]
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


def print_or_table(sect_label, sub, outcomes_list, treat_grp, ctrl_grp):
    print(f"\n{'='*110}")
    print(f"  {sect_label}  |  Treated = {treat_grp!r}  vs  Control = {ctrl_grp!r}")
    print(f"{'='*110}")
    header = (f"  {'Outcome':<30}  {'Cutoff':>7}  "
              f"{'n_Treat':>8} {'ev_Treat':>9} {'%Treat':>7}  "
              f"{'n_Ctrl':>8} {'ev_Ctrl':>9} {'%Ctrl':>7}  "
              f"{'OR':>6} {'95%CI':>14} {'p':>8}")
    print(header)
    print(f"  {'-'*108}")

    for name_o, ev_col, days_col, cutoffs in outcomes_list:
        for cutoff in cutoffs:
            cutoff_str = f"{cutoff}d" if cutoff else "any"
            row = compute_or(sub, ev_col, days_col, cutoff)
            nt, et, pt, nc, ec, pc, or_v, lo, hi, p = row
            if np.isnan(or_v):
                ci_str, or_str, p_str = "N/A", "N/A", "N/A"
            else:
                ci_str = f"[{lo:.2f}, {hi:.2f}]"
                or_str = f"{or_v:.2f}"
                p_str  = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"  {name_o:<30}  {cutoff_str:>7}  "
                  f"{nt:>8,} {et:>9,} {pt:>7}  "
                  f"{nc:>8,} {ec:>9,} {pc:>7}  "
                  f"{or_str:>6} {ci_str:>14} {p_str:>8}")


# ── KM helper ─────────────────────────────────────────────────────────────────

def km_at(kmf, t):
    s  = float(kmf.predict(t))
    ci = kmf.confidence_interval_survival_function_
    idx = min(ci.index.searchsorted(t, side='right'), len(ci) - 1)
    lo  = float(ci.iloc[idx, 0])
    hi  = float(ci.iloc[idx, 1])
    return s, lo, hi


# ── Run analysis for one comparison ───────────────────────────────────────────

def run_analysis(df, comp_label, treat_grp, ctrl_grp):
    sep = f"\n{'#'*110}\n  COMPARISON {comp_label}: {treat_grp!r}  vs  {ctrl_grp!r}  (reference)\n{'#'*110}"
    print(sep)
    print(f"  n = {len(df):,}  |  treated ({treat_grp}): {df['treated'].sum():,}  "
          f"  control ({ctrl_grp}): {(df['treated']==0).sum():,}")

    # ── OR tables ─────────────────────────────────────────────────────────────
    outcomes_list = [
        ('Mortality',         'died_{lbl}',       'days_to_death',        TIMEPOINTS),
        ('Readmission',       'readmit_{lbl}',    'days_to_readmit',      TIMEPOINTS),
        ('Aspiration PNA',    'asp_{lbl}',        'days_to_aspiration',   TIMEPOINTS),
        ('Dysphagia Dx',      'dysphagia_{lbl}',  'days_to_dysphagia',    TIMEPOINTS),
        ('G-tube',            'gtube_{lbl}',      'days_to_gtube',        TIMEPOINTS),
        ('SNF 30d',           'snf_30d',          'days_to_snf',          [None]),
        ('HHA 90d',           'hha_90d',          'days_to_snf',          [None]),
    ]

    # Expand {lbl} templates
    expanded = []
    for name, ev_tmpl, days_col, cutoffs in outcomes_list:
        if '{lbl}' in ev_tmpl:
            for c in cutoffs:
                lbl = f"{c}d"
                expanded.append((name, ev_tmpl.replace('{lbl}', lbl), days_col, [c]))
        else:
            expanded.append((name, ev_tmpl, days_col, cutoffs))

    q1 = df['van_walraven_score'].quantile(1/3)
    q2 = df['van_walraven_score'].quantile(2/3)

    strata = [
        ('All matched',        df),
        ('Ischemic',           df[df['stroke_type'] == 'Ischemic']),
        ('ICH',                df[df['stroke_type'] == 'ICH']),
        ('Age < 75',           df[df['age_at_adm'] < 75]),
        ('Age 75-84',          df[(df['age_at_adm'] >= 75) & (df['age_at_adm'] < 85)]),
        ('Age 85+',            df[df['age_at_adm'] >= 85]),
        ('Dysphagia POA',      df[df['dysphagia_poa'] == 1]),
        ('No Dysphagia POA',   df[df['dysphagia_poa'] == 0]),
        ('Elix Low',           df[df['elix_grp'] == 'Low']),
        ('Elix Mid',           df[df['elix_grp'] == 'Mid']),
        ('Elix High',          df[df['elix_grp'] == 'High']),
        # Discharge disposition strata (matched within group — same-vs-same comparisons)
        ('Disch: Home',        df[df['dschg_group'] == 'Home']),
        ('Disch: Home+HHA',    df[df['dschg_group'] == 'Home+HHA']),
        ('Disch: SNF',         df[df['dschg_group'] == 'SNF']),
        ('Disch: IRF',         df[df['dschg_group'] == 'IRF']),
        ('Disch: LTACH',       df[df['dschg_group'] == 'LTACH']),
    ]

    for strat_label, sub in strata:
        print_or_table(strat_label, sub, expanded, treat_grp, ctrl_grp)

    # ── Time-varying Cox PH ───────────────────────────────────────────────────
    # Person-time is split at days_to_slp_outpt:
    #   Segment 1: [0, days_to_slp_outpt)  trt=0  (pre-SLP; both groups unexposed)
    #   Segment 2: [days_to_slp_outpt, end) trt=group_flag (post-SLP)
    # HR on 'trt' = hazard of treated group (after SLP) vs reference group (after SLP),
    # with immortal time correctly attributed as unexposed person-time.
    print(f"\n{'='*80}")
    print(f"  Time-varying Cox PH — Comparison {comp_label} (censored at {MAX_FOLLOW}d)")
    print(f"  Person-time split at days_to_slp_outpt; trt=0 pre-SLP, trt=1/0 post-SLP")
    print(f"{'='*80}")

    TV_COVARIATES = ['age_at_adm', 'van_walraven_score', 'dysphagia_poa',
                     'aspiration_poa', 'mech_vent', 'peg_placed', 'trach_placed', 'index_los']

    # (label, primary_event_days_col, competing_event_days_col or None)
    # Non-mortality outcomes treat death as a competing event (censored).
    cox_tv_outcomes = [
        ('All-cause mortality',  'days_to_death',        None),
        ('Aspiration PNA',       'days_to_aspiration',   'days_to_death'),
        ('Dysphagia Dx',         'days_to_dysphagia',    'days_to_death'),
    ]

    def build_tv_df(df, event_col, competing_col):
        records = []
        for _, row in df.iterrows():
            slp_day = float(row['days_to_slp_outpt'])
            ev_day  = row[event_col]
            comp_day = row[competing_col] if competing_col and pd.notna(row[competing_col]) else np.nan

            # End time = earliest of: primary event, competing event, MAX_FOLLOW
            candidates = [float(MAX_FOLLOW)]
            if pd.notna(ev_day):   candidates.append(float(ev_day))
            if pd.notna(comp_day): candidates.append(float(comp_day))
            end_time = min(candidates)

            # Primary event occurred if it's the end_time trigger (not death, not censor)
            final_event = int(
                pd.notna(ev_day)
                and float(ev_day) <= MAX_FOLLOW
                and float(ev_day) == end_time
            )

            group_flag = 1 if row['slp_timing_group'] == treat_grp else 0
            base = {col: float(row[col]) if pd.notna(row[col]) else 0.0
                    for col in TV_COVARIATES}

            if end_time <= slp_day:
                # Event/censor before patient reaches SLP date — single unexposed segment
                records.append({
                    'id': row['DSYSRTKY'], 'start': 0.0,
                    'stop': max(end_time, 0.5),
                    'trt': 0, 'event': final_event, **base
                })
            else:
                # Segment 1: pre-SLP [0, slp_day)
                records.append({
                    'id': row['DSYSRTKY'], 'start': 0.0,
                    'stop': slp_day,
                    'trt': 0, 'event': 0, **base
                })
                # Segment 2: post-SLP [slp_day, end_time]
                records.append({
                    'id': row['DSYSRTKY'], 'start': slp_day,
                    'stop': max(end_time, slp_day + 0.5),
                    'trt': group_flag, 'event': final_event, **base
                })
        return pd.DataFrame(records)

    for label_cox, event_col, competing_col in cox_tv_outcomes:
        print(f"\n  {label_cox}")
        try:
            tv_df = build_tv_df(df, event_col, competing_col)
            tv_df = tv_df.dropna(subset=TV_COVARIATES)
            ctv = CoxTimeVaryingFitter()
            ctv.fit(tv_df, id_col='id', start_col='start', stop_col='stop',
                    event_col='event', show_progress=False)
            row  = ctv.summary.loc['trt']
            hr   = np.exp(row['coef'])
            lo95 = np.exp(row['coef lower 95%'])
            hi95 = np.exp(row['coef upper 95%'])
            p    = row['p']
            n_pts  = tv_df['id'].nunique()
            n_evts = int(tv_df['event'].sum())
            print(f"    HR={hr:.2f}  95%CI [{lo95:.2f}, {hi95:.2f}]  p={p:.4f}"
                  f"  (n={n_pts:,}  events={n_evts:,})")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── KM ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Kaplan-Meier / Cumulative Incidence — Comparison {comp_label}")
    print(f"{'='*80}")

    km_outcomes = [
        ('All-cause mortality',  'died_365d',       'censor_days',         True),
        ('Aspiration PNA',       'asp_365d',         'days_to_aspiration',  False),
        ('Dysphagia Dx',         'dysphagia_365d',   'days_to_dysphagia',   False),
        ('G-tube',               'gtube_365d',       'days_to_gtube',       False),
    ]

    for label_km, event_col, dur_col, show_surv in km_outcomes:
        sub = df.copy()
        sub['_dur'] = sub[dur_col].fillna(sub['censor_days']).astype(float).clip(lower=0.5)
        sub['_ev']  = sub[event_col].astype(int)

        grp1 = sub[sub['treated'] == 1]
        grp0 = sub[sub['treated'] == 0]

        kmf1 = KaplanMeierFitter()
        kmf0 = KaplanMeierFitter()
        kmf1.fit(grp1['_dur'], grp1['_ev'], label=treat_grp)
        kmf0.fit(grp0['_dur'], grp0['_ev'], label=ctrl_grp)

        lr_res = logrank_test(grp1['_dur'], grp0['_dur'],
                              event_observed_A=grp1['_ev'],
                              event_observed_B=grp0['_ev'])
        p_str = "<0.0001" if lr_res.p_value < 0.0001 else f"{lr_res.p_value:.4f}"

        metric = "Survival %" if show_surv else "Cum. incidence %"
        print(f"\n  {label_km}  [{metric}]  Log-rank p={p_str}")
        print(f"  {'Time':<8}  {treat_grp:>20}  {ctrl_grp:>20}")
        print(f"  {'-'*52}")
        for t in TIMEPOINTS:
            s1, lo1, hi1 = km_at(kmf1, t)
            s0, lo0, hi0 = km_at(kmf0, t)
            if show_surv:
                v1 = f"{100*s1:.1f} [{100*lo1:.1f},{100*hi1:.1f}]"
                v0 = f"{100*s0:.1f} [{100*lo0:.1f},{100*hi0:.1f}]"
            else:
                v1 = f"{100*(1-s1):.1f} [{100*(1-hi1):.1f},{100*(1-lo1):.1f}]"
                v0 = f"{100*(1-s0):.1f} [{100*(1-hi0):.1f},{100*(1-lo0):.1f}]"
            print(f"  {t}d{'':<5}  {v1:>20}  {v0:>20}")

    # ── Cost ────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Medicare Cost (365d) — Comparison {comp_label}")
    print(f"{'='*80}")
    print(f"  {'Group':<14}  {'N':>7}  {'Mean $':>11}  {'Median $':>10}  {'MW p':>8}")
    print(f"  {'-'*56}")

    c1 = df[df['treated'] == 1]['total_pmt_365d'].dropna()
    c0 = df[df['treated'] == 0]['total_pmt_365d'].dropna()
    if len(c1) >= 10 and len(c0) >= 10:
        _, mw_p = mannwhitneyu(c1, c0, alternative='two-sided')
        p_str = "<0.0001" if mw_p < 0.0001 else f"{mw_p:.4f}"
    else:
        p_str = "N/A"
    print(f"  {treat_grp:<14}  {len(c1):>7,}  {c1.mean():>11,.0f}  {c1.median():>10,.0f}  {p_str:>8}")
    print(f"  {ctrl_grp:<14}  {len(c0):>7,}  {c0.mean():>11,.0f}  {c0.median():>10,.0f}  {'':>8}")

    for stype in ['Ischemic', 'ICH', 'SAH']:
        sub = df[df['stroke_type'] == stype]
        s   = sub[sub['treated'] == 1]['total_pmt_365d'].dropna()
        n   = sub[sub['treated'] == 0]['total_pmt_365d'].dropna()
        if len(s) < 10 or len(n) < 10:
            continue
        _, p = mannwhitneyu(s, n, alternative='two-sided')
        print(f"  {stype:<14}  treat: ${s.mean():,.0f}  ctrl: ${n.mean():,.0f}  "
              f"p={p:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {DB_PATH} ...")
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("SET memory_limit='24GB'; SET threads=12;")

    for comp_label, treat_grp, ctrl_grp, match_col in COMPARISONS:
        df = load_cohort(con, match_col)
        df, q1, q2 = add_derived(df, treat_grp)
        print(f"\nElix tertiles ({comp_label}): Low<={q1:.0f}  Mid {q1:.0f}-{q2:.0f}  High>{q2:.0f}")
        run_analysis(df, comp_label, treat_grp, ctrl_grp)

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
