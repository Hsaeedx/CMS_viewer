"""
comp_b_stratified.py

Subgroup analysis: Early SLP (days 8-35) vs Late SLP (days 36-90, reference).
Uses the PRIMARY PSM-matched cohort (psm_matched_A = TRUE in stroke_propensity).
PSM is NOT re-run — matched pairs from the primary analysis are reused for consistency.

Strata: Overall | Age (<70/70-79/>=80) | Sex | Stroke type | VWS tier |
        Discharge type | Atrial fibrillation | Prior stroke

Outputs:
  Supp_Figure3.png  — 3-outcome subgroup forest plot
  Supp_Table3.xlsx  — Full HR table
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

DB_PATH       = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
MAX_FOLLOW    = 365
TV_COVARIATES = ['age_at_adm', 'van_walraven_score', 'index_los']
TREAT_GRP     = 'Early'
CTRL_GRP      = 'Late'

OUT_DIR = Path(__file__).parent / "output_files"
OUT_DIR.mkdir(exist_ok=True)

# OSU brand colors
SCARLET     = '#ba0c2f'
GRAY        = '#a7b1b7'
GRAY_DARK   = '#6b7880'
SCARLET_D40 = '#70071c'
SCARLET_D60 = '#4a0513'

OUTCOMES = [
    ('Aspiration-related PNA', 'asp', 'days_to_asp_related', 'days_to_death', False),
    ('PEG/G-tube',     'gtube', 'days_to_gtube',      'days_to_death', True),
    ('Mortality',      'mort',  'days_to_death',       None,            False),
]

# ── Forest layout definition ───────────────────────────────────────────────────
# ('data', key, display_label) | ('header', None, section_title) | ('spacer', None, None)
LAYOUT = [
    ('data',   'Overall',         'Overall'),
    ('spacer',  None,              None),
    ('header',  None,              'Age'),
    ('data',   'age_lt70',        '   <70'),
    ('data',   'age_70to79',      '   70\u201379'),
    ('data',   'age_ge80',        '   \u226580'),
    ('spacer',  None,              None),
    ('header',  None,              'Sex'),
    ('data',   'sex_Male',        '   Male'),
    ('data',   'sex_Female',      '   Female'),
    ('spacer',  None,              None),
    ('header',  None,              'Stroke Type'),
    ('data',   'stroke_Ischemic', '   Ischemic'),
    ('data',   'stroke_Hemorr',   '   Hemorrhagic'),
    ('spacer',  None,              None),
    ('header',  None,              'Comorbidity (VWS)'),
    ('data',   'vws_le0',         '   \u22640'),
    ('data',   'vws_1to9',        '   1\u20139'),
    ('data',   'vws_ge10',        '   \u226510'),
    ('spacer',  None,              None),
    ('header',  None,              'Discharge Type'),
    ('data',   'dschg_Home',      '   Home'),
    ('data',   'dschg_HHA',       '   Home + HHA'),
    ('spacer',  None,              None),
    ('header',  None,              'Atrial Fibrillation'),
    ('data',   'afib_Yes',        '   Yes'),
    ('data',   'afib_No',         '   No'),
    ('spacer',  None,              None),
    ('header',  None,              'Prior Stroke'),
    ('data',   'prior_Yes',       '   Yes'),
    ('data',   'prior_No',        '   No'),
]

STEP = 0.78   # vertical unit per data or header row
SEP  = 0.38   # vertical unit per spacer

# Compute y positions for every LAYOUT item (data AND header)
y_item = []   # list of (type, key, label, y)
_y = 22.0
for typ, key, label in LAYOUT:
    if typ == 'spacer':
        _y -= SEP
    elif typ == 'header':
        y_item.append((typ, key, label, _y))
        _y -= STEP
    else:
        y_item.append((typ, key, label, _y))
        _y -= STEP

y_pos     = {item[1]: item[3] for item in y_item if item[0] == 'data'}
y_headers = [(item[3], item[2]) for item in y_item if item[0] == 'header']
YLIM      = (_y - 0.3, 23.0)

# ── Stratum filter functions ───────────────────────────────────────────────────
STRATA_FN = {
    'Overall':         lambda df: df,
    'age_lt70':        lambda df: df[df['age_at_adm'] < 70],
    'age_70to79':      lambda df: df[(df['age_at_adm'] >= 70) & (df['age_at_adm'] < 80)],
    'age_ge80':        lambda df: df[df['age_at_adm'] >= 80],
    'sex_Male':        lambda df: df[df['sex'] == 'Male'],
    'sex_Female':      lambda df: df[df['sex'] == 'Female'],
    'stroke_Ischemic': lambda df: df[df['stroke_type'] == 'Ischemic'],
    'stroke_Hemorr':   lambda df: df[df['stroke_type'].isin(['ICH', 'SAH'])],
    'vws_le0':         lambda df: df[df['van_walraven_score'] <= 0],
    'vws_1to9':        lambda df: df[(df['van_walraven_score'] >= 1) & (df['van_walraven_score'] <= 9)],
    'vws_ge10':        lambda df: df[df['van_walraven_score'] >= 10],
    'dschg_Home':      lambda df: df[df['dschg_group'] == 'Home'],
    'dschg_HHA':       lambda df: df[df['dschg_group'] == 'Home+HHA'],
    'afib_Yes':        lambda df: df[df['afib'] == 1],
    'afib_No':         lambda df: df[df['afib'] == 0],
    'prior_Yes':       lambda df: df[df['prior_stroke'] == 1],
    'prior_No':        lambda df: df[df['prior_stroke'] == 0],
}

# ── Load primary matched cohort ────────────────────────────────────────────────
print("Loading primary matched cohort (psm_matched_A = TRUE)...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_prop = con.execute("""
    SELECT
        DSYSRTKY, slp_timing_group, days_to_slp_outpt,
        age_at_adm, index_los, van_walraven_score,
        sex, stroke_type, dschg_group,
        afib, prior_stroke, peg_placed
    FROM stroke_propensity
    WHERE psm_matched_A = TRUE
      AND slp_timing_group IN ('Early', 'Late')
""").df()
df_out = con.execute("""
    SELECT DSYSRTKY, days_to_death, days_to_aspiration, days_to_gtube,
           days_to_pneumonia, first_pneumonia_code, pre_stroke_tube
    FROM stroke_outcomes
""").df()
con.close()

df_all = df_prop.merge(df_out, on='DSYSRTKY', how='inner')
df_all['days_to_asp_related'] = np.where(
    df_all['first_pneumonia_code'].isin(['J18', 'J69']),
    df_all['days_to_pneumonia'], np.nan
)
n_early = (df_all['slp_timing_group'] == 'Early').sum()
n_late  = (df_all['slp_timing_group'] == 'Late').sum()
print(f"  Matched cohort: {len(df_all):,} total  |  Early={n_early:,}  Late={n_late:,}")

for col in TV_COVARIATES:
    df_all[col] = df_all[col].fillna(df_all[col].median())


# ── Vectorized TV dataset builder ──────────────────────────────────────────────
def build_tv_df(df, event_col, competing_col):
    """Build person-split time-varying dataset (vectorized)."""
    slp_day  = df['days_to_slp_outpt'].values.astype(float)
    ev_raw   = np.array(pd.to_numeric(df[event_col],   errors='coerce'), dtype=float)
    if competing_col:
        cp_raw = np.array(pd.to_numeric(df[competing_col], errors='coerce'), dtype=float)
    else:
        cp_raw = np.full(len(df), np.nan)

    ev_valid = ~np.isnan(ev_raw)
    cp_valid = ~np.isnan(cp_raw)

    end_time = np.full(len(df), float(MAX_FOLLOW))
    end_time = np.where(ev_valid, np.minimum(end_time, ev_raw), end_time)
    end_time = np.where(cp_valid, np.minimum(end_time, cp_raw), end_time)

    final_ev = (ev_valid & (ev_raw <= MAX_FOLLOW) & (ev_raw == end_time)).astype(int)
    grp_flag = (df['slp_timing_group'].values == TREAT_GRP).astype(int)
    ids      = df['DSYSRTKY'].values

    cov = {c: df[c].values.astype(float) for c in TV_COVARIATES}

    reaches_slp = end_time > slp_day  # patient survives past their SLP day
    has_pre_seg = reaches_slp & (slp_day > 0)

    segments = []

    # Segment A: censored/event before SLP day
    mA = ~reaches_slp
    if mA.any():
        segments.append(pd.DataFrame({
            'id':    ids[mA], 'start': 0.0,
            'stop':  np.maximum(end_time[mA], 0.5),
            'trt':   0, 'event': final_ev[mA],
            **{c: cov[c][mA] for c in TV_COVARIATES}
        }))

    # Segment B: pre-SLP interval [0, slp_day)
    mB = has_pre_seg
    if mB.any():
        segments.append(pd.DataFrame({
            'id':    ids[mB], 'start': 0.0,
            'stop':  slp_day[mB],
            'trt':   0, 'event': 0,
            **{c: cov[c][mB] for c in TV_COVARIATES}
        }))

    # Segment C: post-SLP interval [slp_day, end_time]
    mC = reaches_slp
    if mC.any():
        segments.append(pd.DataFrame({
            'id':    ids[mC], 'start': slp_day[mC],
            'stop':  np.maximum(end_time[mC], slp_day[mC] + 0.5),
            'trt':   grp_flag[mC], 'event': final_ev[mC],
            **{c: cov[c][mC] for c in TV_COVARIATES}
        }))

    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()


def run_tv_cox(df, event_col, competing_col, min_events=10):
    """Fit TV Cox and return (hr, lo, hi, p), n_pts, n_evts. Returns None result on failure."""
    tv      = build_tv_df(df, event_col, competing_col)
    n_pts   = tv['id'].nunique()
    n_evts  = int(tv['event'].sum())
    if n_evts < min_events:
        return None, n_pts, n_evts
    # Standardize covariates
    for col in TV_COVARIATES:
        sd = tv[col].std()
        if sd > 0:
            tv[col] = (tv[col] - tv[col].mean()) / sd
    try:
        ctv = CoxTimeVaryingFitter()
        ctv.fit(tv, id_col='id', start_col='start', stop_col='stop',
                event_col='event', show_progress=False)
        r  = ctv.summary.loc['trt']
        hr = np.exp(r['coef'])
        lo = np.exp(r['coef lower 95%'])
        hi = np.exp(r['coef upper 95%'])
        p  = float(r['p'])
        return (hr, lo, hi, p), n_pts, n_evts
    except Exception as e:
        print(f"    Cox failed: {e}")
        return None, n_pts, n_evts


# ── Run all strata × outcomes ──────────────────────────────────────────────────
print("\nRunning TV Cox for all strata...")
results = {}    # (key, out_key) -> (res_or_None, n_pts, n_evts)
n_stratum = {}  # key -> n (total matched, before peg exclusion)

for key in y_pos:
    fn  = STRATA_FN[key]
    sub = fn(df_all)
    n_stratum[key] = len(sub)
    for out_label, out_key, ev_col, comp_col, excl_peg in OUTCOMES:
        df_sub = sub.copy()
        if excl_peg:
            df_sub = df_sub[
                (df_sub['peg_placed'].fillna(0) == 0) &
                (df_sub['pre_stroke_tube'].fillna(0) == 0)
            ]
        res, n, n_ev = run_tv_cox(df_sub, ev_col, comp_col)
        results[(key, out_key)] = (res, n, n_ev)
        if res:
            hr, lo, hi, p = res
            print(f"  [{key:20s}] {out_label:15s}: n={n:6,}  ev={n_ev:5}  "
                  f"HR={hr:.2f} [{lo:.2f}-{hi:.2f}]  p={p:.4f}")
        else:
            print(f"  [{key:20s}] {out_label:15s}: n={n:6,}  ev={n_ev:5}  (insufficient events)")


# ── Excel output ───────────────────────────────────────────────────────────────
rows_xl = []
for typ, key, label_txt in LAYOUT:
    if typ != 'data':
        continue
    for out_label, out_key, *_ in OUTCOMES:
        res, n, n_ev = results[(key, out_key)]
        row = {
            'Subgroup': label_txt.strip(),
            'Outcome':  out_label,
            'N':        n,
            'Events':   n_ev,
            'Event_pct': round(100 * n_ev / n, 1) if n > 0 else None,
        }
        if res:
            hr, lo, hi, p = res
            row.update({
                'HR':      round(hr, 2),
                'CI_lo':   round(lo, 2),
                'CI_hi':   round(hi, 2),
                'P':       round(p, 4),
                'HR_fmt':  f"{hr:.2f} [{lo:.2f}\u2013{hi:.2f}]",
                'Sig':     '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '',
            })
        else:
            row.update({'HR': None, 'CI_lo': None, 'CI_hi': None, 'P': None,
                        'HR_fmt': '\u2014', 'Sig': ''})
        rows_xl.append(row)

xl_path = OUT_DIR / 'Supp_Table3.xlsx'
pd.DataFrame(rows_xl).to_excel(xl_path, index=False)
print(f"\nSaved {xl_path}")


# ── Forest plot ────────────────────────────────────────────────────────────────
def fmt_p(p):
    if p < 0.001: return 'p<0.001'
    if p < 0.01:  return f'p={p:.3f}'
    return f'p={p:.2f}'

def fmt_ci(hr, lo, hi):
    return f'{hr:.2f} [{lo:.2f}\u2013{hi:.2f}]'


fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('white')

# 7-column gridspec: labels | asp_forest | asp_text | gtube_forest | gtube_text | mort_forest | mort_text
gs = gridspec.GridSpec(
    1, 7,
    figure=fig,
    width_ratios=[3.5, 1.9, 2.4, 1.9, 2.4, 1.9, 2.4],
    wspace=0.0,
    left=0.01, right=0.99, top=0.92, bottom=0.06,
)

ax_lbl = fig.add_subplot(gs[0, 0])
axes_f = [fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[0, 3]),
          fig.add_subplot(gs[0, 5])]
axes_t = [fig.add_subplot(gs[0, 2]),
          fig.add_subplot(gs[0, 4]),
          fig.add_subplot(gs[0, 6])]

XLIM_F = (0.20, 2.80)    # forest x limits (linear; log scale applied below)
X_TICKS = [0.25, 0.5, 1.0, 1.5, 2.0]
OUT_TITLES = ['Aspiration Pneumonia', 'PEG/G-tube Placement', 'All-cause Mortality']

# ── Background section shading ─────────────────────────────────────────────────
# Build sections: list of (y_top, y_bot) for each non-spacer block
section_blocks = []
in_block = False
block_top = None
prev_was_data_or_hdr = False
_y2 = 22.0 + STEP / 2
shade_alt = False

# Identify the y-ranges for each section (header + its data rows)
# by traversing the computed y_item list
sec_ranges = []
cur_sec_top = None
cur_sec_bot = None
for typ, key, label, y in y_item:
    hw = STEP / 2
    if typ == 'header':
        if cur_sec_top is not None:
            sec_ranges.append((cur_sec_top, cur_sec_bot, shade_alt))
            shade_alt = not shade_alt
        cur_sec_top = y + hw
        cur_sec_bot = y - hw
    elif typ == 'data':
        if cur_sec_top is None:  # Overall (before any header)
            if not sec_ranges:
                cur_sec_top = y + hw
                cur_sec_bot = y - hw
            else:
                cur_sec_bot = y - hw
        else:
            cur_sec_bot = y - hw

if cur_sec_top is not None:
    sec_ranges.append((cur_sec_top, cur_sec_bot, shade_alt))

# Also add Overall as its own block (first data row, before any header)
# Already captured above if cur_sec_top initialized on first data

# ── Configure each panel ───────────────────────────────────────────────────────
for idx, (ax_f, ax_t) in enumerate(zip(axes_f, axes_t)):
    out_label, out_key, ev_col, comp_col, excl_peg = OUTCOMES[idx]

    # Forest axes setup
    ax_f.set_xscale('log')
    ax_f.set_xlim(XLIM_F)
    ax_f.set_ylim(YLIM)
    ax_f.set_xticks(X_TICKS)
    ax_f.set_xticklabels([str(x) for x in X_TICKS], fontsize=7.5)
    ax_f.tick_params(axis='x', which='minor', bottom=False)
    ax_f.set_yticks([])
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.spines['left'].set_visible(False)
    ax_f.axvline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, zorder=1)

    # Section shading on forest axes
    for (yt, yb, shade) in sec_ranges:
        if shade:
            ax_f.axhspan(yb, yt, color='#f4f4f4', zorder=0)

    # Title
    ax_f.set_title(OUT_TITLES[idx], fontsize=9, fontweight='bold', pad=5)

    # Text axes setup
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(YLIM)
    ax_t.axis('off')
    # Column header in text panel
    ax_t.text(0.05, YLIM[1] - 0.30, 'HR [95% CI]',
              fontsize=7.5, fontweight='bold', va='center', ha='left')
    ax_t.text(0.80, YLIM[1] - 0.30, 'p',
              fontsize=7.5, fontweight='bold', va='center', ha='right')

    # Section shading on text axes
    for (yt, yb, shade) in sec_ranges:
        if shade:
            ax_t.axhspan(yb, yt, color='#f4f4f4', zorder=0)

    # ── Draw per-row results ────────────────────────────────────────────────────
    for typ2, key2, lbl2, y2 in y_item:
        if typ2 != 'data':
            continue
        res, n, n_ev = results[(key2, out_key)]
        is_overall = (key2 == 'Overall')

        if res is not None:
            hr, lo, hi, p = res
            sig = p < 0.05
            color = SCARLET if sig else GRAY_DARK
            ms    = 9 if is_overall else 6
            mk    = 'D' if is_overall else 'o'
            lw    = 1.5 if is_overall else 1.1

            lo_p = max(lo, XLIM_F[0] * 1.02)
            hi_p = min(hi, XLIM_F[1] * 0.98)
            hr_p = max(min(hr, XLIM_F[1] * 0.98), XLIM_F[0] * 1.02)

            # CI line
            ax_f.plot([lo_p, hi_p], [y2, y2],
                      color=color, lw=lw, solid_capstyle='round', zorder=3)
            # Arrow caps if CI extends beyond xlim
            if lo < XLIM_F[0]:
                ax_f.annotate('', xy=(XLIM_F[0] * 1.04, y2),
                               xytext=(XLIM_F[0] * 1.10, y2),
                               arrowprops=dict(arrowstyle='<-', color=color, lw=1))
            if hi > XLIM_F[1]:
                ax_f.annotate('', xy=(XLIM_F[1] * 0.96, y2),
                               xytext=(XLIM_F[1] * 0.90, y2),
                               arrowprops=dict(arrowstyle='<-', color=color, lw=1))
            # Central point
            ax_f.plot(hr_p, y2, marker=mk, color=color, ms=ms, zorder=5,
                      markeredgewidth=0.5, markeredgecolor='white')

            # Text
            ci_str = fmt_ci(hr, lo, hi)
            p_str  = fmt_p(p)
            star   = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            fw = 'bold' if is_overall else 'normal'
            ax_t.text(0.05, y2 + 0.14, ci_str,
                      fontsize=6.8, va='center', ha='left',
                      fontweight=fw, color=SCARLET if sig else 'black')
            ax_t.text(0.80, y2 - 0.14, f'{p_str}{star}',
                      fontsize=6.5, va='center', ha='right',
                      color=SCARLET if sig else '#888888')
        else:
            ev_note = f'({n_ev} events)' if n_ev < 10 else 'failed'
            ax_t.text(0.05, y2, f'\u2014 {ev_note}',
                      fontsize=6.8, va='center', ha='left', color='#aaaaaa')

    # Reference label at x=1.0
    ax_f.text(1.0, YLIM[0] + 0.05, 'Late\n(ref)',
              fontsize=6.5, ha='center', va='bottom', color='#666666')


# ── Label axis ─────────────────────────────────────────────────────────────────
ax_lbl.set_xlim(0, 1)
ax_lbl.set_ylim(YLIM)
ax_lbl.axis('off')

# Column header
ax_lbl.text(0.01, YLIM[1] - 0.30, 'Subgroup',
            fontsize=7.5, fontweight='bold', va='center', ha='left')
ax_lbl.text(0.88, YLIM[1] - 0.30, 'N (Early/Late)',
            fontsize=7.5, fontweight='bold', va='center', ha='right')

# Section shading on label axis
for (yt, yb, shade) in sec_ranges:
    if shade:
        ax_lbl.axhspan(yb, yt, color='#f4f4f4', zorder=0)

# Header labels
for hy, hlabel in y_headers:
    ax_lbl.text(0.01, hy, hlabel,
                fontsize=8.5, fontweight='bold', va='center', ha='left',
                color='black')
    ax_lbl.plot([0.01, 0.88], [hy - 0.22, hy - 0.22],
                color='#cccccc', lw=0.6,
                transform=ax_lbl.get_yaxis_transform())

# Data row labels
for typ2, key2, lbl2, y2 in y_item:
    if typ2 != 'data':
        continue
    is_overall = (key2 == 'Overall')
    n_tot = n_stratum[key2]
    n_e   = len(STRATA_FN[key2](df_all)[df_all['slp_timing_group'] == 'Early'])
    n_l   = len(STRATA_FN[key2](df_all)[df_all['slp_timing_group'] == 'Late'])

    fw = 'bold' if is_overall else 'normal'
    ax_lbl.text(0.01, y2, lbl2 if is_overall else lbl2,
                fontsize=8, va='center', ha='left', fontweight=fw, color='black')
    ax_lbl.text(0.88, y2, f'{n_e:,}\u00a0/\u00a0{n_l:,}',
                fontsize=7, va='center', ha='right', color='#555555')

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SCARLET, markersize=7,
           label='Significant (p<0.05): Early SLP favored' if True else ''),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY_DARK, markersize=7,
           label='Not significant'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8,
           label='Overall estimate'),
    Line2D([0], [0], color='black', lw=0.8, ls='--', label='HR = 1.0 (no difference)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.00), frameon=False)

# ── Figure titles ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.965,
         'Subgroup Analysis: Early SLP (Days 8\u201335) vs Late SLP (Days 36\u201390)',
         ha='center', va='center', fontsize=11, fontweight='bold')
fig.text(0.50, 0.945,
         f'Primary PSM-matched cohort (n\u2009=\u2009{n_early + n_late:,}; {n_early:,} Early, {n_late:,} Late). '
         'Time-varying Cox adjusted for age, van Walraven score, LOS. Reference: Late SLP.',
         ha='center', va='center', fontsize=8.5, color='#444444')

fig_path = OUT_DIR / 'Supp_Figure3.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {fig_path}")
print("\nDone.")
