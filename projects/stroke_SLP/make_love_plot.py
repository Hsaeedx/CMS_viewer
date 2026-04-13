"""
make_love_plot.py

Covariate balance plot (love plot) showing standardized mean differences (SMD)
before and after propensity score matching.

Comparison: Early SLP (days 8-35) vs Late SLP (days 36-90).
PSM covariates mirror those used in stroke_psm.py.

Output: output_files/Supp_Figure4.png
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DB_PATH = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
OUT_DIR = Path(__file__).parent / "output_files"
OUT_DIR.mkdir(exist_ok=True)

TREAT_GRP = 'Early'
CTRL_GRP  = 'Late'

SCARLET     = '#ba0c2f'
GRAY        = '#a7b1b7'
GRAY_DARK   = '#6b7880'

CONT_VARS   = ['age_at_adm', 'index_los', 'van_walraven_score', 'adm_year']
BINARY_VARS = ['afib', 'hypertension', 'mech_vent', 'prior_stroke', 'dual_eligible']
CAT_VARS    = ['sex', 'race', 'stroke_type', 'drg_group', 'adm_source', 'rucc_group']


def bucket_drg(drg_cd):
    if pd.isna(drg_cd): return 'Other'
    try: n = int(str(drg_cd).strip())
    except ValueError: return 'Other'
    if 61 <= n <= 69: return 'Medical_stroke'
    if 20 <= n <= 38: return 'Neurosurgical'
    if 52 <= n <= 60: return 'Spinal'
    if 70 <= n <= 74: return 'TIA_headache'
    return 'Other'


def smd(x, treated):
    """Standardized mean difference (absolute value)."""
    x1 = x[treated == 1].astype(float)
    x0 = x[treated == 0].astype(float)
    mu1, mu0 = x1.mean(), x0.mean()
    pooled_sd = np.sqrt((x1.std()**2 + x0.std()**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(mu1 - mu0) / pooled_sd


# ── Load data ───────────────────────────────────────────────────────────────────
print("Loading propensity data...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df_raw = con.execute("""
    SELECT
        DSYSRTKY, slp_timing_group, psm_matched_A,
        age_at_adm, index_los, van_walraven_score, adm_year,
        sex, race, stroke_type, DRG_CD AS drg_cd, adm_source,
        afib, hypertension, mech_vent, prior_stroke,
        rucc_group, dual_eligible
    FROM stroke_propensity
    WHERE slp_timing_group IN ('Early', 'Late')
""").df()
con.close()

print(f"  Loaded {len(df_raw):,} rows  "
      f"(Early={( df_raw['slp_timing_group']=='Early').sum():,}  "
      f"Late={(df_raw['slp_timing_group']=='Late').sum():,})")

# Pre-process
for col in CONT_VARS:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(df_raw[col].median())

df_raw['drg_group']    = df_raw['drg_cd'].apply(bucket_drg)
df_raw['adm_source']   = df_raw['adm_source'].fillna('Unknown').astype(str)
df_raw['rucc_group']   = df_raw['rucc_group'].fillna('Unknown').astype(str)
df_raw['dual_eligible'] = df_raw['dual_eligible'].fillna(0).astype(int)
df_raw['treated']    = (df_raw['slp_timing_group'] == TREAT_GRP).astype(int)

# Build feature matrix (same dummies as PSM)
dummies = pd.get_dummies(df_raw[CAT_VARS], drop_first=False)
df_feat = pd.concat([df_raw[CONT_VARS + BINARY_VARS].astype(float),
                     dummies, df_raw['treated']], axis=1)

# Pre-match and post-match subsets
df_pre  = df_feat.copy()
df_post = df_feat[df_raw['psm_matched_A'] == True].copy()

n_pre_e  = (df_raw['slp_timing_group'] == 'Early').sum()
n_pre_l  = (df_raw['slp_timing_group'] == 'Late').sum()
n_post_e = ((df_raw['psm_matched_A'] == True) & (df_raw['slp_timing_group'] == 'Early')).sum()
n_post_l = ((df_raw['psm_matched_A'] == True) & (df_raw['slp_timing_group'] == 'Late')).sum()
print(f"  Pre-match:  Early={n_pre_e:,}  Late={n_pre_l:,}")
print(f"  Post-match: Early={n_post_e:,}  Late={n_post_l:,}")


# ── Select covariates to display ───────────────────────────────────────────────
# Display only the key dummies (not all dummies from get_dummies)
DISPLAY_VARS = [
    # Label                      column name in df_feat
    ('Age at admission',         'age_at_adm'),
    ('LOS (days)',               'index_los'),
    ('VWS score',                'van_walraven_score'),
    ('Admission year',           'adm_year'),
    ('Male sex',                 'sex_Male'),
    ('Race: White',              'race_White'),
    ('Race: Black',              'race_Black'),
    ('Race: Hispanic',           'race_Hispanic'),
    ('Stroke: Ischemic',         'stroke_type_Ischemic'),
    ('Stroke: ICH',              'stroke_type_ICH'),
    ('Stroke: SAH',              'stroke_type_SAH'),
    ('DRG: Medical stroke',      'drg_group_Medical_stroke'),
    ('DRG: Neurosurgical',       'drg_group_Neurosurgical'),
    ('Mechanical ventilation',   'mech_vent'),
    ('Atrial fibrillation',      'afib'),
    ('Hypertension',             'hypertension'),
    ('Prior stroke',             'prior_stroke'),
    ('Metro county',             'rucc_group_Metro'),
    ('Nonmetro county',          'rucc_group_Nonmetro'),
    ('Rural county',             'rucc_group_Rural'),
    ('Dual eligible (Medicare+Medicaid)', 'dual_eligible'),
]

# ── Compute SMDs ───────────────────────────────────────────────────────────────
results = []
for label, col in DISPLAY_VARS:
    if col not in df_pre.columns:
        print(f"  SKIP (not found): {col}")
        continue
    pre_s  = smd(df_pre[col].fillna(0).values,  df_pre['treated'].values)
    post_s = smd(df_post[col].fillna(0).values, df_post['treated'].values)
    results.append({'label': label, 'col': col, 'pre': pre_s, 'post': post_s})
    print(f"  {label:<35}  pre={pre_s:.3f}  post={post_s:.3f}")

df_res = pd.DataFrame(results)

# ── Plot ───────────────────────────────────────────────────────────────────────
n_vars = len(df_res)
fig, ax = plt.subplots(figsize=(8, 0.42 * n_vars + 2.0))
fig.patch.set_facecolor('white')

y_vals = np.arange(n_vars)

# Shade rows alternately
for i in range(n_vars):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color='#f6f6f6', zorder=0)

# Reference lines
ax.axvline(0.10, color='#cccccc', lw=1.0, ls='--', zorder=1, label='SMD = 0.10 threshold')
ax.axvline(0.00, color='black',   lw=0.8, ls='-',  zorder=1)

# Points
pre_col  = GRAY_DARK
post_col = SCARLET

for i, row in df_res.iterrows():
    y = i
    # Pre-match: open circle
    ax.plot(row['pre'],  y, marker='o', color=pre_col,  ms=7,
            markeredgecolor=pre_col, markerfacecolor='white',
            markeredgewidth=1.5, zorder=5)
    # Post-match: filled circle
    ax.plot(row['post'], y, marker='o', color=post_col, ms=7,
            markeredgecolor='white', markeredgewidth=0.5, zorder=5)
    # Connecting line
    ax.plot([row['pre'], row['post']], [y, y],
            color='#cccccc', lw=0.8, zorder=2)

# Axes formatting
ax.set_yticks(y_vals)
ax.set_yticklabels(df_res['label'], fontsize=8.5)
ax.set_xlabel('Standardized Mean Difference (absolute)', fontsize=9)
ax.set_xlim(-0.01, max(df_res['pre'].max(), 0.35) * 1.10)
ax.set_ylim(-0.5, n_vars - 0.5)
ax.invert_yaxis()   # top-to-bottom order
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', left=False)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='white', markeredgecolor=GRAY_DARK,
           markeredgewidth=1.5, markersize=8,
           label=f'Before matching  (Early n={n_pre_e:,}, Late n={n_pre_l:,})'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=SCARLET, markeredgecolor='white',
           markeredgewidth=0.5, markersize=8,
           label=f'After matching  (Early n={n_post_e:,}, Late n={n_post_l:,})'),
    Line2D([0], [0], color='#cccccc', lw=1.0, ls='--',
           label='SMD = 0.10 (imbalance threshold)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
          frameon=True, framealpha=0.9, edgecolor='#cccccc')

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    'Covariate Balance Before and After Propensity Score Matching\n'
    'Early SLP (Days 8\u201335) vs Late SLP (Days 36\u201390)',
    fontsize=10, fontweight='bold', pad=10
)

plt.tight_layout()
fig_path = OUT_DIR / 'Supp_Figure4.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved {fig_path}")
print("Done.")
