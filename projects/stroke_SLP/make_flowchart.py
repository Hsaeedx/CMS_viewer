"""
make_flowchart.py
Comprehensive cohort derivation CONSORT flowchart.
Vertical (portrait, top-to-bottom) layout with 7 main boxes and 6 exclusion branches.
Upstream counts (from inp_claimsk_all) are hardcoded after a one-time query run;
downstream counts are queried live from the stored analytic tables.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_out_dir = Path(os.getenv("project_paths", ".")) / "stroke_SLP" / "output_files"
_out_dir.mkdir(parents=True, exist_ok=True)
OUT_PATH = _out_dir / "fig1_cohort.png"
DB_PATH  = Path(os.getenv("duckdb_database", "cms_data.duckdb"))

# ── Upstream counts (queried 2026-04-11 from inp_claimsk_all) ─────────────────
# Source query: C:\temp\flowchart_upstream_counts.sql
N_TOTAL_ADM      = 1_997_741   # all acute stroke admission records (I60/I61/I63/I64)
N_UNIQUE_PTS     = 1_737_292   # unique beneficiaries (first stroke per patient)
N_IN_HOSP_DEATH  =   118_881   # died during index hospitalization (STUS_CD='20')
N_HOSPICE        =   112_952   # discharged to hospice (STUS_CD='50'/'51')
N_SNF            =   354_112   # discharged to SNF (STUS_CD='03'/'61'/'64')
N_IRF            =   363_125   # discharged to IRF (STUS_CD='62')
N_LTACH          =    19_173   # discharged to LTACH (STUS_CD='63')
N_HOME_ELIGIBLE  =   709_138   # home discharge (STUS_CD='01'/'06'/'07')

# ── Live queries (fast — against stored analytic tables) ─────────────────────
print("Querying downstream cohort counts...")
con = duckdb.connect(str(DB_PATH), read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")

row = con.execute("""
    SELECT
        (SELECT COUNT(*) FROM stroke_cohort)                                          AS n_stroke_cohort,
        (SELECT COUNT(*) FROM stroke_slp WHERE slp_outpt_any_90d = 1)               AS n_any_slp,
        (SELECT COUNT(*) FROM stroke_slp WHERE first_slp_is_clinic = TRUE)           AS n_clinic_first,
        (SELECT COUNT(*) FROM stroke_slp s
            JOIN stroke_cohort c ON c.DSYSRTKY = s.DSYSRTKY
            WHERE s.first_slp_is_clinic = TRUE
              AND s.days_to_slp_outpt = 0)                                           AS n_day0_slp,
        (SELECT COUNT(*) FROM stroke_slp s
            JOIN stroke_cohort c ON c.DSYSRTKY = s.DSYSRTKY
            WHERE s.first_slp_is_clinic = TRUE
              AND s.days_to_slp_outpt BETWEEN 1 AND 90
              AND (COALESCE(c.dysphagia_poa,0)=1 OR COALESCE(c.aspiration_poa,0)=1
                   OR COALESCE(c.peg_placed,0)=1  OR COALESCE(c.trach_placed,0)=1)) AS n_excl_severity,
        (SELECT COUNT(*) FROM stroke_propensity WHERE slp_timing_group = 'Wk0')     AS n_wk0,
        (SELECT COUNT(*) FROM stroke_propensity WHERE slp_timing_group = 'Early')    AS n_early,
        (SELECT COUNT(*) FROM stroke_propensity WHERE slp_timing_group = 'Late')     AS n_late
""").fetchone()
con.close()

n_stroke_cohort, n_any_slp, n_clinic_first, \
    n_day0_slp, n_excl_severity, n_wk0, n_early, n_late = row

# ── Derived exclusion counts ──────────────────────────────────────────────────
n_repeat         = N_TOTAL_ADM - N_UNIQUE_PTS
n_snf_irf_ltach  = N_SNF + N_IRF + N_LTACH
n_non_home_tot   = N_UNIQUE_PTS - N_HOME_ELIGIBLE
n_other_dschg    = n_non_home_tot - N_IN_HOSP_DEATH - N_HOSPICE - n_snf_irf_ltach
n_excl_age_ffs   = N_HOME_ELIGIBLE - n_stroke_cohort   # age<65, no FFS, 30d death
n_excl_no_slp    = n_stroke_cohort - n_any_slp
n_excl_hha       = n_any_slp - n_clinic_first
n_excl_from_clinic = n_day0_slp + n_excl_severity + n_wk0
n_analytic       = n_early + n_late

fmt = lambda n: f"{n:,}"

print(f"  Total admissions:     {fmt(N_TOTAL_ADM)}")
print(f"  Unique patients:      {fmt(N_UNIQUE_PTS)}  (repeat excl: {fmt(n_repeat)})")
print(f"  Home-eligible:        {fmt(N_HOME_ELIGIBLE)}  (non-home excl: {fmt(n_non_home_tot)})")
print(f"  Stroke cohort:        {fmt(n_stroke_cohort)}  (age/FFS excl: {fmt(n_excl_age_ffs)})")
print(f"  Any SLP 90d:          {fmt(n_any_slp)}  (no SLP: {fmt(n_excl_no_slp)})")
print(f"  Clinic-first:         {fmt(n_clinic_first)}  (HHA-first: {fmt(n_excl_hha)})")
print(f"  Excl from clinic:     {fmt(n_excl_from_clinic)}  "
      f"(day0={fmt(n_day0_slp)}, severity={fmt(n_excl_severity)}, wk0={fmt(n_wk0)})")
print(f"  Analytic:             {fmt(n_analytic)}  Early={fmt(n_early)}  Late={fmt(n_late)}")

# ── Layout constants ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 23))
ax.set_xlim(0, 10)
ax.set_ylim(0, 22.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# Main column
BOX_CX = 4.0
BOX_W  = 4.2
BOX_H  = 1.0

# Exclusion boxes (right side)
EXCL_CX = 8.2
EXCL_W  = 3.4

# Split boxes (bottom)
SPLIT_W  = 3.2
SPLIT_H  = 1.1
SPLIT_CY = 1.2

# 7 main box y-centers (top → bottom)
BOX_CY = [21.0, 18.0, 15.0, 12.0, 9.0, 6.0, 3.2]

# Midpoints between consecutive boxes (exclusion branch y-positions)
MID_Y = [(BOX_CY[i] + BOX_CY[i + 1]) / 2 for i in range(len(BOX_CY) - 1)]

# Early/Late split box x-centers
SPLIT_CX = [2.0, 7.2]

ARROW_COLOR  = '#a7b1b7'
BOX_COLOR    = '#ba0c2f'
EXCL_COLOR   = '#70071c'
SPLIT_COLORS = ['#ba0c2f', '#70071c']


# ── Helper functions ──────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle, color=BOX_COLOR):
    rect = FancyBboxPatch((cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.15, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=4, multialignment='center', linespacing=1.3)
    ax.text(cx, cy - 0.29, subtitle,
            ha='center', va='center', fontsize=9.5, color='#f5d0d8', zorder=4)


def excl_box(cx, cy, text, h=0.85):
    rect = FancyBboxPatch((cx - EXCL_W / 2, cy - h / 2), EXCL_W, h,
                          boxstyle="round,pad=0.05",
                          facecolor='#f2f0f0', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=7.8, color='#4a0513',
            zorder=4, linespacing=1.35, multialignment='center')


def split_box(cx, cy, title, n, color):
    rect = FancyBboxPatch((cx - SPLIT_W / 2, cy - SPLIT_H / 2), SPLIT_W, SPLIT_H,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.20, title,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.22, n,
            ha='center', va='center', fontsize=10, color='white', zorder=4)


# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(BOX_CX, BOX_CY[0],
         "Medicare Acute Stroke Admissions\n(ICD-10 I60/I61/I63/I64; 2016\u20132022)",
         f"N = {fmt(N_TOTAL_ADM)} admissions")

main_box(BOX_CX, BOX_CY[1],
         "Index (First) Stroke Admission\nper Beneficiary",
         f"N = {fmt(N_UNIQUE_PTS)}")

main_box(BOX_CX, BOX_CY[2],
         "Home-Discharged at\nIndex Hospitalization",
         f"N = {fmt(N_HOME_ELIGIBLE)}")

main_box(BOX_CX, BOX_CY[3],
         "Medicare Stroke Cohort\n(Age \u226565, 6-Month FFS Enrolled)",
         f"N = {fmt(n_stroke_cohort)}")

main_box(BOX_CX, BOX_CY[4],
         "Received Outpatient SLP\nWithin 90 Days Post-Discharge",
         f"N = {fmt(n_any_slp)}")

main_box(BOX_CX, BOX_CY[5],
         "Clinic-Based First SLP Visit\n(Not Home Health Agency)",
         f"N = {fmt(n_clinic_first)}")

main_box(BOX_CX, BOX_CY[6],
         "Analytic Cohort\n(Weeks 1\u20134 and Week 5+)",
         f"N = {fmt(n_analytic)}")


# ── Vertical arrows between main boxes ───────────────────────────────────────
for i in range(len(BOX_CY) - 1):
    y_src = BOX_CY[i]     - BOX_H / 2
    y_dst = BOX_CY[i + 1] + BOX_H / 2
    ax.annotate('', xy=(BOX_CX, y_dst + 0.05), xytext=(BOX_CX, y_src - 0.05),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16),
                zorder=2)


# ── Exclusion boxes (right side, horizontal dashed branches) ─────────────────
excl_items = [
    # MID_Y[0]: between Box1 and Box2 — repeat admissions
    (MID_Y[0], 0.90,
     f"Excluded: Repeat stroke admissions\n"
     f"(first per beneficiary only)\n"
     f"n = {fmt(n_repeat)}"),

    # MID_Y[1]: between Box2 and Box3 — non-home discharge
    (MID_Y[1], 1.50,
     f"Excluded: Non-home discharge\n"
     f"  In-hospital death:  n = {fmt(N_IN_HOSP_DEATH)}\n"
     f"  Hospice:            n = {fmt(N_HOSPICE)}\n"
     f"  SNF / IRF / LTACH: n = {fmt(n_snf_irf_ltach)}\n"
     f"  Other discharge:    n = {fmt(n_other_dschg)}\n"
     f"Total: n = {fmt(n_non_home_tot)}"),

    # MID_Y[2]: between Box3 and Box4 — age / FFS / 30d death
    (MID_Y[2], 1.10,
     f"Excluded: Age <65, no 6-month\n"
     f"FFS enrollment, or 30-day\n"
     f"post-discharge mortality\n"
     f"n = {fmt(n_excl_age_ffs)}"),

    # MID_Y[3]: between Box4 and Box5 — no SLP
    (MID_Y[3], 0.90,
     f"Excluded: No outpatient SLP\n"
     f"within 90 days of discharge\n"
     f"n = {fmt(n_excl_no_slp)}"),

    # MID_Y[4]: between Box5 and Box6 — HHA-first
    (MID_Y[4], 0.90,
     f"Excluded: First SLP was\n"
     f"HHA-based (home health)\n"
     f"n = {fmt(n_excl_hha)}"),

    # MID_Y[5]: between Box6 and Box7 — day-0, severity, Week 0
    (MID_Y[5], 1.35,
     f"Excluded:\n"
     f"  SLP on discharge date:  n = {fmt(n_day0_slp)}\n"
     f"  Pre-existing dysphagia\n"
     f"  or index PEG/trach:     n = {fmt(n_excl_severity)}\n"
     f"  Week 0 SLP (days 1\u20137): n = {fmt(n_wk0)}"),
]

main_right = BOX_CX + BOX_W / 2
excl_left  = EXCL_CX - EXCL_W / 2

for mid_y, h, text in excl_items:
    # Dashed horizontal line from main column to exclusion box
    ax.plot([main_right, excl_left - 0.08], [mid_y, mid_y],
            color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
    # Arrowhead into exclusion box
    ax.annotate('', xy=(excl_left, mid_y), xytext=(excl_left - 0.08, mid_y),
                arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                                lw=1.4, mutation_scale=13),
                zorder=2)
    excl_box(EXCL_CX, mid_y, text, h=h)


# ── Split into timing groups ──────────────────────────────────────────────────
labels = ['Early SLP\n(Weeks 1\u20134)\n(Treated)', 'Late SLP\n(Week 5+)\n(Reference)']
counts = [f"n = {fmt(n_early)}", f"n = {fmt(n_late)}"]

box7_bottom = BOX_CY[-1] - BOX_H / 2
branch_y    = SPLIT_CY + SPLIT_H / 2 + 0.55

# Vertical stem from Box7 bottom to T-junction
ax.plot([BOX_CX, BOX_CX], [box7_bottom, branch_y],
        color=ARROW_COLOR, lw=1.8, zorder=2)

# Horizontal T-junction
ax.plot([SPLIT_CX[0], SPLIT_CX[1]], [branch_y, branch_y],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Drop arrows to split boxes
for cx, lab, cnt, col in zip(SPLIT_CX, labels, counts, SPLIT_COLORS):
    box_top = SPLIT_CY + SPLIT_H / 2
    ax.annotate('', xy=(cx, box_top + 0.05), xytext=(cx, branch_y),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    split_box(cx, SPLIT_CY, lab, cnt, col)


# ── Title and footnote ────────────────────────────────────────────────────────
ax.text(5.0, 22.1,
        "Figure 1. Cohort Derivation",
        ha='center', va='center',
        fontsize=14, fontweight='bold', color='#000000')

ax.text(5.0, 0.22,
        "Medicare FFS claims 2016\u20132022  \u2022  "
        "ICD-10-CM principal diagnosis (I60/I61/I63/I64)  \u2022  "
        "SLP timing measured from date of discharge  \u2022  "
        "Week 0 (days 1\u20137) retained for descriptive purposes",
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')


plt.tight_layout(pad=0.3)
plt.savefig(str(OUT_PATH), dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
