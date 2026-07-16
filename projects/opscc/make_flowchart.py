"""
make_flowchart.py
OPSCC cohort flowchart — horizontal layout.
Main flow: left to right  |  Exclusions: drop down  |  Split: 5 groups on right

Right side shows five treatment groups grouped by comparison:
  Comparison A: TORS alone  vs  RT alone
  Comparison B: TORS + RT   vs  CRT
  Comparison C: TORS + CRT  vs  CRT
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
_fig_dir = Path(os.getenv("analysis_directory", r"C:\Users\hsaee\Desktop\CMS_viewer")) / "projects" / "opscc" / "figures"
_fig_dir.mkdir(exist_ok=True)
OUT_PATH = str(_fig_dir / "cohort_flowchart.png")
DB_PATH  = os.getenv("duckdb_database", r"F:\CMS\cms_data.duckdb")

# ── Pull counts dynamically from pipeline tables ───────────────────────────────
import duckdb
con = duckdb.connect(DB_PATH, read_only=True)

def q(sql): return con.execute(sql).fetchone()[0]

n_hnc_raw   = q("SELECT COUNT(DISTINCT DSYSRTKY) FROM hnc_dx_raw")
n_confirmed = q("SELECT COUNT(*) FROM hnc_confirmed")
n_opscc     = q("SELECT COUNT(*) FROM opscc_universe")
n_ffs       = q("SELECT COUNT(*) FROM opscc_cohort")
n_cohort    = q("SELECT COUNT(*) FROM opscc_propensity")

tx_counts = dict(con.execute(
    "SELECT tx_group, COUNT(*) FROM opscc_propensity GROUP BY tx_group"
).fetchall())

n_tors_alone = tx_counts.get('TORS alone', 0)
n_rt_alone   = tx_counts.get('RT alone',   0)
n_tors_rt    = tx_counts.get('TORS + RT',  0)
n_tors_crt   = tx_counts.get('TORS + CRT', 0)
n_ctcrt      = tx_counts.get('CRT',        0)
n_other      = n_cohort - n_tors_alone - n_rt_alone - n_tors_rt - n_tors_crt - n_ctcrt

n_excl_unconfirmed = n_hnc_raw   - n_confirmed
n_excl_nonopscc    = n_confirmed - n_opscc
n_excl_ffs         = n_opscc     - n_ffs
n_excl_mets        = q("""
    SELECT COUNT(*) FROM opscc_cohort
    WHERE DSYSRTKY NOT IN (SELECT DSYSRTKY FROM opscc_propensity)
""")

con.close()

def fmt(n): return f"{n:,}"

fig, ax = plt.subplots(figsize=(38, 15))
ax.set_xlim(0, 28)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── OSU colours ───────────────────────────────────────────────────────────────
SCARLET     = '#BA0C2F'
GRAY        = '#A7B1B7'
DARK40      = '#70071C'
DARK60      = '#4A0513'
EXCL_FG     = '#843C0C'
EXCL_BG     = '#FCE4D6'
EXCL_EDGE   = '#C55A11'

# Treatment group box colours
TORS_COL    = SCARLET    # TORS alone
RT_COL      = '#6B7B7D'  # RT alone   (medium neutral)
TORSR_COL   = DARK40     # TORS + RT
TORSCRT_COL = '#A63044'  # TORS + CRT
CTCRT_COL   = '#5E6E73'  # CRT     (slightly darker neutral)

ARROW_COLOR = DARK60

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W   = 3.5
BOX_H   = 1.35
EXCL_W  = 3.8
EXCL_H  = 1.10
SPLIT_W = 3.9
SPLIT_H = 1.2

CY     = 6.5   # y-centre of main horizontal flow
EXCL_Y = 3.4   # y-centre of main exclusion boxes

CX = [2.4, 6.4, 10.4, 14.4, 18.4]   # x-centres of 5 main flow boxes

# Right-side treatment group y-centres: 5 boxes in column
# Order top → bottom: TORS alone, RT alone, TORS+RT, TORS+CRT, CRT
# Comparisons:
#   Comp A: TORS alone + RT alone
#   Comp B: TORS+RT + CRT
#   Comp C: TORS+CRT + CRT (adjacent boxes — bottom two)
SPLIT_CX  = 24.5
CY_TORS_ALONE = 9.7
CY_RT_ALONE   = 8.4
CY_TORS_RT    = 6.8
CY_TORS_CRT   = 5.5
CY_CRT        = 4.2
ALL_CY = [CY_TORS_ALONE, CY_RT_ALONE, CY_TORS_RT, CY_TORS_CRT, CY_CRT]


# ── Helpers ───────────────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=DARK60, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.20, title,
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.28, subtitle,
            ha='center', va='center', fontsize=21, color='#E0B8C0', zorder=4)


def excl_box(cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor=EXCL_BG, edgecolor=EXCL_EDGE,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=19, color=EXCL_FG,
            zorder=4, linespacing=1.4, multialignment='center')


def split_box(cx, cy, title, subtitle, color):
    rect = FancyBboxPatch((cx - SPLIT_W/2, cy - SPLIT_H/2), SPLIT_W, SPLIT_H,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.22, title,
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='white', zorder=4)
    ax.text(cx, cy - 0.26, subtitle,
            ha='center', va='center', fontsize=21, color='white', zorder=4)


def comp_bracket(y_top, y_bot, label, color):
    """Draw a right-side bracket from y_top to y_bot with a comp label."""
    bx = SPLIT_CX + SPLIT_W/2 + 0.18
    mid_y = (y_top + y_bot) / 2
    # vertical bar
    ax.plot([bx, bx], [y_bot - 0.3, y_top + 0.3],
            color=color, lw=2.0, zorder=4)
    # top tick
    ax.plot([bx, bx + 0.15], [y_top + 0.3, y_top + 0.3],
            color=color, lw=2.0, zorder=4)
    # bottom tick
    ax.plot([bx, bx + 0.15], [y_bot - 0.3, y_bot - 0.3],
            color=color, lw=2.0, zorder=4)
    # label
    ax.text(bx + 0.28, mid_y, label,
            ha='left', va='center', fontsize=19, fontweight='bold',
            color=color, zorder=4, rotation=90, linespacing=1.3)


# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(CX[0], CY, "Medicare HNC Claims\n(C00\u2013C14, C30\u2013C32)", f"N = {fmt(n_hnc_raw)}")
main_box(CX[1], CY, "Confirmed HNC", f"N = {fmt(n_confirmed)}")
main_box(CX[2], CY, "OPSCC Subsite\n(C01/C09/C10/C14)", f"N = {fmt(n_opscc)}")
main_box(CX[3], CY, "Continuous FFS\nEnrollment \u22656 Months", f"N = {fmt(n_ffs)}")
main_box(CX[4], CY, "Non-Metastatic\nOPSCC", f"N = {fmt(n_cohort)}")

# ── Horizontal arrows ─────────────────────────────────────────────────────────
for i in range(4):
    x0 = CX[i]   + BOX_W / 2
    x1 = CX[i+1] - BOX_W / 2
    ax.annotate('', xy=(x1 - 0.05, CY), xytext=(x0 + 0.05, CY),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16), zorder=2)

MID_XS = [(CX[i] + BOX_W/2 + CX[i+1] - BOX_W/2) / 2 for i in range(4)]

# ── Exclusion boxes (main flow) ───────────────────────────────────────────────
excl_texts = [
    f"Excluded: Unconfirmed\ndiagnosis  (n = {fmt(n_excl_unconfirmed)})",
    f"Excluded: Non-OPSCC\nsubsite  (n = {fmt(n_excl_nonopscc)})",
    f"Excluded: Incomplete\nFFS enrollment  (n = {fmt(n_excl_ffs)})",
    f"Excluded: Metastatic\ndisease  (n = {fmt(n_excl_mets)})",
]
for mx, text in zip(MID_XS, excl_texts):
    y_top = CY
    y_bot = EXCL_Y + EXCL_H / 2
    ax.plot([mx, mx], [y_top, y_bot + 0.08],
            color=EXCL_EDGE, lw=1.4, linestyle='--', zorder=2)
    ax.annotate('', xy=(mx, y_bot + 0.03), xytext=(mx, y_bot + 0.20),
                arrowprops=dict(arrowstyle='->', color=EXCL_EDGE,
                                lw=1.4, mutation_scale=13), zorder=2)
    ax.plot(mx, CY, 'o', color=ARROW_COLOR, markersize=5, zorder=4)
    excl_box(mx, EXCL_Y, text)

# ── Connection from last main box to split section ────────────────────────────
rx    = CX[4] + BOX_W / 2
lx    = SPLIT_CX - SPLIT_W / 2
mid_x = (rx + lx) / 2

# Horizontal lead line from box 5
ax.plot([rx + 0.05, mid_x], [CY, CY],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Vertical spine connecting to all 4 treatment boxes
ax.plot([mid_x, mid_x], [ALL_CY[-1], ALL_CY[0]],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Horizontal branch arrows to each treatment box
for sy in ALL_CY:
    ax.plot([mid_x, lx], [sy, sy], color=ARROW_COLOR, lw=1.5, zorder=2)
    ax.annotate('', xy=(lx + 0.05, sy), xytext=(lx - 0.05, sy),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.5, mutation_scale=14), zorder=2)

# "Other" exclusion — drops below the bottom box (CRT)
other_y = 2.5
ax.plot([mid_x, mid_x],
        [ALL_CY[-1] - SPLIT_H/2 - 0.05, other_y + EXCL_H/2 + 0.08],
        color=EXCL_EDGE, lw=1.4, linestyle='--', zorder=2)
ax.annotate('', xy=(mid_x, other_y + EXCL_H/2 + 0.03),
            xytext=(mid_x, other_y + EXCL_H/2 + 0.20),
            arrowprops=dict(arrowstyle='->', color=EXCL_EDGE,
                            lw=1.4, mutation_scale=13), zorder=2)
excl_box(mid_x, other_y,
         f"Other/excluded (n = {fmt(n_other)})\n(chemo alone, etc.)")

# ── Treatment group boxes ─────────────────────────────────────────────────────
split_box(SPLIT_CX, CY_TORS_ALONE, "TORS alone",  f"n = {fmt(n_tors_alone)}", TORS_COL)
split_box(SPLIT_CX, CY_RT_ALONE,   "RT alone",    f"n = {fmt(n_rt_alone)}",   RT_COL)
split_box(SPLIT_CX, CY_TORS_RT,    "TORS + RT",   f"n = {fmt(n_tors_rt)}",    TORSR_COL)
split_box(SPLIT_CX, CY_TORS_CRT,   "TORS + CRT",  f"n = {fmt(n_tors_crt)}",   TORSCRT_COL)
split_box(SPLIT_CX, CY_CRT,        "CRT",         f"n = {fmt(n_ctcrt)}",      CTCRT_COL)

# ── Comparison grouping brackets on right side ────────────────────────────────
# Comp A and Comp B use single (adjacent-rows) brackets at the standard offset.
# Comp C bridges TORS+RT and TORS+CRT (adjacent) at a slightly larger x offset
# to avoid overlap with the Comp B bracket which spans non-adjacent rows.
comp_bracket(CY_TORS_ALONE, CY_RT_ALONE, "Comparison A", SCARLET)

# Comp B bracket spans TORS+RT to CRT — skips TORS+CRT visually but bracket
# itself is just on the standard x offset (vertical bar continuous).
comp_bracket(CY_TORS_RT, CY_CRT, "Comparison B", DARK40)

# Comp C bracket: TORS+CRT and CRT (adjacent bottom two boxes) — drawn
# at a wider x offset so it doesn't visually overlap the Comp B bracket
# which also reaches CRT
bx2 = SPLIT_CX + SPLIT_W/2 + 0.95
y_top = CY_TORS_CRT
y_bot = CY_CRT
mid_y = (y_top + y_bot) / 2
ax.plot([bx2, bx2], [y_bot - 0.3, y_top + 0.3], color=TORSCRT_COL, lw=2.0, zorder=4)
ax.plot([bx2, bx2 + 0.15], [y_top + 0.3, y_top + 0.3], color=TORSCRT_COL, lw=2.0, zorder=4)
ax.plot([bx2, bx2 + 0.15], [y_bot - 0.3, y_bot - 0.3], color=TORSCRT_COL, lw=2.0, zorder=4)
ax.text(bx2 + 0.28, mid_y, 'Comparison C', ha='left', va='center',
        fontsize=19, fontweight='bold', color=TORSCRT_COL, zorder=4,
        rotation=90, linespacing=1.3)

# Separator between Comp A (top 2) and Comp B/C (bottom 3) groups
sep_y = (CY_RT_ALONE + CY_TORS_RT) / 2
ax.plot([SPLIT_CX - SPLIT_W/2 - 0.1, SPLIT_CX + SPLIT_W/2 + 0.1],
        [sep_y, sep_y],
        color=GRAY, lw=0.8, linestyle=':', zorder=2)

# ── PSM note between the comparisons ─────────────────────────────────────────
ax.text(mid_x, sep_y,
        '1:1 PSM per comparison',
        ha='center', va='center', fontsize=18, color='#555555',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=GRAY, linewidth=0.8))

# ── Title & footnote ──────────────────────────────────────────────────────────
ax.text(14.0, 10.65,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=33, fontweight='bold', color=DARK60)

ax.text(14.0, 0.38,
        "Medicare FFS claims 2016\u2013H1 2023  \u2022  OPSCC subsites: base of tongue (C01), "
        "tonsil (C09), oropharynx (C10), NOS (C14)  \u2022  "
        "Comp A = TORS alone vs RT alone;  Comp B = TORS + RT vs CRT;  "
        "Comp C = TORS + CRT vs CRT  \u2022  "
        "1:1 PSM per comparison, caliper = 0.2 \u00d7 SD(logit PS)",
        ha='center', va='center', fontsize=16, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
