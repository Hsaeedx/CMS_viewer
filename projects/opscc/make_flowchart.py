"""
make_flowchart.py
OPSCC cohort flowchart — horizontal layout.
Main flow: left to right  |  Exclusions: drop down  |  Split: 4 groups on right

Right side shows four treatment groups grouped by comparison:
  Comparison A: TORS alone  vs  RT alone
  Comparison B: TORS + RT   vs  CT/CRT
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = r"F:\CMS\projects\opscc\cohort_flowchart.png"

fig, ax = plt.subplots(figsize=(28, 11))
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
CTCRT_COL   = '#5E6E73'  # CT/CRT     (slightly darker neutral)

ARROW_COLOR = DARK60

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W   = 3.1
BOX_H   = 1.15
EXCL_W  = 3.4
EXCL_H  = 0.90
SPLIT_W = 3.5
SPLIT_H = 1.0

CY     = 6.5   # y-centre of main horizontal flow
EXCL_Y = 3.4   # y-centre of main exclusion boxes

CX = [2.4, 6.4, 10.4, 14.4, 18.4]   # x-centres of 5 main flow boxes

# Right-side treatment group y-centres (2 comps, 2 boxes each)
SPLIT_CX  = 24.5
CY_A = [9.4, 8.0]   # Comp A: TORS alone (top), RT alone (bottom)
CY_B = [6.2, 4.8]   # Comp B: TORS+RT (top), CT/CRT (bottom)
ALL_CY = CY_A + CY_B


# ── Helpers ───────────────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=DARK60, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.18, title,
            ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.24, subtitle,
            ha='center', va='center', fontsize=10, color='#E0B8C0', zorder=4)


def excl_box(cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor=EXCL_BG, edgecolor=EXCL_EDGE,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=8.0, color=EXCL_FG,
            zorder=4, linespacing=1.4, multialignment='center')


def split_box(cx, cy, title, subtitle, color):
    rect = FancyBboxPatch((cx - SPLIT_W/2, cy - SPLIT_H/2), SPLIT_W, SPLIT_H,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.19, title,
            ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='white', zorder=4)
    ax.text(cx, cy - 0.22, subtitle,
            ha='center', va='center', fontsize=9.5, color='white', zorder=4)


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
            ha='left', va='center', fontsize=8.5, fontweight='bold',
            color=color, zorder=4, rotation=90, linespacing=1.3)


# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(CX[0], CY, "Medicare HNC Claims\n(C00\u2013C14, C30\u2013C32)", "N = 278,698")
main_box(CX[1], CY, "Confirmed HNC", "N = 195,861")
main_box(CX[2], CY, "OPSCC Subsite\n(C01/C09/C10/C14)", "N = 80,819")
main_box(CX[3], CY, "Continuous FFS\nEnrollment \u22656 Months", "N = 71,441")
main_box(CX[4], CY, "Non-Metastatic\nOPSCC", "N = 37,577")

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
    "Excluded: Unconfirmed\ndiagnosis  (n = 82,837)",
    "Excluded: Non-OPSCC\nsubsite  (n = 115,042)",
    "Excluded: Incomplete\nFFS enrollment  (n = 9,378)",
    "Excluded: Metastatic\ndisease  (n = 33,864)",
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

# "Other" exclusion — drops below CT/CRT
other_y = 3.1
ax.plot([mid_x, mid_x],
        [ALL_CY[-1] - SPLIT_H/2 - 0.05, other_y + EXCL_H/2 + 0.08],
        color=EXCL_EDGE, lw=1.4, linestyle='--', zorder=2)
ax.annotate('', xy=(mid_x, other_y + EXCL_H/2 + 0.03),
            xytext=(mid_x, other_y + EXCL_H/2 + 0.20),
            arrowprops=dict(arrowstyle='->', color=EXCL_EDGE,
                            lw=1.4, mutation_scale=13), zorder=2)
excl_box(mid_x, other_y,
         "Other/excluded (n = 27,226)\n(TORS+chemo, chemo only, etc.)")

# ── Treatment group boxes ─────────────────────────────────────────────────────
split_box(SPLIT_CX, CY_A[0], "TORS alone",   "n = 637",   TORS_COL)
split_box(SPLIT_CX, CY_A[1], "RT alone",     "n = 4,549", RT_COL)
split_box(SPLIT_CX, CY_B[0], "TORS + RT",    "n = 118",   TORSR_COL)
split_box(SPLIT_CX, CY_B[1], "CT/CRT",       "n = 5,047", CTCRT_COL)

# ── Comparison grouping brackets on right side ────────────────────────────────
comp_bracket(CY_A[0], CY_A[1], "Comparison A", SCARLET)
comp_bracket(CY_B[0], CY_B[1], "Comparison B", DARK40)

# Separator line between Comparison A and B groups
sep_y = (CY_A[1] + CY_B[0]) / 2
ax.plot([SPLIT_CX - SPLIT_W/2 - 0.1, SPLIT_CX + SPLIT_W/2 + 0.1],
        [sep_y, sep_y],
        color=GRAY, lw=0.8, linestyle=':', zorder=2)

# ── PSM note between the two comparisons ─────────────────────────────────────
ax.text(mid_x, sep_y,
        '1:1 PSM per comparison',
        ha='center', va='center', fontsize=7.5, color='#555555',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=GRAY, linewidth=0.8))

# ── Title & footnote ──────────────────────────────────────────────────────────
ax.text(14.0, 10.65,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=15, fontweight='bold', color=DARK60)

ax.text(14.0, 0.38,
        "Medicare FFS claims 2016\u2013H1 2023  \u2022  OPSCC subsites: base of tongue (C01), "
        "tonsil (C09), oropharynx (C10), NOS (C14)  \u2022  "
        "Comp A = TORS alone vs RT alone;  Comp B = TORS + RT vs CT/CRT  \u2022  "
        "1:1 PSM per comparison, caliper = 0.2 \u00d7 SD(logit PS)",
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
