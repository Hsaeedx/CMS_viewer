"""
make_flowchart.py
OPSCC cohort flowchart - horizontal layout (matches stroke SLP style)
Main flow: left to right  |  Exclusions: drop down  |  Split: stacked on right
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = r"F:\CMS\projects\opscc\cohort_flowchart.png"

fig, ax = plt.subplots(figsize=(26, 10))
ax.set_xlim(0, 26)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colors ────────────────────────────────────────────────────────────────────
BOX_COLOR    = '#1F4E79'
EXCL_COLOR   = '#C55A11'
TORS_COLOR   = '#1F6B3A'
CTCRT_COLOR  = '#2E5090'
MATCH_COLOR  = '#4A4A8A'
ARROW_COLOR  = '#2E5090'

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W   = 3.1    # main inclusion box width
BOX_H   = 1.15   # main inclusion box height
EXCL_W  = 3.4    # exclusion box width
EXCL_H  = 0.90   # exclusion box height
SPLIT_W = 3.2    # right-side split box width
SPLIT_H = 1.05   # right-side split box height

CY     = 6.5     # y-center of main horizontal flow
EXCL_Y = 3.4     # y-center of exclusion boxes

# x-centers of 5 main inclusion boxes
CX = [2.4, 6.4, 10.4, 14.4, 18.4]

# x-center and y-centers of 3 right-side split boxes
SPLIT_CX = 24.0
SPLIT_CY = [8.5, 6.0, 3.5]


# ── Helper functions ──────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=BOX_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.18, title,
            ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.24, subtitle,
            ha='center', va='center', fontsize=10, color='#D6E4F0', zorder=4)


def excl_box(cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor='#FCE4D6', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=8.0, color='#843C0C',
            zorder=4, linespacing=1.4, multialignment='center')


def split_box(cx, cy, title, subtitle, color):
    rect = FancyBboxPatch((cx - SPLIT_W/2, cy - SPLIT_H/2), SPLIT_W, SPLIT_H,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.19, title,
            ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='white', zorder=4)
    ax.text(cx, cy - 0.22, subtitle,
            ha='center', va='center', fontsize=9.5, color='white', zorder=4)


# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(CX[0], CY, "Medicare HNC Claims\n(C00\u2013C14, C30\u2013C32)", "N = 252,352")
main_box(CX[1], CY, "Confirmed HNC", "N = 175,989")
main_box(CX[2], CY, "OPSCC Subsite\n(C01/C09/C10/C14)", "N = 72,146")
main_box(CX[3], CY, "Continuous FFS\nEnrollment \u22656 Months", "N = 65,275")
main_box(CX[4], CY, "Non-Metastatic\nOPSCC", "N = 34,621")


# ── Horizontal arrows between main boxes ─────────────────────────────────────
for i in range(4):
    x0 = CX[i]   + BOX_W / 2
    x1 = CX[i+1] - BOX_W / 2
    ax.annotate('', xy=(x1 - 0.05, CY), xytext=(x0 + 0.05, CY),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16), zorder=2)

# Midpoints of horizontal arrows (branch points for exclusion drops)
MID_XS = [(CX[i] + BOX_W/2 + CX[i+1] - BOX_W/2) / 2 for i in range(4)]


# ── Exclusion boxes with dashed vertical drops ────────────────────────────────
excl_texts = [
    "Excluded: Unconfirmed\ndiagnosis  (n = 76,363)",
    "Excluded: Non-OPSCC\nsubsite  (n = 103,843)",
    "Excluded: Incomplete\nFFS enrollment  (n = 6,871)",
    "Excluded: Metastatic\ndisease  (n = 30,654)",
]

for mx, text in zip(MID_XS, excl_texts):
    y_top = CY
    y_bot = EXCL_Y + EXCL_H / 2
    # Dashed vertical drop
    ax.plot([mx, mx], [y_top, y_bot + 0.08],
            color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
    # Arrowhead
    ax.annotate('', xy=(mx, y_bot + 0.03), xytext=(mx, y_bot + 0.20),
                arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                                lw=1.4, mutation_scale=13), zorder=2)
    # Junction dot
    ax.plot(mx, CY, 'o', color=ARROW_COLOR, markersize=5, zorder=4)
    excl_box(mx, EXCL_Y, text)


# ── Connection from box 5 to split section ────────────────────────────────────
rx    = CX[4] + BOX_W / 2
lx    = SPLIT_CX - SPLIT_W / 2
mid_x = (rx + lx) / 2

# Horizontal line from box 5 right edge to vertical bracket
ax.plot([rx + 0.05, mid_x], [CY, CY],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Vertical bracket spanning all split box y-centers
ax.plot([mid_x, mid_x], [SPLIT_CY[-1], SPLIT_CY[0]],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Horizontal arrows from bracket to each split box
for sy in SPLIT_CY:
    ax.plot([mid_x, lx], [sy, sy], color=ARROW_COLOR, lw=1.5, zorder=2)
    ax.annotate('', xy=(lx + 0.05, sy), xytext=(lx - 0.05, sy),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.5, mutation_scale=14), zorder=2)

# 5th exclusion: no treatment / both — drops down from mid_x
excl5_y = 1.7
ax.plot([mid_x, mid_x], [SPLIT_CY[-1], excl5_y + EXCL_H/2 + 0.08],
        color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
ax.annotate('', xy=(mid_x, excl5_y + EXCL_H/2 + 0.03),
            xytext=(mid_x, excl5_y + EXCL_H/2 + 0.20),
            arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                            lw=1.4, mutation_scale=13), zorder=2)
excl_box(mid_x, excl5_y,
         "Excluded: No treatment (n=21,432)\nor received both (n=306)")


# ── Split boxes ───────────────────────────────────────────────────────────────
split_box(SPLIT_CX, SPLIT_CY[0], "TORS Only",           "n = 598",    TORS_COLOR)
split_box(SPLIT_CX, SPLIT_CY[1], "CT/CRT Only",         "n = 12,285", CTCRT_COLOR)
split_box(SPLIT_CX, SPLIT_CY[2], "PSM-Matched Cohort",
          "n = 1,196  (598 + 598)", MATCH_COLOR)

# PSM note: dotted line from CT/CRT to PSM box
ax.plot([SPLIT_CX, SPLIT_CX],
        [SPLIT_CY[1] - SPLIT_H/2 - 0.05, SPLIT_CY[2] + SPLIT_H/2 + 0.05],
        color=MATCH_COLOR, lw=1.3, linestyle=':', zorder=2)

# Unmatched CT/CRT annotation
ax.text(SPLIT_CX + SPLIT_W/2 + 0.12, (SPLIT_CY[1] + SPLIT_CY[2]) / 2,
        "11,687 unmatched\nCT/CRT excluded",
        ha='left', va='center', fontsize=7.5, color='#843C0C',
        style='italic', linespacing=1.4)


# ── Title & footnote ──────────────────────────────────────────────────────────
ax.text(13.0, 9.55,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=15, fontweight='bold', color='#1F4E79')

ax.text(13.0, 0.38,
        "Medicare FFS claims 2016\u20132022  \u2022  OPSCC subsites: base of tongue (C01), tonsil (C09), "
        "oropharynx (C10), NOS (C14)  \u2022  1:1 PSM, caliper = 0.2 \u00d7 SD(logit PS)",
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')


plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
