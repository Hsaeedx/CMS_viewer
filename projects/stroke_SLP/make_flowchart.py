"""
make_flowchart.py
Generates a cohort flowchart as a high-resolution PNG for PowerPoint.
Horizontal (landscape, left-to-right) layout.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_out_dir = Path(os.getenv("project_paths", ".")) / "stroke_SLP"
_out_dir.mkdir(parents=True, exist_ok=True)
OUT_PATH = _out_dir / "cohort_flowchart.png"

# ── Layout constants ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 9))
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

BOX_W    = 3.2     # main inclusion box width
BOX_H    = 1.1     # main inclusion box height
EXCL_W   = 3.5     # exclusion box width
EXCL_H   = 0.9     # exclusion box height
SPLIT_W  = 3.0     # timing group box width
SPLIT_H  = 1.05    # timing group box height

CY       = 5.5     # y-center of main horizontal flow
EXCL_Y   = 2.3     # y-center of exclusion boxes

# x-centers of 4 main inclusion boxes (evenly spaced with 0.7 gap)
CX = [2.0, 5.9, 9.8, 13.7]
# x-center and y-centers of the 3 final timing group boxes (right side, stacked)
SPLIT_CX = 18.5
SPLIT_CY = [7.5, 5.5, 3.5]

ARROW_COLOR  = '#2E5090'
BOX_COLOR    = '#1F4E79'
EXCL_COLOR   = '#C55A11'
SPLIT_COLORS = ['#1F6B3A', '#2E7D32', '#376E37']


# ── Helper functions ─────────────────────────────────────────────────────────

def main_box(ax, cx, cy, title, subtitle, color=BOX_COLOR):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.16, title,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.23, subtitle,
            ha='center', va='center', fontsize=10, color='#D6E4F0', zorder=4)


def excl_box(ax, cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor='#FCE4D6', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=8.5, color='#843C0C',
            zorder=4, linespacing=1.4, multialignment='center')


def split_box(ax, cx, cy, title, n, color):
    rect = FancyBboxPatch((cx - SPLIT_W/2, cy - SPLIT_H/2), SPLIT_W, SPLIT_H,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.20, title,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='white', zorder=4)
    ax.text(cx, cy - 0.20, n,
            ha='center', va='center', fontsize=10, color='white', zorder=4)


# ── Main flow boxes ──────────────────────────────────────────────────────────
main_box(ax, CX[0], CY,
         "Medicare Stroke Admissions",
         "N = 1,129,908")

main_box(ax, CX[1], CY,
         "Received SLP\nWithin 90 Days",
         "N = 441,692")

main_box(ax, CX[2], CY,
         "Survived\n90-Day Landmark",
         "N = 410,664")

main_box(ax, CX[3], CY,
         "Discharged Home or\nHome Health Agency",
         "N = 108,695")


# ── Horizontal arrows between main boxes ─────────────────────────────────────
for i in range(3):
    x0 = CX[i]   + BOX_W / 2
    x1 = CX[i+1] - BOX_W / 2
    ax.annotate('', xy=(x1 - 0.05, CY), xytext=(x0 + 0.05, CY),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16),
                zorder=2)

# Midpoints of the horizontal arrows (branch points for exclusion)
MID_XS = [(CX[i] + BOX_W/2 + CX[i+1] - BOX_W/2) / 2 for i in range(3)]


# ── Exclusion boxes and dashed downward arrows ───────────────────────────────
excl_texts = [
    "Excluded: No SLP\nwithin 90 days  (n = 688,216)",
    "Excluded: Died within\n90 days of discharge  (n = 31,028)",
    "Excluded: Discharged to\nSNF or IRF  (n = 301,969)",
]

for mx, text in zip(MID_XS, excl_texts):
    excl_box(ax, mx, EXCL_Y, text)
    y_top = CY
    y_bot = EXCL_Y + EXCL_H / 2
    # Dashed vertical line from main flow down to near exclusion box top
    ax.plot([mx, mx], [y_top, y_bot + 0.08],
            color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
    # Small solid arrowhead at bottom
    ax.annotate('', xy=(mx, y_bot + 0.03), xytext=(mx, y_bot + 0.18),
                arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                                lw=1.4, mutation_scale=13),
                zorder=2)


# ── Split into timing groups (right side, stacked vertically) ────────────────
labels = ['SLP 0 – 14 Days', 'SLP 15 – 30 Days', 'SLP 31 – 90 Days']
counts = ['n = 20,614', 'n = 31,965', 'n = 56,116']

# Geometry: horizontal line from box 4 right edge → vertical bracket → split boxes
rx    = CX[3] + BOX_W / 2                # right edge of box 4
lx    = SPLIT_CX - SPLIT_W / 2           # left edge of split boxes
mid_x = (rx + lx) / 2                    # x of vertical bracket

# Horizontal connector from box 4
ax.plot([rx + 0.05, mid_x], [CY, CY],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Vertical bracket spanning all split box y-centers
ax.plot([mid_x, mid_x], [SPLIT_CY[-1], SPLIT_CY[0]],
        color=ARROW_COLOR, lw=1.5, zorder=2)

# Horizontal arrows from bracket to each split box
for sy, lab, cnt, col in zip(SPLIT_CY, labels, counts, SPLIT_COLORS):
    ax.plot([mid_x, lx], [sy, sy], color=ARROW_COLOR, lw=1.5, zorder=2)
    ax.annotate('', xy=(lx + 0.05, sy), xytext=(lx - 0.05, sy),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    split_box(ax, SPLIT_CX, sy, lab, cnt, col)


# ── Title and footnote ────────────────────────────────────────────────────────
ax.text(9.0, 8.55,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=15, fontweight='bold', color='#1F4E79')

ax.text(9.0, 0.35,
        "Medicare FFS claims 2016–2022  •  SLP timing measured from date of discharge  •  90-day landmark analysis",
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')


plt.tight_layout(pad=0.3)
plt.savefig(str(OUT_PATH), dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
