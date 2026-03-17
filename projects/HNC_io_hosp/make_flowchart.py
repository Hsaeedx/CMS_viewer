"""
make_flowchart.py
IO Hospice cohort CONSORT flowchart — horizontal layout
Main flow: left to right  |  Exclusions: drop down
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\flowchart.png"

fig, ax = plt.subplots(figsize=(34, 10))
ax.set_xlim(0, 34)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colors (OSU) ─────────────────────────────────────────────────────────────
BOX_COLOR   = '#ba0c2f'   # OSU Scarlet
EXCL_COLOR  = '#7a0820'   # Darker scarlet for exclusions
FINAL_COLOR = '#4a0612'   # Deep scarlet for final box
ARROW_COLOR = '#a7b1b7'   # OSU Gray

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W  = 3.6
BOX_H  = 1.20
EXCL_W = 3.5
EXCL_H = 0.95

CY     = 6.5    # y-center of main flow
EXCL_Y = 3.2    # y-center of exclusion boxes

# 7 main boxes evenly spaced
CX = [2.3, 7.0, 11.7, 16.4, 21.1, 25.8, 30.5]

# ── Helper functions ──────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle, color=BOX_COLOR):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.20, title,
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.25, subtitle,
            ha='center', va='center', fontsize=12, color='#D6E4F0', zorder=4)

def excl_box(cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor='#f5d0d6', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=10, color='#7a0820',
            zorder=4, linespacing=1.4, multialignment='center')

# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(CX[0], CY, "Medicare Decedents\nAge ≥66, 2017–2023",  "N = 14,507,067")
main_box(CX[1], CY, "≥2 HNC Claims\non Separate Dates",        "N = 48,449")
main_box(CX[2], CY, "Eligible HNC\nSubsite",                   "N = 40,890")
main_box(CX[3], CY, "≥1 IO Claim\n(J9271 / J9299)",            "N = 7,123")
main_box(CX[4], CY, "Continuous FFS\nEnrollment",              "N = 5,947")
main_box(CX[5], CY, "Curative-Intent\nTherapy Before IO",      "N = 3,827")
main_box(CX[6], CY, "Final Analytic\nCohort",                  "N = 2,766", color=FINAL_COLOR)

# ── Horizontal arrows between main boxes ─────────────────────────────────────
for i in range(6):
    x0 = CX[i]   + BOX_W / 2
    x1 = CX[i+1] - BOX_W / 2
    ax.annotate('', xy=(x1 - 0.05, CY), xytext=(x0 + 0.05, CY),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16), zorder=2)

# Midpoints for exclusion drop positions
MID_XS = [(CX[i] + BOX_W/2 + CX[i+1] - BOX_W/2) / 2 for i in range(6)]

# ── Exclusion labels ──────────────────────────────────────────────────────────
excl_texts = [
    "Excluded: <2 HNC claims\nor no claims  (n = 14,458,618)",
    "Excluded: Excluded subsite\n(nasopharynx, salivary,\nsinonasal, cutaneous)\n(n = 7,559)",
    "Excluded: No IO\nclaims within\n24 months of death\n(n = 33,767)",
    "Excluded: ESRD,\nmissing geography, or\nMA enrollment  (n = 1,176)",
    "Excluded: No curative-intent\ntherapy before last\nIO episode  (n = 2,120)",
    "Excluded: <180 days\nfrom HNC dx to last\nIO episode  (n = 1,061)",
]

for mx, text in zip(MID_XS, excl_texts):
    y_top = CY - BOX_H / 2
    y_bot = EXCL_Y + EXCL_H / 2
    ax.plot([mx, mx], [y_top, y_bot + 0.08],
            color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
    ax.annotate('', xy=(mx, y_bot + 0.03), xytext=(mx, y_bot + 0.20),
                arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                                lw=1.4, mutation_scale=13), zorder=2)
    ax.plot(mx, y_top, 'o', color=ARROW_COLOR, markersize=5, zorder=4)
    excl_box(mx, EXCL_Y, text)

# ── Title & footnote ──────────────────────────────────────────────────────────
ax.text(17.0, 9.55,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=20, fontweight='bold', color='#ba0c2f')

ax.text(17.0, 0.45,
        "Medicare FFS claims 2017–2023  •  IO agents: pembrolizumab (J9271), nivolumab (J9299)  "
        "•  HNC subsites: oral cavity, oropharynx, hypopharynx, larynx, other  "
        "•  Lookback: 24 months prior to death",
        ha='center', va='center', fontsize=9, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")
