"""
make_flowchart.py
IO Hospice cohort CONSORT flowchart — two versions:
  Horizontal: main flow left to right, exclusions drop down
  Vertical:   main flow top to bottom, exclusions branch right
N values queried live from cms_data.duckdb.
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DB_PATH  = r'F:\CMS\cms_data.duckdb'
OUT_PATH = r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures\flowchart.png'

# ── Query N values from database ──────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)

n_decedents   = con.execute("SELECT COUNT(*) FROM io_decedents").fetchone()[0]
n_hnc_confirm = con.execute("SELECT COUNT(*) FROM io_hnc_confirmed").fetchone()[0]
n_subsite     = con.execute("SELECT COUNT(*) FROM io_subsite").fetchone()[0]
n_io          = con.execute("""
    SELECT COUNT(DISTINCT e.DSYSRTKY)
    FROM io_episodes e JOIN io_subsite s ON e.DSYSRTKY = s.DSYSRTKY
""").fetchone()[0]
n_ffs         = con.execute("SELECT COUNT(*) FROM io_ffs_eligible").fetchone()[0]
n_curative    = con.execute("SELECT COUNT(*) FROM io_curative").fetchone()[0]
n_cohort      = con.execute("SELECT COUNT(*) FROM io_cohort").fetchone()[0]

con.close()

def fmt(n):
    return f"N = {n:,}"

# Exclusion counts (difference between consecutive steps)
excl_no_hnc      = n_decedents   - n_hnc_confirm
excl_subsite     = n_hnc_confirm - n_subsite
excl_no_io       = n_subsite     - n_io
excl_ffs         = n_io          - n_ffs
excl_no_curative = n_ffs         - n_curative
excl_180d        = n_curative    - n_cohort

fig, ax = plt.subplots(figsize=(34, 7))
ax.set_xlim(0, 34)
ax.set_ylim(1.9, 8.2)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colors (OSU) ─────────────────────────────────────────────────────────────
BOX_COLOR   = '#ba0c2f'
EXCL_COLOR  = '#7a0820'
FINAL_COLOR = '#4a0612'
ARROW_COLOR = '#a7b1b7'

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W  = 3.6
BOX_H  = 1.20
EXCL_W = 3.5
EXCL_H = 0.95

CY     = 6.5
EXCL_Y = 3.2

CX = [2.3, 7.0, 11.7, 16.4, 21.1, 25.8, 30.5]

# ── Helper functions ──────────────────────────────────────────────────────────
def main_box(cx, cy, title, subtitle, color=BOX_COLOR):
    rect = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.20, title,
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='white', zorder=4, multialignment='center')
    ax.text(cx, cy - 0.25, subtitle,
            ha='center', va='center', fontsize=16, color='#D6E4F0', zorder=4)

def excl_box(cx, cy, text):
    rect = FancyBboxPatch((cx - EXCL_W/2, cy - EXCL_H/2), EXCL_W, EXCL_H,
                          boxstyle="round,pad=0.05",
                          facecolor='#f5d0d6', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=18, color='#7a0820',
            zorder=4, linespacing=1.4, multialignment='center')

# ── Main flow boxes ───────────────────────────────────────────────────────────
main_box(CX[0], CY, "Medicare Decedents\nAge ≥66, 2017–2023",  fmt(n_decedents))
main_box(CX[1], CY, "≥2 HNC Claims\non Separate Dates",        fmt(n_hnc_confirm))
main_box(CX[2], CY, "Eligible HNC\nSubsite",                   fmt(n_subsite))
main_box(CX[3], CY, "≥1 ICI Claim\n(J9271 / J9299)",           fmt(n_io))
main_box(CX[4], CY, "Continuous FFS\nEnrollment",              fmt(n_ffs))
main_box(CX[5], CY, "Curative-Intent\nTherapy Before ICI",     fmt(n_curative))
main_box(CX[6], CY, "Final Analytic\nCohort",                  fmt(n_cohort), color=FINAL_COLOR)

# ── Horizontal arrows between main boxes ─────────────────────────────────────
for i in range(6):
    x0 = CX[i]   + BOX_W / 2
    x1 = CX[i+1] - BOX_W / 2
    ax.annotate('', xy=(x1 - 0.05, CY), xytext=(x0 + 0.05, CY),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.8, mutation_scale=16), zorder=2)

MID_XS = [(CX[i] + BOX_W/2 + CX[i+1] - BOX_W/2) / 2 for i in range(6)]

# ── Exclusion labels ──────────────────────────────────────────────────────────
excl_texts = [
    f"Excluded: <2 HNC claims\nN = {excl_no_hnc:,}",
    f"Excluded: Non-mucosal\nHNC subsite\nN = {excl_subsite:,}",
    f"Excluded: No ICI claims\nwithin 24 months\nof death\nN = {excl_no_io:,}",
    f"Excluded: ESRD,\nmissing geography, or\nMA enrollment\nN = {excl_ffs:,}",
    f"Excluded: No curative-intent\ntherapy before last\nICI episode\nN = {excl_no_curative:,}",
    f"Excluded: <180 days\nfrom HNC dx to last\nICI episode\nN = {excl_180d:,}",
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
ax.text(17.0, 7.85,
        "Cohort Derivation",
        ha='center', va='center',
        fontsize=20, fontweight='bold', color='#ba0c2f')

ax.text(17.0, 2.1,
        "Medicare FFS claims 2017–2023  •  ICI agents: pembrolizumab (J9271), nivolumab (J9299)  "
        "•  HNC subsites: oral cavity, oropharynx, hypopharynx, larynx, other  "
        "•  Lookback: 24 months prior to death",
        ha='center', va='center', fontsize=9, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH}")

# ── VERTICAL VERSION ──────────────────────────────────────────────────────────
OUT_PATH_V = r'C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp\figures\flowchart_vertical.png'

BOX_W_V  = 5.5
BOX_H_V  = 1.2
EXCL_W_V = 4.5
EXCL_H_V = 1.5

CX_MAIN = 4.0
CX_EXCL = 10.5

CY_VALS = [20.0, 17.0, 14.0, 11.0, 8.0, 5.0, 2.0]
MID_YS  = [(CY_VALS[i] + CY_VALS[i+1]) / 2 for i in range(6)]

fig2, ax2 = plt.subplots(figsize=(12, 18))
ax2.set_xlim(0, 13)
ax2.set_ylim(0.2, 21.5)
ax2.axis('off')
fig2.patch.set_facecolor('white')

def main_box_v(cy, title, subtitle, color=BOX_COLOR):
    rect = FancyBboxPatch((CX_MAIN - BOX_W_V/2, cy - BOX_H_V/2), BOX_W_V, BOX_H_V,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax2.add_patch(rect)
    ax2.text(CX_MAIN, cy + 0.18, title,
             ha='center', va='center', fontsize=16, fontweight='bold',
             color='white', zorder=4, multialignment='center')
    ax2.text(CX_MAIN, cy - 0.34, subtitle,
             ha='center', va='center', fontsize=16, color='#D6E4F0', zorder=4)

def excl_box_v(cy, text):
    rect = FancyBboxPatch((CX_EXCL - EXCL_W_V/2, cy - EXCL_H_V/2), EXCL_W_V, EXCL_H_V,
                          boxstyle="round,pad=0.05",
                          facecolor='#f5d0d6', edgecolor=EXCL_COLOR,
                          linewidth=1.2, zorder=3)
    ax2.add_patch(rect)
    lines = text.split('\n')
    # Last line is always the "(n = X,XXX)" count
    n_line   = lines[-1]
    label    = '\n'.join(lines[:-1])
    ax2.text(CX_EXCL, cy + 0.22, label,
             ha='center', va='center', fontsize=16, fontweight='bold', color='#7a0820',
             zorder=4, linespacing=1.3, multialignment='center')
    ax2.text(CX_EXCL, cy - 0.48, n_line,
             ha='center', va='center', fontsize=16, color='#a0253a',
             zorder=4)

# Main flow boxes
box_data_v = [
    ("Medicare Decedents\nAge ≥66, 2017–2023", fmt(n_decedents),   BOX_COLOR),
    ("≥2 HNC Claims\non Separate Dates",       fmt(n_hnc_confirm), BOX_COLOR),
    ("Eligible HNC\nSubsite",                  fmt(n_subsite),     BOX_COLOR),
    ("≥1 ICI Claim\n(J9271 / J9299)",          fmt(n_io),          BOX_COLOR),
    ("Continuous FFS\nEnrollment",             fmt(n_ffs),         BOX_COLOR),
    ("Curative-Intent\nTherapy Before ICI",    fmt(n_curative),    BOX_COLOR),
    ("Final Analytic\nCohort",                 fmt(n_cohort),      FINAL_COLOR),
]

for (title, subtitle, color), cy in zip(box_data_v, CY_VALS):
    main_box_v(cy, title, subtitle, color)

# Vertical arrows between main boxes
for i in range(6):
    y0 = CY_VALS[i]   - BOX_H_V / 2
    y1 = CY_VALS[i+1] + BOX_H_V / 2
    ax2.annotate('', xy=(CX_MAIN, y1 + 0.05), xytext=(CX_MAIN, y0 - 0.05),
                 arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                 lw=1.8, mutation_scale=16), zorder=2)

# Exclusion branches
for my, text in zip(MID_YS, excl_texts):
    ax2.plot(CX_MAIN, my, 'o', color=ARROW_COLOR, markersize=5, zorder=4)
    x_left = CX_EXCL - EXCL_W_V / 2
    ax2.plot([CX_MAIN, x_left - 0.05], [my, my],
             color=EXCL_COLOR, lw=1.4, linestyle='--', zorder=2)
    ax2.annotate('', xy=(x_left, my), xytext=(CX_MAIN + 0.2, my),
                 arrowprops=dict(arrowstyle='->', color=EXCL_COLOR,
                                 lw=1.4, mutation_scale=13), zorder=2)
    excl_box_v(my, text)

# Title & footnote
ax2.text(6.5, 21.1,
         "Figure 1. Cohort Derivation",
         ha='center', va='center',
         fontsize=20, fontweight='bold', color='#ba0c2f')

ax2.text(6.5, 0.75,
         "ICI agents: pembrolizumab, nivolumab\n"
         "HNC subsites: oral cavity, oropharynx, hypopharynx, larynx\n"
         "Lookback: 24 months prior to death",
         ha='center', va='center', fontsize=16, color='#555555', style='italic',
         multialignment='center')

plt.tight_layout(pad=0.1)
plt.savefig(OUT_PATH_V, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_PATH_V}")
