"""
make_figures.py
Figure 1: Hospice enrollment rate by year (bar chart)
Figure 2: Distribution of days from last IO to death (histogram)
Figure 3: Hospice LOS distribution among enrolled patients (histogram)
Output: C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/fig1_hospice_by_year.png
        C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/fig2_days_io_to_death.png
        C:/Users/hsaee/Desktop/CMS_viewer/projects/HNC_io_hosp/fig3_hospice_los.png
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats

DB_PATH = r"F:\CMS\cms_data.duckdb"
OUT_DIR = r"C:\Users\hsaee\Desktop\CMS_viewer\projects\HNC_io_hosp"

PRIMARY  = '#ba0c2f'   # OSU Scarlet
ACCENT   = '#7a0820'   # Dark scarlet
GRAY     = '#a7b1b7'   # OSU Gray
LIGHT    = '#f5d0d6'   # Light scarlet tint
GREEN    = '#ba0c2f'   # reuse scarlet (no green in OSU palette)

print("Loading data...")
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("SET memory_limit='24GB'; SET threads=12;")
df = con.execute("""
    SELECT hospice_enrolled, hospice_los_days,
           days_last_io_to_death, days_last_io_to_death_cat,
           days_last_io_to_hospice,
           death_year, io_agent, io_regimen
    FROM io_analytic
""").df()
con.close()

df['hospice_enrolled'] = df['hospice_enrolled'].fillna(0).astype(int)

# ── Figure 1: Hospice enrollment rate by year ─────────────────────────────────
print("Figure 1: hospice by year...")

yr_data = (df.groupby('death_year')
             .agg(total=('hospice_enrolled', 'count'),
                  enrolled=('hospice_enrolled', 'sum'))
             .reset_index())
yr_data['rate'] = 100.0 * yr_data['enrolled'] / yr_data['total']

# Cochran-Armitage trend test (approximate via logistic)
yr_data_sorted = yr_data.sort_values('death_year')
years = yr_data_sorted['death_year'].values
rates = yr_data_sorted['rate'].values

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')

bars = ax.bar(yr_data_sorted['death_year'].astype(int),
              yr_data_sorted['rate'],
              color=PRIMARY, edgecolor='white', linewidth=0.8, width=0.6)

# Value labels on bars
for bar, r, n in zip(bars, yr_data_sorted['rate'], yr_data_sorted['total']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{r:.1f}%\n(n={n:,})',
            ha='center', va='bottom', fontsize=8.5, color='#333333')

# Trend line
z = np.polyfit(yr_data_sorted['death_year'].astype(int), yr_data_sorted['rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(years.min(), years.max(), 100)
ax.plot(x_line, p(x_line), color=ACCENT, lw=2, linestyle='--', label='Linear trend')

ax.set_xlabel('Year of Death', fontsize=11)
ax.set_ylabel('Hospice Enrollment Rate (%)', fontsize=11)
ax.set_title('Figure 1. Hospice Enrollment Rate Among HNC Patients\nReceiving Immune Checkpoint Inhibitors, 2017–2023',
             fontsize=12, fontweight='bold', color=PRIMARY, pad=12)
ax.set_ylim(0, max(yr_data_sorted['rate']) * 1.25)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

plt.tight_layout()
out1 = f"{OUT_DIR}\\fig1_hospice_by_year.png"
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out1}")

# ── Figure 2: Days from last IO to death ─────────────────────────────────────
print("Figure 2: days last IO to death...")

days = df['days_last_io_to_death'].dropna()
days_clip = days.clip(upper=365)  # cap at 1 year for display

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

# Left: histogram
ax = axes[0]
ax.hist(days_clip, bins=40, color=PRIMARY, edgecolor='white', linewidth=0.5)
ax.axvline(days.median(), color=ACCENT, lw=2, linestyle='--',
           label=f'Median = {days.median():.0f} days')
ax.set_xlabel('Days from Last IO Dose to Death', fontsize=10)
ax.set_ylabel('Number of Patients', fontsize=10)
ax.set_title('Distribution (capped at 365 days)', fontsize=10, fontweight='bold', color=PRIMARY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

# Right: bar chart by category
ax = axes[1]
cat_order = ['<=3 days', '4-14 days', '15-30 days', '31-90 days', '>90 days']
cat_labels = ['≤3', '4–14', '15–30', '31–90', '>90']
cat_counts = [(df['days_last_io_to_death_cat'] == c).sum() for c in cat_order]
cat_pcts   = [100.0 * n / len(df) for n in cat_counts]

bars = ax.bar(range(len(cat_order)), cat_pcts, color=PRIMARY, edgecolor='white', linewidth=0.8)
for i, (bar, n, p) in enumerate(zip(bars, cat_counts, cat_pcts)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{p:.1f}%\n(n={n:,})',
            ha='center', va='bottom', fontsize=8.5, color='#333333')

ax.set_xticks(range(len(cat_order)))
ax.set_xticklabels(cat_labels)
ax.set_xlabel('Days from Last IO Dose to Death', fontsize=10)
ax.set_ylabel('Patients (%)', fontsize=10)
ax.set_title('By Time Category', fontsize=10, fontweight='bold', color=PRIMARY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

fig.suptitle('Figure 2. Timing from Last IO Dose to Death',
             fontsize=12, fontweight='bold', color=PRIMARY, y=1.01)
plt.tight_layout()
out2 = f"{OUT_DIR}\\fig2_days_io_to_death.png"
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out2}")

# ── Figure 3: Hospice LOS distribution ───────────────────────────────────────
print("Figure 3: hospice LOS...")

hosp = df[df['hospice_enrolled'] == 1].copy()
los  = hosp['hospice_los_days'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

# Left: histogram (capped at 180 days)
ax = axes[0]
los_clip = los.clip(upper=180)
ax.hist(los_clip, bins=36, color=GREEN, edgecolor='white', linewidth=0.5)
ax.axvline(7, color=ACCENT, lw=1.8, linestyle='--', label='7-day threshold')
ax.axvline(los.median(), color='#843C0C', lw=2, linestyle=':',
           label=f'Median = {los.median():.0f} days')
ax.set_xlabel('Hospice LOS (days, capped at 180)', fontsize=10)
ax.set_ylabel('Number of Patients', fontsize=10)
ax.set_title('Distribution', fontsize=10, fontweight='bold', color=PRIMARY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

# Right: LOS category bar chart
ax = axes[1]
los_cats = [
    ('1–3 days',  los <= 3),
    ('4–7 days',  los.between(4, 7)),
    ('8–14 days', los.between(8, 14)),
    ('15–30 days',los.between(15, 30)),
    ('>30 days',  los > 30),
]
cat_labels2 = [c[0] for c in los_cats]
cat_n = [c[1].sum() for c in los_cats]
cat_p = [100.0 * n / len(hosp) for n in cat_n]

bars = ax.bar(range(len(cat_n)), cat_p, color=GREEN, edgecolor='white', linewidth=0.8)
for bar, n, p in zip(bars, cat_n, cat_p):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{p:.1f}%\n(n={n:,})',
            ha='center', va='bottom', fontsize=8.5, color='#333333')

ax.set_xticks(range(len(cat_labels2)))
ax.set_xticklabels(cat_labels2, rotation=20, ha='right')
ax.set_ylabel('Hospice-Enrolled Patients (%)', fontsize=10)
ax.set_title('By LOS Category', fontsize=10, fontweight='bold', color=PRIMARY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

fig.suptitle(f'Figure 3. Hospice Length of Stay Among Enrolled Patients (n = {len(hosp):,})',
             fontsize=12, fontweight='bold', color=PRIMARY, y=1.01)
plt.tight_layout()
out3 = f"{OUT_DIR}\\fig3_hospice_los.png"
plt.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out3}")

print("\nAll figures complete.")
