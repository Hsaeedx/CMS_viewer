"""Quick statistical tests: hospice vs no-hospice on key outcomes."""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
from scipy.stats import mannwhitneyu, median_test, chi2_contingency

con = duckdb.connect(r'F:\CMS\cms_data.duckdb', read_only=True)
df = con.execute("""
    SELECT hospice_enrolled, days_last_io_to_death, in_hospital_death
    FROM io_analytic
""").df()
con.close()

# Days from last ICI to death — continuous, skewed
h  = df[df['hospice_enrolled'] == 1]['days_last_io_to_death'].dropna().values
nh = df[df['hospice_enrolled'] == 0]['days_last_io_to_death'].dropna().values
print('Days from last ICI to death:')
print(f'  Hospice    (n={len(h):,}): median = {np.median(h):.0f} (IQR {np.quantile(h, .25):.0f} - {np.quantile(h, .75):.0f})')
print(f'  No hospice (n={len(nh):,}): median = {np.median(nh):.0f} (IQR {np.quantile(nh, .25):.0f} - {np.quantile(nh, .75):.0f})')
u, p_mwu = mannwhitneyu(h, nh, alternative='two-sided')
print(f'  Mann-Whitney U: U = {u:,.0f}, p = {p_mwu:.3e}')
stat, p_mood, _, _ = median_test(h, nh)
print(f"  Mood's median test: chi2 = {stat:.2f}, p = {p_mood:.3e}")

print()

# In-hospital death — binary
ct = np.array([
    [int(((df['hospice_enrolled'] == 1) & (df['in_hospital_death'] == 1)).sum()),
     int(((df['hospice_enrolled'] == 1) & (df['in_hospital_death'] == 0)).sum())],
    [int(((df['hospice_enrolled'] == 0) & (df['in_hospital_death'] == 1)).sum()),
     int(((df['hospice_enrolled'] == 0) & (df['in_hospital_death'] == 0)).sum())],
])
print('In-hospital death:')
print(f'  Hospice    : {ct[0,0]:,} / {ct[0].sum():,} ({100*ct[0,0]/ct[0].sum():.1f}%)')
print(f'  No hospice : {ct[1,0]:,} / {ct[1].sum():,} ({100*ct[1,0]/ct[1].sum():.1f}%)')
chi2, p_chi, _, _ = chi2_contingency(ct)
print(f'  Chi-square (with continuity correction): chi2 = {chi2:.2f}, p = {p_chi:.3e}')
