"""
Hodges-Lehmann median difference + 95% CI for days_last_io_to_death,
comparing hospice-enrolled vs non-enrolled. This is the JAMA-preferred
effect-size + CI alternative to a Mann-Whitney p-value.
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
from math import sqrt, floor, ceil

con = duckdb.connect(r'F:\CMS\cms_data.duckdb', read_only=True)
df = con.execute("""
    SELECT hospice_enrolled, days_last_io_to_death
    FROM io_analytic
""").df()
con.close()

a = df[df['hospice_enrolled'] == 1]['days_last_io_to_death'].dropna().values   # hospice
b = df[df['hospice_enrolled'] == 0]['days_last_io_to_death'].dropna().values   # no hospice
n1, n2 = len(a), len(b)
print(f'Hospice    n = {n1:,}; median = {np.median(a):.0f} (IQR {np.quantile(a, .25):.0f}-{np.quantile(a, .75):.0f})')
print(f'No hospice n = {n2:,}; median = {np.median(b):.0f} (IQR {np.quantile(b, .25):.0f}-{np.quantile(b, .75):.0f})')
print(f'Raw median difference (hospice - no hospice) = {np.median(a) - np.median(b):.0f} days')
print()

# Hodges-Lehmann estimator: median of all pairwise differences a_i - b_j
# n1 * n2 = 1670 * 857 = 1,431,190 — manageable in memory
print(f'Computing {n1 * n2:,} pairwise differences ...')
diffs = (a[:, None] - b[None, :]).ravel()
diffs.sort()
N = len(diffs)

HL = np.median(diffs)
print(f'Hodges-Lehmann median difference = {HL:.1f} days')

# 95% CI via the standard rank-based (Wilcoxon) method, normal approximation
# Lower rank: k = N/2 - z_{0.975} * sqrt(n1*n2*(n1+n2+1)/12)
# Upper rank: N - k + 1
z = 1.959964        # two-sided 95%
se = sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
k = int(floor(N / 2.0 - z * se))
ci_lo = diffs[k - 1]                # 1-indexed rank → 0-indexed array
ci_hi = diffs[N - k]                # mirror
print(f'95% CI (rank-based, normal approximation): {ci_lo:.1f} to {ci_hi:.1f} days')

# Also report the bootstrap CI as a sanity check (10,000 resamples)
rng = np.random.default_rng(42)
boot_HL = np.empty(10_000)
for i in range(10_000):
    boot_a = rng.choice(a, size=n1, replace=True)
    boot_b = rng.choice(b, size=n2, replace=True)
    boot_HL[i] = np.median((boot_a[:, None] - boot_b[None, :]).ravel())
boot_lo, boot_hi = np.quantile(boot_HL, [0.025, 0.975])
print(f'95% CI (10,000 bootstrap resamples): {boot_lo:.1f} to {boot_hi:.1f} days')
