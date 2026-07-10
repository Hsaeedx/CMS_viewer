"""
Validate the forest plot ORs against a fresh logit fit on io_analytic.
Mirrors the model in make_regression.py.
"""
import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Numbers from the forest plot (variable label -> (OR, lo, hi))
FOREST = {
    'Age at death (per year)':            (1.01, 1.00, 1.02),
    'Female vs male':                     (0.95, 0.78, 1.15),
    'Black vs White':                     (0.76, 0.51, 1.14),
    'Hispanic vs White':                  (0.73, 0.30, 1.75),
    'Other/Unknown vs White':             (0.99, 0.70, 1.39),
    'Dual eligible (Medicaid)':           (0.65, 0.50, 0.83),
    'Non-metro vs metro':                 (0.99, 0.80, 1.21),
    'Unknown rurality vs metro':          (0.91, 0.42, 1.96),
    'Northeast vs South':                 (0.70, 0.56, 0.88),
    'Midwest vs South':                   (0.79, 0.63, 0.99),
    'West vs South':                      (0.74, 0.58, 0.95),
    'Unknown region vs South':            (0.79, 0.13, 4.98),
    'Larynx vs hypopharynx':              (1.42, 0.99, 2.05),
    'Oral cavity vs hypopharynx':         (1.37, 0.96, 1.94),
    'Oropharynx vs hypopharynx':          (1.34, 0.95, 1.89),
    'Both ICIs vs pembrolizumab':         (0.98, 0.57, 1.69),
    'Nivolumab vs pembrolizumab':         (1.06, 0.84, 1.32),
    'Chemo-ICI vs ICI monotherapy':       (0.81, 0.66, 1.00),
    'ICI administrations in final episode':(0.99, 0.96, 1.01),
    'van Walraven score (per unit)':      (1.00, 0.99, 1.01),
    'Surgery vs radiation':               (0.91, 0.77, 1.08),
    'Calendar year (per year from 2017)': (1.01, 0.95, 1.07),
}

con = duckdb.connect(r'F:\CMS\cms_data.duckdb', read_only=True)
df = con.execute("""
    SELECT hospice_enrolled, in_hospital_death,
           age_at_death, sex, race,
           dual_eligible, urban_rural, census_region,
           subsite_category, io_agent, io_regimen,
           last_episode_doses, van_walraven_score,
           primary_curative_type, death_year
    FROM io_analytic
""").df()
con.close()

df['hospice_enrolled']   = df['hospice_enrolled'].fillna(0).astype(int)
df['in_hospital_death']  = df['in_hospital_death'].fillna(0).astype(int)
df['dual_eligible']      = df['dual_eligible'].fillna(0).astype(int)
df['van_walraven_score'] = pd.to_numeric(df['van_walraven_score'], errors='coerce').fillna(0)
df['urban_rural']        = df['urban_rural'].fillna('Unknown')
df['census_region']      = df['census_region'].fillna('Unknown')
df['race_collapsed']     = df['race'].replace({
    'Asian/PI': 'Other/Unknown', 'Native American': 'Other/Unknown',
    'Other': 'Other/Unknown', 'Unknown': 'Other/Unknown',
})
df['io_regimen'] = df['io_regimen'].replace(
    {'IO monotherapy': 'ICI monotherapy', 'chemo-IO': 'chemo-ICI'})

cat_specs = {
    'sex':                ('Male',           ['Male', 'Female']),
    'race_collapsed':     ('White',          ['White', 'Black', 'Hispanic', 'Other/Unknown']),
    'urban_rural':        ('Metro',          ['Metro', 'Non-metro', 'Unknown']),
    'census_region':      ('South',          ['Northeast', 'Midwest', 'South', 'West', 'Unknown']),
    'subsite_category':   ('Hypopharynx',    ['Hypopharynx', 'Larynx', 'Oral Cavity', 'Oropharynx']),
    'io_agent':           ('pembrolizumab',  None),
    'io_regimen':         ('ICI monotherapy',['ICI monotherapy', 'chemo-ICI']),
    'primary_curative_type': ('radiation',   ['radiation', 'surgery']),
}
for col, (ref, order) in cat_specs.items():
    if order is None:
        df[col] = pd.Categorical(df[col])
    else:
        exist = [o for o in order if o in df[col].unique()]
        df[col] = pd.Categorical(df[col], categories=exist)
    if ref in df[col].cat.categories:
        cats = [ref] + [c for c in df[col].cat.categories if c != ref]
        df[col] = df[col].cat.reorder_categories(cats)

df['age_at_death']       = pd.to_numeric(df['age_at_death'], errors='coerce')
df['last_episode_doses'] = pd.to_numeric(df['last_episode_doses'], errors='coerce')
df['death_year_c']       = df['death_year'].astype(float) - 2017

predictors = ['age_at_death', 'sex', 'race_collapsed', 'dual_eligible', 'urban_rural',
              'census_region', 'subsite_category', 'io_agent', 'io_regimen',
              'last_episode_doses', 'van_walraven_score',
              'primary_curative_type', 'death_year_c']
df_model = df[predictors + ['hospice_enrolled']].dropna()

formula = (
    'hospice_enrolled ~ age_at_death + C(sex) + C(race_collapsed) + dual_eligible '
    '+ C(urban_rural) + C(census_region) + C(subsite_category) '
    '+ C(io_agent) + C(io_regimen) '
    '+ last_episode_doses + van_walraven_score + C(primary_curative_type) '
    '+ death_year_c'
)
model = smf.logit(formula, data=df_model).fit(method='newton', maxiter=500, disp=False)

# Build term -> readable-label map matching the forest plot
TERM_LABEL = {
    'age_at_death':                                'Age at death (per year)',
    'C(sex)[T.Female]':                            'Female vs male',
    'C(race_collapsed)[T.Black]':                  'Black vs White',
    'C(race_collapsed)[T.Hispanic]':               'Hispanic vs White',
    'C(race_collapsed)[T.Other/Unknown]':          'Other/Unknown vs White',
    'dual_eligible':                               'Dual eligible (Medicaid)',
    'C(urban_rural)[T.Non-metro]':                 'Non-metro vs metro',
    'C(urban_rural)[T.Unknown]':                   'Unknown rurality vs metro',
    'C(census_region)[T.Northeast]':               'Northeast vs South',
    'C(census_region)[T.Midwest]':                 'Midwest vs South',
    'C(census_region)[T.West]':                    'West vs South',
    'C(census_region)[T.Unknown]':                 'Unknown region vs South',
    'C(subsite_category)[T.Larynx]':               'Larynx vs hypopharynx',
    'C(subsite_category)[T.Oral Cavity]':          'Oral cavity vs hypopharynx',
    'C(subsite_category)[T.Oropharynx]':           'Oropharynx vs hypopharynx',
    'C(io_agent)[T.both]':                         'Both ICIs vs pembrolizumab',
    'C(io_agent)[T.nivolumab]':                    'Nivolumab vs pembrolizumab',
    'C(io_regimen)[T.chemo-ICI]':                  'Chemo-ICI vs ICI monotherapy',
    'last_episode_doses':                          'ICI administrations in final episode',
    'van_walraven_score':                          'van Walraven score (per unit)',
    'C(primary_curative_type)[T.surgery]':         'Surgery vs radiation',
    'death_year_c':                                'Calendar year (per year from 2017)',
}

coef = model.params
ci   = model.conf_int()

print(f"{'Variable':<42} {'Fit OR':>8} {'Fit Lo':>8} {'Fit Hi':>8} {'Forest OR':>10} {'Forest Lo':>10} {'Forest Hi':>10}  Match?")
print('-' * 110)
all_match = True
for term, label in TERM_LABEL.items():
    if term not in coef.index:
        print(f"{label:<42}   MISSING from fit")
        all_match = False
        continue
    fit_or  = float(np.exp(coef[term]))
    fit_lo  = float(np.exp(ci.loc[term, 0]))
    fit_hi  = float(np.exp(ci.loc[term, 1]))
    f_or, f_lo, f_hi = FOREST[label]
    # Match within rounding to 2 decimals
    ok = abs(round(fit_or, 2) - f_or) <= 0.01 and \
         abs(round(fit_lo, 2) - f_lo) <= 0.01 and \
         abs(round(fit_hi, 2) - f_hi) <= 0.01
    mark = 'OK' if ok else 'MISMATCH'
    if not ok:
        all_match = False
    print(f"{label:<42} {fit_or:>8.2f} {fit_lo:>8.2f} {fit_hi:>8.2f} {f_or:>10.2f} {f_lo:>10.2f} {f_hi:>10.2f}  {mark}")
print()
print('ALL MATCH' if all_match else 'SOME MISMATCHES — see above')
