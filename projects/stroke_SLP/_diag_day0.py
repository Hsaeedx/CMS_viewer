import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
import duckdb, pandas as pd, numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu

DB_PATH = Path(os.getenv('duckdb_database'))
con = duckdb.connect(str(DB_PATH), read_only=True)
df = con.execute("""
    SELECT p.DSYSRTKY, p.days_to_slp_outpt,
           p.age_at_adm, p.sex, p.stroke_type,
           p.index_los, p.van_walraven_score,
           p.dysphagia_poa, p.aspiration_poa,
           p.mech_vent, p.peg_placed, p.trach_placed,
           p.prior_stroke, p.dementia, p.afib, p.hypertension,
           p.dschg_group,
           o.days_to_death, o.days_to_aspiration, o.days_to_gtube
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
    WHERE p.slp_timing_group = '0-14d'
""").df()
con.close()

g0 = df[df['days_to_slp_outpt'] == 0]
g1 = df[df['days_to_slp_outpt'] == 1]
print(f"Day-0: {len(g0):,}  |  Day-1: {len(g1):,}\n")

def safe_chi2(a, b):
    try:
        _, p, _, _ = chi2_contingency([[a[0],a[1]],[b[0],b[1]]])
        return f"{p:.4f}"
    except:
        return "—"

rows = []
def cont(label, col):
    m0, m1 = g0[col].median(), g1[col].median()
    _, p = mannwhitneyu(g0[col].dropna(), g1[col].dropna(), alternative='two-sided')
    rows.append([label, f"{m0:.1f}", f"{m1:.1f}", f"{p:.4f}"])

def binary(label, col=None, fn0=None, fn1=None):
    if col is not None:
        v0, v1 = g0[col].mean()*100, g1[col].mean()*100
        a = [int(g0[col].sum()), int(len(g0)-g0[col].sum())]
        b = [int(g1[col].sum()), int(len(g1)-g1[col].sum())]
    else:
        v0, v1 = fn0(g0)*100, fn1(g1)*100
        a = [int(fn0(g0)*len(g0)), int((1-fn0(g0))*len(g0))]
        b = [int(fn1(g1)*len(g1)), int((1-fn1(g1))*len(g1))]
    rows.append([label, f"{v0:.1f}%", f"{v1:.1f}%", safe_chi2(a, b)])

def outcome(label, col, cutoff=365):
    e0 = (g0[col].notna() & (g0[col] <= cutoff))
    e1 = (g1[col].notna() & (g1[col] <= cutoff))
    v0, v1 = e0.mean()*100, e1.mean()*100
    a = [int(e0.sum()), int(len(e0)-e0.sum())]
    b = [int(e1.sum()), int(len(e1)-e1.sum())]
    rows.append([label, f"{v0:.1f}%", f"{v1:.1f}%", safe_chi2(a, b)])

print("--- DEMOGRAPHICS ---------------------------------------------------")
cont  ('Age, median (yrs)',           'age_at_adm')
binary('Female',    fn0=lambda x:(x['sex']=='Female').mean(), fn1=lambda x:(x['sex']=='Female').mean())
binary('Ischemic',  fn0=lambda x:(x['stroke_type']=='Ischemic').mean(), fn1=lambda x:(x['stroke_type']=='Ischemic').mean())
binary('ICH',       fn0=lambda x:(x['stroke_type']=='ICH').mean(), fn1=lambda x:(x['stroke_type']=='ICH').mean())
binary('SAH',       fn0=lambda x:(x['stroke_type']=='SAH').mean(), fn1=lambda x:(x['stroke_type']=='SAH').mean())

print("--- HOSPITAL COURSE --------------------------------")
cont  ('Index LOS, median (days)',    'index_los')
cont  ('Van Walraven score, median',  'van_walraven_score')
binary('Mech ventilation',            'mech_vent')
binary('PEG placed',                  'peg_placed')
binary('Tracheostomy',                'trach_placed')

print("--- SWALLOWING -------------------------------------")
binary('Dysphagia POA',               'dysphagia_poa')
binary('Aspiration POA',              'aspiration_poa')

print("--- COMORBIDITIES -----------------------------------")
binary('Prior stroke',                'prior_stroke')
binary('Dementia',                    'dementia')
binary('Afib',                        'afib')
binary('Hypertension',                'hypertension')

print("--- DISCHARGE DESTINATION ---------------------------")
binary('Home (codes 01/07)',  fn0=lambda x:(x['dschg_group']=='home').mean(), fn1=lambda x:(x['dschg_group']=='home').mean())
binary('Home+HHA (code 06)', fn0=lambda x:(x['dschg_group']=='hha').mean(),  fn1=lambda x:(x['dschg_group']=='hha').mean())

print("--- OUTCOMES (365 days) ----------------------------")
outcome('Aspiration PNA',   'days_to_aspiration')
outcome('G-tube',           'days_to_gtube')
outcome('Mortality',        'days_to_death')

print()
hdr = f"{'Variable':<35} {'Day-0':>10} {'Day-1':>10} {'p':>8}"
print(hdr)
print("-"*67)
for r in rows:
    print(f"{r[0]:<35} {r[1]:>10} {r[2]:>10} {r[3]:>8}")
