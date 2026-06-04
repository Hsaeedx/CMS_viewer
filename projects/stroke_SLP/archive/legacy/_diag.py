import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
import duckdb, numpy as np, pandas as pd

DB_PATH = Path(os.getenv('duckdb_database'))
con = duckdb.connect(str(DB_PATH), read_only=True)
df = con.execute("""
    SELECT p.DSYSRTKY, p.slp_timing_group, p.days_to_slp_outpt,
           p.dysphagia_poa, p.psm_matched_A,
           o.days_to_aspiration, o.days_to_death
    FROM stroke_propensity p
    JOIN stroke_outcomes o ON o.DSYSRTKY = p.DSYSRTKY
    WHERE p.psm_matched_A = TRUE
""").df()
con.close()

print('Comp A cohort:', len(df))
print()
print('days_to_slp_outpt == 0:', (df.days_to_slp_outpt == 0).sum())
print('days_to_slp_outpt == 1:', (df.days_to_slp_outpt == 1).sum())
print()
print('days_to_slp_outpt distribution by group:')
print(df.groupby('slp_timing_group')['days_to_slp_outpt'].describe().round(1))
print()
print('dysphagia_poa by group:')
print(df.groupby('slp_timing_group')['dysphagia_poa'].mean().round(3))
print()
has_event = df['days_to_aspiration'].notna() & (df['days_to_aspiration'] <= 365)
print('Aspiration events:', has_event.sum())
print('Dysphagia among events:', df.loc[has_event, 'dysphagia_poa'].mean().round(3))
print('Dysphagia among non-events:', df.loc[~has_event, 'dysphagia_poa'].mean().round(3))
