import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import duckdb

DB_PATH = Path(os.getenv("duckdb_database", "cms_data.duckdb"))

con = duckdb.connect(str(DB_PATH), read_only=True)
print("stroke_propensity columns:")
print(con.execute("DESCRIBE stroke_propensity").df().to_string())
con.close()
