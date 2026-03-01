import duckdb, sys

db_path = sys.argv[1]
keyword = sys.argv[2].lower()

con = duckdb.connect(db_path)

tables = con.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
      AND LOWER(table_name) LIKE ?
""", [f"%{keyword}%"]).fetchall()

for schema, name in tables:
    con.execute(f'DROP TABLE IF EXISTS "{schema}"."{name}"')

print(f"Dropped {len(tables)} tables containing '{keyword}'")
con.close()
