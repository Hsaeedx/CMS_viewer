import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import duckdb

con = duckdb.connect(r'F:\CMS\cms_data.duckdb')
con.execute("SET memory_limit='24GB'; SET threads=12;")

views = con.execute("SELECT view_name, sql FROM duckdb_views() WHERE sql LIKE '%E:\\%'").fetchall()
print(f'Fixing {len(views)} views...')

for name, sql in views:
    new_sql = sql.replace('E:\\CMS\\', 'F:\\CMS\\')
    new_sql = new_sql.replace('CREATE VIEW ', 'CREATE OR REPLACE VIEW ', 1)
    con.execute(new_sql)
    print(f'  Fixed: {name}')

remaining = con.execute("SELECT COUNT(*) FROM duckdb_views() WHERE sql LIKE '%E:\\%'").fetchone()[0]
print(f'Done. Views still referencing E:\\: {remaining}')
con.close()
