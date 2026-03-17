import sys
sys.path.insert(0, r'C:\users\hsaee\desktop\cms_viewer\env\Lib\site-packages')
import duckdb
con = duckdb.connect(r'F:\CMS\cms_data.duckdb', read_only=True)
print("stroke_propensity columns:")
print(con.execute("DESCRIBE stroke_propensity").df().to_string())
con.close()
