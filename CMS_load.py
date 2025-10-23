from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import sys

def csv_to_parquet(csv_file_path, parquet_file_path, chunksize=250_000):
    writer = None
    schema = None
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize, dtype=str, low_memory=False):
        chunk = chunk.fillna("")  # avoid "all NA" => null type
        if schema is None:
            fields = [pa.field(col, pa.string()) for col in chunk.columns]
            schema = pa.schema(fields)
            writer = pq.ParquetWriter(parquet_file_path, schema)
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        writer.write_table(table)
        print(f"Wrote chunk of size {len(chunk)} to {parquet_file_path}")
    if writer:
        writer.close()

if __name__ == "__main__":
    cms_dir = Path("/Volumes/EPIC SSDDDD/CMS")

    if len(sys.argv) > 1:
        cms_dir = cms_dir / sys.argv[1]
        # csv_to_parquet(cms_dir / f"mbsf_lds_100_{sys.argv[1]}.csv", cms_dir / f"mbsf_{sys.argv[1]}.parquet")
        # csv_to_parquet(cms_dir / f"inp_claimsk_{sys.argv[1]}.csv", cms_dir / f"inp_{sys.argv[1]}.parquet")
        # csv_to_parquet(cms_dir / f"OUT_claimsk_{sys.argv[1]}.csv", cms_dir / f"out_{sys.argv[1]}.parquet")
        csv_to_parquet(cms_dir / f"inp_instcond_lds_100_{sys.argv[1]}.csv", cms_dir / f"inplink_{sys.argv[1]}.parquet")
    else:
        print("Usage: python load_CMS.py YEAR")
        print("E.g.: python load_CMS.py 2019")
        sys.exit(1)