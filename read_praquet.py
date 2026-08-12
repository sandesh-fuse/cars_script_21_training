import pandas as pd


praquet_file = "artifacts/script21/sample_test_rows.parquet"

df = pd.read_parquet(praquet_file)

df.to_csv("artifacts/script21/sample_test_rows.csv", index=False)
