
import pandas as pd
from pathlib import Path

file_path = Path("data/Instagram_Analytics.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Check that the CSV is inside the data folder."
    )

df = pd.read_csv(file_path)

print("\nSOCIAL MEDIA ENGAGEMENT ANALYSIS")
print("=" * 40)

print("\n1. Dataset dimensions")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. Column names")
print(df.columns.tolist())

print("\n3. First five records")
print(df.head().to_string())

print("\n4. Data types")
print(df.dtypes.to_string())

print("\n5. Missing values")
print(df.isnull().sum().to_string())

print("\n6. Duplicate rows")
print(df.duplicated().sum())

print("\n7. Performance category counts")
if "performance_bucket_label" in df.columns:
    print(df["performance_bucket_label"].value_counts().to_string())

print("\n8. Numerical summary")
print(df.describe().to_string())

print("\n9. Unique accounts and repeated follower counts")
if "account_id" in df.columns:
    print("Unique accounts:", df["account_id"].nunique())

if "follower_count" in df.columns:
    print("Unique follower counts:", df["follower_count"].nunique())

print("\nDataset inspection completed.")
