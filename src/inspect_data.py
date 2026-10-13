import pandas as pd
import glob

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

for f in sorted(glob.glob("data/*.csv")):
    df = pd.read_csv(f)
    print("=" * 80)
    print(f, "shape:", df.shape)
    print(df.head(3).to_string())
    print("\nDTYPES:\n", df.dtypes.to_string())
    print("\nMISSING:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    num = df.select_dtypes("number")
    if not num.empty:
        print("\nNUMERIC SUMMARY:\n", num.describe().T.to_string())
    for c in df.select_dtypes("object").columns:
        print(f"\n{c}: {df[c].nunique()} unique -> {df[c].value_counts().head(5).to_dict()}")