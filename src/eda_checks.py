import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

tr = pd.read_csv("data/train-test.csv", parse_dates=["date"])
va = pd.read_csv("data/validation.csv", parse_dates=["date"])

print("TRAIN dates:", tr.date.min().date(), "->", tr.date.max().date())
print("VALID dates:", va.date.min().date(), "->", va.date.max().date())

# --- Weight problems ---
print("\nNegative weights  train:", (tr.weight < 0).sum(), " valid:", (va.weight < 0).sum())
print("Zero weights      train:", (tr.weight == 0).sum(), " valid:", (va.weight == 0).sum())
print("Small weights (<1000) train:", ((tr.weight > 0) & (tr.weight < 1000)).sum())

# --- Rate per mile (RPM) outliers ---
tr["rpm"] = tr.posted_rate / tr.distance
print("\nRATE PER MILE:\n", tr.rpm.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).to_string())
show = ["load_id", "pickup", "delivery", "distance", "equipment", "weight", "date",
        "market_index", "quote_signal", "posted_rate", "rpm"]
print("\nLOWEST RPM ROWS:\n", tr.nsmallest(8, "rpm")[show].to_string())
print("\nHIGHEST RPM ROWS:\n", tr.nlargest(8, "rpm")[show].to_string())

# --- Unseen cities ---
train_cities = set(tr.pickup) | set(tr.delivery)
new_cities = (set(va.pickup) | set(va.delivery)) - train_cities
print("\nCities in validation but NOT in train:", sorted(new_cities))
print("Validation rows touching an unseen city:",
      (va.pickup.isin(new_cities) | va.delivery.isin(new_cities)).sum())

# --- City coordinate consistency ---
for df, name in [(tr, "train"), (va, "valid")]:
    c = pd.concat([
        df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates()
    n = c.groupby("city").size()
    print(f"{name}: cities with more than one coordinate ->", n[n > 1].to_dict())

print("\nPickup == delivery  train:", (tr.pickup == tr.delivery).sum(), " valid:", (va.pickup == va.delivery).sum())

# --- Duplicates ignoring load_id ---
cols = [c for c in tr.columns if c not in ("load_id", "rpm")]
print("Duplicate rows ignoring load_id:", tr.duplicated(cols).sum())

# --- Distance vs straight-line distance ---
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))

hv = haversine(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
print("\nDISTANCE / straight-line ratio:\n", (tr.distance / hv.replace(0, np.nan)).describe().to_string())

# --- Monthly trends ---
tr["month"] = tr.date.dt.to_period("M")
va["month"] = va.date.dt.to_period("M")
print("\nTRAIN BY MONTH:\n", tr.groupby("month").agg(
    loads=("load_id", "size"), median_rate=("posted_rate", "median"), median_rpm=("rpm", "median"),
    market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean")).round(3).to_string())
print("\nVALID BY MONTH:\n", va.groupby("month").agg(
    loads=("load_id", "size"), market_index=("market_index", "mean"),
    quote_signal=("quote_signal", "mean")).round(3).to_string())

# --- Is market_index the same for all loads on one day? ---
print("\nAvg distinct market_index values per date  train:",
      round(tr.groupby("date").market_index.nunique().mean(), 2),
      " valid:", round(va.groupby("date").market_index.nunique().mean(), 2))

# --- What drives rate per mile ---
print("\nCORRELATION WITH RPM:\n",
      tr[["rpm", "posted_rate", "distance", "weight", "market_index", "quote_signal"]].corr()["rpm"].round(3).to_string())
print("\nMEDIAN RPM BY EQUIPMENT:\n", tr.groupby("equipment").rpm.median().round(3).to_string())

# --- The December chart lane in train ---
lane = tr[(tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")]
print(f"\nLexington -> Fort Wayne loads in train: {len(lane)}")
if len(lane):
    print(lane.groupby("equipment").agg(n=("load_id", "size"), distance=("distance", "median"),
          median_rate=("posted_rate", "median")).to_string())