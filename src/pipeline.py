"""
Freight rate prediction pipeline.

  1. Load data
  2. Clean data-quality issues
  3. Build features
  4. Time-based validation: train Jan-Aug, test Sep-Oct (mimics predicting Nov-Dec)
  5. Retrain on all Jan-Oct data, predict validation.csv and the December chart inputs
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
HOLDOUT_START = pd.Timestamp("2025-09-01")
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
SEED = 42

FEATURES = [
    "distance", "straight_miles",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "lat_change", "lon_change",
    "equipment", "weight",
    "market_index", "quote_signal",
    "day_of_week", "is_weekend",
    "weight_was_missing", "market_index_was_missing",
]

PARAMS = dict(
    n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, verbose=-1,
)


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def straight_line(df):
    return haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])


# ------------------------------------------------------------------ cleaning
def learn_reference(train):
    """Values learned from the training data only, reused for every dataset."""
    ratio = train["distance"] / straight_line(train)
    normal = ratio[(ratio >= 1.0) & (ratio <= 1.5)]
    return {
        "weight_median": float(train["weight"].abs().median()),
        "market_index_median": float(train["market_index"].median()),
        "detour": float(normal.median()),
        "max_detour": 1.5,
    }


def clean(df, ref, name):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 1. Negative weights are a sign error -> take absolute value
    neg = df["weight"] < 0
    df.loc[neg, "weight"] = df.loc[neg, "weight"].abs()
    # 2. Missing weight -> training median (+ flag)
    df["weight_was_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].fillna(ref["weight_median"])
    # 3. Missing market_index -> median of the same day (+ flag)
    df["market_index_was_missing"] = df["market_index"].isna().astype(int)
    daily = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(daily).fillna(ref["market_index_median"])

    # 4. Distance check: high road/straight-line ratios are short trips (routing detours and
    #    a 70-mile minimum), not errors, so recorded distances are kept and only reported.
    ratio = df["distance"] / straight_line(df)
    high_ratio = int(((ratio > ref["max_detour"]) | (ratio < 1.0)).sum())

    print(f"[clean {name}] negative weights fixed: {int(neg.sum())} | "
          f"missing weight filled: {int(df['weight_was_missing'].sum())} | "
          f"missing market_index filled: {int(df['market_index_was_missing'].sum())} | "
          f"short trips with high distance ratio (kept): {high_ratio}")
    return df


def drop_rate_outliers(df):
    """Remove rows whose rate-per-mile is far from similar loads (training data only)."""
    rpm = df["posted_rate"] / df["distance"]
    bucket = pd.qcut(df["distance"], 10, labels=False, duplicates="drop")
    typical = rpm.groupby([bucket, df["equipment"]]).transform("median")
    ratio = rpm / typical
    bad = (ratio < 0.5) | (ratio > 2.0)
    df.loc[bad].assign(rate_per_mile=rpm[bad]).to_csv(OUT_DIR / "removed_rate_outliers.csv", index=False)
    print(f"[outliers] removed {int(bad.sum())} rows ({bad.mean():.2%}) with implausible rate per mile")
    return df.loc[~bad].copy()


# ------------------------------------------------------------------ features
def add_features(df):
    df = df.copy()
    df["straight_miles"] = straight_line(df)
    df["lat_change"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_change"] = df["delivery_lon"] - df["pickup_lon"]
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    return df


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    return {
        "MAE": round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "MAPE_%": round(float(np.mean(np.abs(err) / y_true) * 100), 2),
    }


# ------------------------------------------------------------------ December chart
def build_december(raw_dec, raw_train, raw_valid, valid):
    coords = pd.concat([
        df[[f"{s}", f"{s}_lat", f"{s}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for df in (raw_train, raw_valid) for s in ("pickup", "delivery")
    ]).drop_duplicates("city").set_index("city")
    d = raw_dec.copy()
    d["date"] = pd.to_datetime(d["date"])
    for s in ("pickup", "delivery"):
        d[f"{s}_lat"] = d[s].map(coords["lat"])
        d[f"{s}_lon"] = d[s].map(coords["lon"])
    # Market inputs are not given for December -> daily median of real Dry Van loads on that date
    dry_van = valid[valid["equipment"] == "Dry Van"]
    daily = dry_van.groupby("date")[["market_index", "quote_signal"]].median()
    d = d.drop(columns=["predicted_rate"]).join(daily, on="date")
    d["weight_was_missing"] = 0
    d["market_index_was_missing"] = 0
    d = add_features(d)
    missing = d[FEATURES].isna().sum()
    assert missing.sum() == 0, f"December features missing: {missing[missing > 0].to_dict()}"
    return d


# ------------------------------------------------------------------ main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA_DIR / "train-test.csv")
    raw_valid = pd.read_csv(DATA_DIR / "validation.csv")
    raw_dec = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")
    template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")

    ref = learn_reference(raw_train)
    print(f"[reference] typical road/straight-line ratio: {ref['detour']:.3f}")

    train = add_features(drop_rate_outliers(clean(raw_train, ref, "train-test")))
    valid = add_features(clean(raw_valid, ref, "validation"))

    # ---- Time-based split
    fit = train[train["date"] < HOLDOUT_START]
    hold = train[train["date"] >= HOLDOUT_START]
    print(f"[split] train Jan-Aug: {len(fit):,} rows | holdout Sep-Oct: {len(hold):,} rows")

    # ---- Baselines
    eq_rpm = (fit["posted_rate"] / fit["distance"]).groupby(fit["equipment"].astype(str)).median()
    base_equipment = hold["equipment"].astype(str).map(eq_rpm) * hold["distance"]
    base_quote = hold["quote_signal"] * hold["distance"]

    # ---- LightGBM on log(rate) with early stopping on the holdout
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        fit[FEATURES], np.log(fit["posted_rate"]),
        eval_set=[(hold[FEATURES], np.log(hold["posted_rate"]))],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    best_iter = model.best_iteration_
    hold_pred = np.exp(model.predict(hold[FEATURES], num_iteration=best_iter))

    results = {
        "baseline_median_rpm_by_equipment": metrics(hold["posted_rate"], base_equipment),
        "baseline_quote_signal_x_distance": metrics(hold["posted_rate"], base_quote),
        "lightgbm": metrics(hold["posted_rate"], hold_pred),
    }
    by_month = {
        str(m): metrics(g["posted_rate"], hold_pred[hold["date"].dt.month.values == m])
        for m, g in hold.groupby(hold["date"].dt.month)
    }
    print("\n[holdout Sep-Oct results]")
    for name, r in results.items():
        print(f"  {name:36s} {r}")
    print("  LightGBM by month:", by_month)
    print(f"  best iteration: {best_iter}")

    # ---- Final model on all Jan-Oct data
    final = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": int(best_iter * 1.1)})
    final.fit(train[FEATURES], np.log(train["posted_rate"]))

    importance = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    importance = (importance / importance.sum() * 100).sort_values(ascending=False).round(2)
    importance.to_csv(OUT_DIR / "feature_importance.csv", header=["gain_%"])
    print("\n[feature importance, % of gain]\n" + importance.to_string())

    # ---- Validation predictions
    valid_pred = pd.DataFrame({
        "load_id": valid["load_id"],
        "predicted_rate": np.round(np.exp(final.predict(valid[FEATURES])), 2),
    })
    submission = template[["load_id"]].merge(valid_pred, on="load_id", how="left")
    assert len(submission) == 12_000 and submission["predicted_rate"].notna().all()
    assert (submission["predicted_rate"] > 0).all()
    submission.to_csv("validation_predictions.csv", index=False)
    print(f"\n[output] validation_predictions.csv written: {len(submission):,} rows, "
          f"median ${submission['predicted_rate'].median():,.0f}, min ${submission['predicted_rate'].min():,.0f}")

    # ---- December chart predictions
    dec_features = build_december(raw_dec, raw_train, raw_valid, valid)
    dec_out = raw_dec.copy()
    dec_out["predicted_rate"] = np.round(np.exp(final.predict(dec_features[FEATURES])), 2)
    dec_out.to_csv(OUT_DIR / "december_chart_predictions.csv", index=False)
    print(f"[output] December predictions: min ${dec_out['predicted_rate'].min():,.0f}, "
          f"max ${dec_out['predicted_rate'].max():,.0f}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"holdout": results, "holdout_by_month": by_month, "best_iteration": best_iter,
                   "reference": ref}, f, indent=2)


if __name__ == "__main__":
    main()
