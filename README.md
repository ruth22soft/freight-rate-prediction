# Freight Rate Prediction

Predicts truckload freight rates (USD per load) from lane, distance, equipment,
weight, date and market signals. Built for the Spotter ML Engineer assessment.

## Results (time-based holdout: Sep–Oct 2025)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Baseline: median rate-per-mile by equipment × distance | $175 | $258 | 8.1% |
| Baseline: quote_signal × distance | $193 | $344 | 9.9% |
| **LightGBM (final)** | **$82** | **$118** | **3.2%** |

## Project Structure

```
data/                               # put the provided CSVs here (not committed)
src/
  inspect_data.py                   # first look at every file
  eda_checks.py                     # data-quality and trend checks
  pipeline.py                       # cleaning, features, validation, final predictions
score.py                            # provided scorer
validation_predictions.csv          # final submission (load_id, predicted_rate)
outputs/
  december_chart_predictions.csv    # filled December chart inputs
  metrics.json                      # holdout results
  feature_importance.csv
  removed_rate_outliers.csv         # rows dropped from training, for transparency
scorer_results/candidate_december.png
report/report.pdf
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Place the provided CSV files in `data/`. File names used:
`train-test.csv`, `validation.csv`, `validation-predictions-template.csv`,
`december-chart-inputs.csv`.

## How to Run

```bash
python src/pipeline.py
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv
```

The first command writes `validation_predictions.csv` and the December predictions.
The second validates both files and creates `scorer_results/candidate_december.png`.

## Approach

**Validation split.** Development data covers Jan–Oct 2025 and the validation set
covers Nov–Dec 2025, so this is a forecasting problem. A random split would leak
future market conditions into training. I trained on Jan–Aug and tested on Sep–Oct,
a two-month holdout that mirrors the two-month prediction horizon. The final model
was retrained on all Jan–Oct data.

**Data-quality fixes** (thresholds learned from training data only):
- 292 negative weights in train (145 in validation): sign error, so the absolute value is used
- 300 / 165 missing weights: filled with the training median, plus a missing flag
- 374 / 249 missing market_index: filled with that day's median, plus a missing flag
- 214 / 55 impossible distances (more than 1.5× straight-line distance): replaced with
  straight-line distance × 1.18, the typical road-to-straight-line ratio
- 697 training rows (1.45%) with implausible rate per mile (below 0.5× or above 2× similar
  loads) were removed from training only; validation rows are never dropped

**Features.** Distance, straight-line miles, pickup and delivery coordinates, direction
of travel, equipment, weight, market_index, quote_signal, day of week, and missing flags.
City names are not used because 8 validation cities (1,447 rows) never appear in
training; coordinates generalise to any city. Month is not used because Nov–Dec are
outside the training range and tree models cannot extrapolate to unseen months.

**Model.** LightGBM on log(rate), with early stopping on the holdout. Gradient boosting
handles non-linear distance effects and feature interactions, needs no scaling, and
handles missing values. The log target keeps errors proportional across short
and long hauls.

**December chart.** The December inputs contain no coordinates or market signals.
Coordinates were looked up from the city names; market_index and quote_signal use
the daily median of real Dry Van loads in validation.csv on each date. Predictions
range from $808 to $818, consistent with the $808 historical median for Dry Van
loads on this lane.

**Limitation.** Training data contains no Nov–Dec, so holiday seasonality cannot be
learned directly; it is only reflected through the market inputs.