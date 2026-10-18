import os
import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Metrics calculations before full clean
raw_rows = len(raw)
missing_dist = raw["distance_km"].isna().sum()
missing_weight = raw["parcel_weight_kg"].isna().sum()
missing_val = raw["declared_value"].isna().sum()
missing_pay = raw["payment_mode"].isna().sum()
sentinel_count = (raw["parcel_weight_kg"] > 100).sum()

no_dupes = raw.drop_duplicates()
dupes_removed = raw_rows - len(no_dupes)
spellings_before = no_dupes["service_type"].nunique()

clean_df, medians = clean_shipments(raw)
spellings_after = clean_df["service_type"].nunique()

model_df = build_features(clean_df, hubs)
if "delivery_hours" in model_df.columns:
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

model_df.to_csv("data/modelling_table.csv", index=False)

# Checks
nulls_total = model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
q1 = model_df["delivery_hours"].quantile(0.25)
q3 = model_df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ((model_df["delivery_hours"] < lower_bound) | (model_df["delivery_hours"] > upper_bound)).sum()

metrics = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": dupes_removed,
    "rows_after_dedup": len(no_dupes),
    "missing_distance_km_before": missing_dist,
    "missing_parcel_weight_before": missing_weight,
    "sentinel_weight_rows": sentinel_count,
    "missing_declared_value_before": missing_val,
    "missing_payment_mode_before": missing_pay,
    "service_type_spellings_before": spellings_before,
    "service_type_spellings_after": spellings_after,
    "declared_value_mean_inr": round(model_df["declared_value_inr"].mean(), 4),
    "declared_value_median_inr": medians["declared_value_inr"],
    "log_declared_value_mean": round(model_df["log_declared_value"].mean(), 4),
    "log_declared_value_median": round(model_df["log_declared_value"].median(), 4),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(model_df),
    "nulls_in_features_and_target": nulls_total,
    "delayed_count": model_df["is_delayed"].sum(),
    "delay_rate": round(model_df["is_delayed"].mean(), 4),
    "hours_q1": round(q1, 4),
    "hours_q3": round(q3, 4),
    "hours_lower_bound": round(lower_bound, 4),
    "hours_upper_bound": round(upper_bound, 4),
    "hours_outlier_count": outliers
}
pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

# Delay by group
model_df["distance_band"] = pd.cut(model_df["distance_km"], bins=[0, 200, 400, 600, 2000], labels=["0-200", "201-400", "401-600", "601+"])
groups = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grp = model_df.groupby(dim, observed=False).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index().rename(columns={dim: "group"})
    grp.insert(0, "dimension", dim)
    grp["delay_rate"] = (grp["delayed"] / grp["shipments"]).round(4)
    grp["mean_hours"] = grp["mean_hours"].round(4)
    groups.append(grp)
pd.concat(groups).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Correlations
numeric_cols = model_df.select_dtypes(include=[np.number])
corr = numeric_cols.corr().round(4)
corr.reset_index().rename(columns={"index": "feature"}).to_csv("outputs/part_a_correlations.csv", index=False)
