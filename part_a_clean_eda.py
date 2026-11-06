
import os
import pandas as pd
import numpy as np

from src.shipment_prep import clean_shipments, build_features, FEATURES

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Read original data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Remove duplicates for before-cleaning checks
raw_dedup = raw.drop_duplicates().copy()

# Calculate values before cleaning and filling
raw_rows = len(raw)
duplicate_rows_removed = raw_rows - len(raw_dedup)
missing_distance_before = raw_dedup["distance_km"].isna().sum()
missing_weight_before = raw_dedup["parcel_weight_kg"].isna().sum()
missing_value_before = raw_dedup["declared_value"].isna().sum()
missing_payment_before = raw_dedup["payment_mode"].isna().sum()
sentinel_rows = (raw_dedup["parcel_weight_kg"] > 100).sum()
spellings_before = raw_dedup["service_type"].nunique()

# Clean the data
clean, medians = clean_shipments(raw)

# Count service names after cleaning
spellings_after = clean["service_type"].nunique()

# Build features and add target
model_table = build_features(clean, hubs)

model_table["is_delayed"] = (
    model_table["delivery_hours"] > model_table["promised_hours"]
).astype(int)

# Save modelling table
model_table.to_csv("data/modelling_table.csv", index=False)

# Calculate delay statistics
delayed_count = int(model_table["is_delayed"].sum())
delay_rate = model_table["is_delayed"].mean()

# Calculate declared value statistics
value_mean = model_table["declared_value_inr"].mean()
value_median = model_table["declared_value_inr"].median()

log_values = np.log(model_table["declared_value_inr"])
log_mean = log_values.mean()
log_median = log_values.median()

# Calculate delivery-hour outliers using the IQR method
q1 = model_table["delivery_hours"].quantile(0.25)
q3 = model_table["delivery_hours"].quantile(0.75)

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = (
    (model_table["delivery_hours"] < lower_bound) |
    (model_table["delivery_hours"] > upper_bound)
).sum()

# Count missing values in features and target
null_count = model_table[FEATURES + [
    "delivery_hours", "is_delayed"
]].isna().sum().sum()

# Prepare checks table
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", len(raw_dedup)),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("sentinel_weight_rows", sentinel_rows),
    ("missing_declared_value_before", missing_value_before),
    ("missing_payment_mode_before", missing_payment_before),
    ("service_type_spellings_before", spellings_before),
    ("service_type_spellings_after", spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(model_table)),
    ("nulls_in_features_and_target", null_count),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    ("hours_outlier_count", outlier_count),
    ("declared_value_mean_inr", value_mean),
    ("declared_value_median_inr", value_median),
    ("log_declared_value_mean", log_mean),
    ("log_declared_value_median", log_median)
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = checks_df["value"].round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# Create distance bands
model_table["distance_band"] = pd.cut(
    model_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True
)

# Function to calculate group statistics
def group_statistics(data, column, dimension):

    result = data.groupby(
        column, observed=False
    ).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    result = result.rename(columns={column: "group"})
    result["dimension"] = dimension

    return result[
        ["dimension", "group", "shipments", "delayed",
         "delay_rate", "mean_hours"]
    ]


# Calculate delay statistics for three dimensions
service_stats = group_statistics(
    model_table, "service_type", "service_type"
)

tier_stats = group_statistics(
    model_table, "hub_tier", "hub_tier"
)

distance_stats = group_statistics(
    model_table, "distance_band", "distance_band"
)

# Combine all groups
delay_groups = pd.concat(
    [service_stats, tier_stats, distance_stats],
    ignore_index=True
)

delay_groups["delay_rate"] = delay_groups["delay_rate"].round(4)
delay_groups["mean_hours"] = delay_groups["mean_hours"].round(4)

delay_groups.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# Create correlation matrix of numeric columns
numeric_data = model_table.select_dtypes(include="number")
correlations = numeric_data.corr()

correlations.round(4).to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    index_label="column"
)

print("PART A COMPLETED SUCCESSFULLY")
print("Modelling table:", model_table.shape)
print("Checks:", len(checks_df), "metrics")
print("Delay groups:", len(delay_groups))
print("Correlation matrix:", correlations.shape)
print("Delayed shipments:", delayed_count)
