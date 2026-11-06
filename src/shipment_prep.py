
import pandas as pd
import numpy as np

# Delivery promise for each service
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

# Exact 17 machine learning features in required order
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

# These five columns are bookkeeping/noise columns
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):

    # Make a copy of the data
    df = raw.copy()

    # Step 1: Remove duplicate rows
    df = df.drop_duplicates()

    # Step 2: Clean service names
    df["service_type"] = df["service_type"].str.strip().str.title()

    # Step 3: Fill missing payment modes
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: Clean declared value
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value_inr"], errors="coerce"
    )

    # Remove old text column
    df = df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: Treat weights above 100 as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step 7: Calculate medians if not provided
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median())
        }

    # Fill missing values using medians
    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])

    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(
        medians["parcel_weight_kg"]
    )

    df["declared_value_inr"] = df["declared_value_inr"].fillna(
        medians["declared_value_inr"]
    )

    return df, medians


def build_features(clean, hubs):

    # Make a copy
    df = clean.copy()

    # Add hub information using left merge
    df = df.merge(hubs, on="hub_id", how="left")

    # Add promised delivery hours
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # COD = 1, other payment modes = 0
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # Weekend = 1, weekday = 0
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Extract dispatch hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # Natural log of declared value
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
	