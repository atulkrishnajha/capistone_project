import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    clean_df = raw.drop_duplicates().copy()
    
    clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
    
    # Parse declared_value
    if "declared_value" in clean_df.columns:
        clean_df["declared_value_inr"] = (
            clean_df["declared_value"].astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        clean_df = clean_df.drop(columns=["declared_value"])
        
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
    
    # Sentinel removal
    clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    if medians is None:
        medians = {
            "distance_km": clean_df["distance_km"].median(),
            "parcel_weight_kg": clean_df["parcel_weight_kg"].median(),
            "declared_value_inr": clean_df["declared_value_inr"].median()
        }
        
    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return clean_df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
