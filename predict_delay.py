import pandas as pd
import numpy as np
import joblib
from src.shipment_prep import clean_shipments, build_features

raw_new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segs = pd.read_csv("outputs/hub_segments.csv")
med_df = pd.read_csv("models/impute_values.csv")
medians = dict(zip(med_df["column"], med_df["median"]))
feats = pd.read_csv("models/feature_order.csv")["feature"].tolist()
thresh = pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
model = joblib.load("models/delay_model.pkl")

clean_df, _ = clean_shipments(raw_new, medians=medians)
model_df = build_features(clean_df, hubs)
X_new = model_df[feats]

clean_df["delay_probability"] = np.round(model.predict_proba(X_new)[:, 1], 4)
clean_df["flagged"] = (clean_df["delay_probability"] >= thresh).astype(int)
clean_df = clean_df.merge(segs[["hub_id", "segment"]], on="hub_id", how="left")

def get_action(row):
    if not row["flagged"]: return "No action"
    return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"

clean_df["action"] = clean_df.apply(get_action, axis=1)
clean_df[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv("outputs/new_shipment_actions.csv", index=False)
