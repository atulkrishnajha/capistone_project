import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
rf = RandomForestClassifier(
    n_estimators=int(cfg["n_estimators"]),
    max_depth=None if cfg["max_depth"] == "unlimited" else int(cfg["max_depth"]),
    min_samples_leaf=int(cfg["min_samples_leaf"]),
    random_state=42
)
pipe = Pipeline([("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))), ("model", rf)]).fit(X_train, y_train)

joblib.dump(pipe, "models/delay_model.pkl")
pd.DataFrame([{"threshold": cfg["threshold"]}]).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame(list(medians.items()), columns=["column", "median"]).to_csv("models/impute_values.csv", index=False)

loaded_pipe = joblib.load("models/delay_model.pkl")
mem_probs = pipe.predict_proba(X_test)[:, 1]
load_probs = loaded_pipe.predict_proba(X_test)[:, 1]
diff = np.max(np.abs(mem_probs - load_probs))

pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {"metric": "reload_max_abs_probability_diff", "value": diff},
    {"metric": "threshold", "value": cfg["threshold"]}
]).to_csv("outputs/part_d_reload_check.csv", index=False)
