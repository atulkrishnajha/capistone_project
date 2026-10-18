import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

# Split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

metrics = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": round(np.max(np.abs(X_train_scaled.mean(axis=0))), 6),
    "test_scaled_max_abs_mean": round(np.max(np.abs(X_test_scaled.mean(axis=0))), 6)
}
pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

# Baseline
y_pred_base = np.full(len(y_val), y_train.mean())
base_rmse = np.sqrt(mean_squared_error(y_val, y_pred_base))

# Tree diagnostics
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_res = []
for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    t_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
    v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
    tree_res.append({"max_depth": d if d else "unlimited", "train_rmse": round(t_rmse, 4), "val_rmse": round(v_rmse, 4)})
tree_df = pd.DataFrame(tree_res)
tree_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
best_depth = tree_df.loc[tree_df["val_rmse"].idxmin(), "max_depth"]
best_depth = None if best_depth == "unlimited" else int(best_depth)

# Models
models = {
    "mean_baseline": None,
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

mod_res = []
for name, mod in models.items():
    if mod is None:
        y_p = y_pred_base
    else:
        mod.fit(X_train, y_train)
        y_p = mod.predict(X_val)
    mod_res.append({
        "model": name, "split": "val",
        "MAE": round(mean_absolute_error(y_val, y_p), 4),
        "MSE": round(mean_squared_error(y_val, y_p), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_val, y_p)), 4),
        "R2": round(r2_score(y_val, y_p), 4)
    })

# LR Experiment
lr_res = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    p = gb.predict(X_val)
    lr_res.append({"learning_rate": lr, "val_rmse": round(np.sqrt(mean_squared_error(y_val, p)), 4), "val_r2": round(r2_score(y_val, p), 4)})
pd.DataFrame(lr_res).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Top 2 CV
df_mods = pd.DataFrame(mod_res)
top2 = df_mods[df_mods["model"] != "mean_baseline"].sort_values("RMSE").head(2)["model"].tolist()

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_res = []
for name in top2:
    mod = models[name]
    scores = cross_val_score(mod, X_tv, y_tv, cv=kf, scoring="r2")
    cv_res.append({"model": name, "mean_r2": round(scores.mean(), 4), "std_r2": round(scores.std(), 4)})
pd.DataFrame(cv_res).to_csv("outputs/part_b_cv.csv", index=False)

# Final test
final_name = top2[0]
final_mod = models[final_name]
final_mod.fit(X_train, y_train) # refitted on train as requested
y_pt = final_mod.predict(X_test)
mod_res.append({
    "model": final_name, "split": "test",
    "MAE": round(mean_absolute_error(y_test, y_pt), 4),
    "MSE": round(mean_squared_error(y_test, y_pt), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pt)), 4),
    "R2": round(r2_score(y_test, y_pt), 4)
})
pd.DataFrame(mod_res).to_csv("outputs/part_b_metrics.csv", index=False)
