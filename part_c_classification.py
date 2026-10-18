import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Stratified Split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

def get_metrics(y_t, y_p, name):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel() if sum(y_p)>0 else (len(y_t)-sum(y_t), 0, sum(y_t), 0)
    acc, prec, rec, f1 = accuracy_score(y_t, y_p), precision_score(y_t, y_p, zero_division=0), recall_score(y_t, y_p), f1_score(y_t, y_p)
    return {"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": round(acc,4), "precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4)}

# Models
res = [get_metrics(y_val, np.zeros(len(y_val)), "blind_all_on_time")]

lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
lr.fit(X_train, y_train)
res.append(get_metrics(y_val, lr.predict(X_val), "logistic_regression"))

best_dt, best_f1 = None, 0
for d in [2, 3, 4, 5, 6, 8, 10]:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f1 = f1_score(y_val, dt.predict(X_val))
    if f1 > best_f1: best_f1, best_dt = f1, d
dt = DecisionTreeClassifier(criterion="gini", max_depth=best_dt, random_state=42).fit(X_train, y_train)
res.append(get_metrics(y_val, dt.predict(X_val), f"decision_tree_depth_{best_dt}"))

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
res.append(get_metrics(y_val, rf.predict(X_val), "random_forest_default"))
pd.DataFrame(res).to_csv("outputs/part_c_models.csv", index=False)

# Feature importance
skb = SelectKBest(f_classif, k="all").fit(X_train, y_train)
f_ranks = pd.Series(skb.scores_).rank(ascending=False).astype(int).values
f_df = pd.DataFrame({
    "feature": FEATURES, "f_score": np.round(skb.scores_, 4), "f_rank": f_ranks,
    "rf_importance": np.round(rf.feature_importances_, 4),
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
}).sort_values("f_rank")
f_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

k_res = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    k_res.append({"k": k, "val_f1": round(f1_score(y_val, pipe.fit(X_train, y_train).predict(X_val)), 4)})
pd.DataFrame(k_res).to_csv("outputs/part_c_selectk.csv", index=False)

# Tuning
sel = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xt_train, Xt_val = sel.transform(X_train), sel.transform(X_val)

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
gs.fit(Xt_train, y_train)

rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
rs.fit(Xt_train, y_train)

rf_def = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xt_train, y_train)

tune_res = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135, "best_params": str(gs.best_params_), "best_cv_accuracy": round(gs.best_score_, 4), "val_f1": round(f1_score(y_val, gs.predict(Xt_val)), 4)},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45, "best_params": str(rs.best_params_), "best_cv_accuracy": round(rs.best_score_, 4), "val_f1": round(f1_score(y_val, rs.predict(Xt_val)), 4)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1, "best_params": "defaults", "best_cv_accuracy": "", "val_f1": round(f1_score(y_val, rf_def.predict(Xt_val)), 4)}
]
tune_df = pd.DataFrame(tune_res)
tune_df.to_csv("outputs/part_c_tuning.csv", index=False)

# Threshold Selection
best_cfg = tune_df.loc[tune_df["val_f1"].idxmax()]
final_rf = gs.best_estimator_ if best_cfg["search"] == "grid_search" else (rs.best_estimator_ if best_cfg["search"] == "randomized_search" else rf_def)
final_pipe = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", final_rf)]).fit(X_train, y_train)

probs = final_pipe.predict_proba(X_val)[:, 1]
thresh_res = []
chosen_t = 0.1
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    preds = (probs >= t).astype(int)
    m = get_metrics(y_val, preds, str(t))
    m["threshold"] = t
    del m["model"]
    thresh_res.append(m)
    if m["recall"] >= 0.80: chosen_t = t

pd.DataFrame(thresh_res).to_csv("outputs/part_c_thresholds.csv", index=False)

# Test Evaluation
test_probs = final_pipe.predict_proba(X_test)[:, 1]
test_m = get_metrics(y_test, (test_probs >= chosen_t).astype(int), str(chosen_t))
test_m["threshold"] = chosen_t
del test_m["model"]
pd.DataFrame([test_m]).to_csv("outputs/part_c_final_test.csv", index=False)

cfg = {
    "selector_k": 7, "n_estimators": final_rf.n_estimators,
    "max_depth": final_rf.max_depth if final_rf.max_depth else "unlimited",
    "min_samples_leaf": final_rf.min_samples_leaf, "threshold": chosen_t,
    "chosen_from": best_cfg["search"]
}
pd.DataFrame([cfg]).to_csv("outputs/part_c_final_config.csv", index=False)
