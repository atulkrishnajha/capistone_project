# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run
1. Ensure Python 3.9+ is installed. Clone the repository and navigate into it.
2. Install the exact package versions required: `pip install -r requirements.txt`
3. Generate the dataset: `python generate_data.py`
4. Run the data preparation and EDA step: `python part_a_clean_eda.py`
5. Run the regression models: `python part_b_regression.py`
6. Run the classification models and threshold selection: `python part_c_classification.py`
7. Generate the hub segments: `python part_d_segments.py`
8. Package the models: `python package_model.py` (Note: `models/delay_model.pkl` is regenerated here to guarantee compatibility with your environment's pinned scikit-learn version).
9. Run predictions on new data: `python predict_delay.py`

## Data findings
- **Service Types:** The `Express` service is delayed most often, with a delay rate of `0.4897`.
- **Hub Tiers:** Delay rates climb sharply as tier downgrades; Tier 1 hubs see a delay rate of `0.0400`, jumping to `0.2150` for Tier 2, and `0.4400` for Tier 3.
- **Distance:** Proximity is safety. The delay rate is only `0.0219` for 0-200 km, rising steadily to a massive `0.8523` for the 601+ km distance band.
- **Correlations:** The three features most strongly correlated with `delivery_hours` (excluding itself) are `promised_hours` (0.8367), `distance_km` (0.4011), and `n_stops` (0.2458). 
- **Transformations:** `declared_value_inr` was log-transformed because it was heavily right-skewed; its mean was `1533.05` versus a median of `1121.50`, whereas `log_declared_value` has a mean of `7.0280` and a median of `7.0220`, restoring a normal distribution for linear models.

## Model results (regression)
- **Baseline:** The naive mean baseline achieved a validation R² of `-0.0211`, meaning predicting the global mean performs worse than a simple flat horizontal line, capturing zero variance.
- **Winning model:** `GradientBoosting` and `LinearRegression` tied for the top validation spot (RMSE `6.1118`, R² `0.9310`), but Gradient Boosting narrowly won on Cross-Validation (Mean R² `0.9351` vs `0.9255`).
- **Tree Diagnostics:** High-bias models (underfitting) occurred at depth 1 (training RMSE `11.0261` vs validation RMSE `11.5303` - gap `0.5042`). High-variance models (overfitting) occurred with unlimited depth (training RMSE `0.0000` vs validation RMSE `7.7852` - gap `7.7852`).
- **Learning Rate:** A learning rate of 0.1 was optimal. 0.01 was too slow (Validation R² `0.7127`), failing to converge, while 1.0 was overshooting the minima (Validation R² `0.8436`).
- **Final Pick:** Gradient Boosting is the model to ship due to its slightly higher robustness during the 5-fold cross-validation phase.

## Model results (classification)
- **Baseline Limits:** A blind baseline that predicts everything is on-time yields a validation accuracy of `0.7667`. This is useless because the true cost lies in the false negatives (missing actual delays); an accuracy of 76% just reflects class imbalance, catching exactly 0 delays.
- **Noise Analysis:** The five bookkeeping/noise columns absorbed `0.2250` (22.5%) of the default Random Forest's feature importance, but a univariate F-test successfully stripped them, ranking all five at position 8 or lower.
- **K Selection:** The `k` table shows validation F1 peaks when using the top 7 features (F1 `0.8333`), filtering out the algorithmic noise without stripping actual signal.
- **Tuning Limits:** Grid tuning barely improved over a default forest using the same 7 features (`0.8333` vs `0.8333`). Over-tuning on a 180-row validation set risks fitting to the validation noise.
- **Threshold:** To hit the operational rule of ≥80% recall, a probability threshold of `0.50` was selected (yielding Validation Recall of `0.8333`).
- **Final Test:** On the test set, this threshold achieved a Precision of `0.6889`, a Recall of `0.7561`, and an F1 of `0.7209`. Because the test set only contains 41 delayed parcels, variance easily explains why the 75.6% test recall fell slightly short of the 80% validation goal.

## Operations recommendation
- **Escalate Hubs:** The hubs grouped into the `Escalate` segment are HUB09, HUB13, HUB15, HUB17, HUB18, HUB19, HUB20, HUB21, and HUB24. As a block, they experience a `0.4600` (46%) delay rate.
- **Mismatches:** Four hubs are behaving radically differently from their corporate tier assignments: HUB07 Kolkata (Tier 1 operating at `Watch` levels), HUB13 Nagpur (Tier 2 at `Escalate` levels), HUB15 Kochi (Tier 2 at `Escalate` levels), and HUB22 Dehradun (Tier 3 operating cleanly at `Watch` levels). Their operational playbooks should match their Segment, not their Tier.
- **The K=3 Constraint:** We kept K=3 for clustering because business limits operations to three playbooks. While the separation ratio improves for K=4 and K=5 (up to `6.8867`), those cuts produce clusters of fewer than 6 hubs, which are too small to justify maintaining isolated, standalone operational playbooks.
- **Action Outcomes:** Of the 48 new parcels in the daily batch, 14 were flagged for delays. Under our segmented playbook, exactly 7 of those require immediate escalation to a hub manager.

## Production plan
1. **Consistency:** Using `src/shipment_prep.py` functions guarantees that training data and live streaming predictions undergo the exact same preprocessing logic.
2. **Reproducibility:** A pinned `requirements.txt` and hard-coded `random_state` seeds ensure the models output the exact same splits and probabilities regardless of what server runs them.
3. **Accuracy / Reliability:** The pipeline dynamically stores imputation values (`impute_values.csv`) and thresholds (`delay_threshold.csv`) during training to gracefully handle missing values at inference without breaking.
- **Monitoring:** (1) Monitor the Weekly Recall rate on delivered parcels (Alert if < 0.70); (2) Monitor the share of parcels flagged daily (Alert if > 0.30); (3) Monitor feature nulls on new streaming logs (Alert if `distance_km` missing > 5%).
- **Retraining:** Retrain monthly. For sale windows (`sale_event_flag`), models should be compared against the performance during previous equivalent sale periods rather than ordinary quiet weeks, as standard drift alerts will aggressively false-alarm during promotional events due to shifted volume/weight metrics.

## Limitations
- **Synthetic Foundation:** The entire pipeline operates on a synthetic benchmark which lacks the true geospatial, traffic, and temporal complexities of a real logistics network.
- **Test Set Size:** The test set relies on merely 41 positively delayed parcels, creating high variance in our final confidence estimates (a shift of just 3 correct predictions swings recall by ~7%).
- **Single-split Thresholding:** The critical operational threshold (0.50) was tuned on a singular 180-row validation split. A cross-validated threshold selection over multiple temporal splits would provide far better real-world robustness.
