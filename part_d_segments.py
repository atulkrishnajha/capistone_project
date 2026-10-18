import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/modelling_table.csv")
profile = df.groupby("hub_id", as_index=False).agg(
    shipments=("is_delayed", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).sort_values("hub_id")
profile.to_csv("outputs/hub_profile.csv", index=False)

cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[cols])

sep_res = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    labels = km.labels_
    sizes = "-".join(map(str, sorted(pd.Series(labels).value_counts().values, reverse=True)))
    
    within, across = [], []
    for i in range(24):
        for j in range(i+1, 24):
            d = np.linalg.norm(scaled[i] - scaled[j])
            if labels[i] == labels[j]: within.append(d)
            else: across.append(d)
    ratio = np.mean(across) / np.mean(within) if within else 0
    sep_res.append({"K": k, "across_over_within": round(ratio, 4), "cluster_sizes": sizes})
pd.DataFrame(sep_res).to_csv("outputs/part_d_separation.csv", index=False)

# Final Segments K=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
profile["cluster"] = km3.labels_
seg_map = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
name_map = {seg_map[0]: "Standard", seg_map[1]: "Watch", seg_map[2]: "Escalate"}
profile["segment"] = profile["cluster"].map(name_map)

hubs = pd.read_csv("data/hubs.csv")
merged = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
merged[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)

seg_prof = merged.groupby("segment", as_index=False).agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).round(4)
seg_prof.to_csv("outputs/segment_profile.csv", index=False)

# Mismatches
tier_seg = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = merged[merged["hub_tier"].map(tier_seg) != merged["segment"]]
mismatch[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
