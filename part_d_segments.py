import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

os.makedirs("outputs", exist_ok=True)

# Load data
data = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# Create hub profile
hub_profile = data.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

# Save hub profile
hub_profile.round(4).to_csv(
    "outputs/hub_profile.csv", index=False
)

# Add city and tier information
hub_profile = hub_profile.merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

# Features for clustering
cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[cluster_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate separation for K = 2, 3, 4, 5
separation_results = []

distances = pairwise_distances(X_scaled)

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    within = []
    across = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == labels[j]:
                within.append(distances[i, j])
            else:
                across.append(distances[i, j])

    ratio = np.mean(across) / np.mean(within)

    sizes = pd.Series(labels).value_counts().sort_values(
        ascending=False
    )

    size_text = "-".join(str(x) for x in sizes.tolist())

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": size_text
    })

separation_df = pd.DataFrame(separation_results)

separation_df["across_over_within"] = (
    separation_df["across_over_within"].round(4)
)

separation_df.to_csv(
    "outputs/part_d_separation.csv", index=False
)

# Final K-Means model with 3 clusters
final_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = final_model.fit_predict(X_scaled)

# Order clusters by their average delay rate
cluster_delay = hub_profile.groupby("cluster")[
    "delay_rate"
].mean().sort_values()

cluster_names = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(cluster_names)

# Save hub segments
hub_segments = hub_profile[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].copy()

hub_segments.to_csv(
    "outputs/hub_segments.csv", index=False
)

# Create segment profile
segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values("segment")
segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile.round(4).to_csv(
    "outputs/segment_profile.csv", index=False
)

# Find tier mismatches
expected_segments = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = hub_segments[
    "hub_tier"
].map(expected_segments)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv", index=False
)

# Display results
print("\nHUB PROFILE")
print(hub_profile.to_string(index=False))

print("\nSEPARATION RESULTS")
print(separation_df.to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.to_string(index=False))

print("\nTIER MISMATCHES")
print(mismatches.to_string(index=False))

print("\nPART D SEGMENTATION COMPLETED")