import os
import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load new shipment data
new_data = pd.read_csv("data/new_shipments.csv")

# Load hubs data
hubs = pd.read_csv("data/hubs.csv")

# Load saved median values
median_data = pd.read_csv("models/impute_values.csv")
medians = dict(zip(median_data["feature"], median_data["value"]))

# Clean new shipment data
cleaned_data, _ = clean_shipments(new_data, medians=medians)

# Create model features
featured_data = build_features(cleaned_data, hubs)

# Load selected feature order
feature_data = pd.read_csv("models/feature_order.csv")
feature_order = feature_data["feature"].tolist()

# Prepare input for prediction
X_new = featured_data[feature_order]

# Load saved model
model = joblib.load("models/delay_model.pkl")

# Load selected threshold
threshold_data = pd.read_csv("models/delay_threshold.csv")
threshold = threshold_data["threshold"].iloc[0]

# Predict delay probabilities
probabilities = model.predict_proba(X_new)[:, 1]

# Predict delayed or not delayed
predictions = (probabilities >= threshold).astype(int)

# Prepare prediction results
results = featured_data[["shipment_id", "hub_id"]].copy()

results["delay_probability"] = probabilities.round(4)
results["predicted_is_delayed"] = predictions

# Add hub segment information
segments = pd.read_csv("outputs/hub_segments.csv")

results = results.merge(
    segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
)

# Recommend operational actions
def recommend_action(row):

    if row["predicted_is_delayed"] == 1:

        if row["segment"] == "Escalate":
            return "Escalate to hub manager"

        else:
            return "Proactive customer call"

    else:
        return "No action"


results["recommended_action"] = results.apply(
    recommend_action,
    axis=1
)

# Save predictions
results.to_csv(
    "outputs/new_shipments_predictions.csv",
    index=False
)

# Display results
print("\nNEW SHIPMENT PREDICTIONS")
print("Total shipments:", len(results))

print("\nPrediction counts:")
print(results["predicted_is_delayed"].value_counts())

print("\nRecommended actions:")
print(results["recommended_action"].value_counts())

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))

print("\nPrediction file saved successfully!")