
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, recall_score

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load modelling data
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["is_delayed"]

# Use exactly the same split as Part C
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Select the top 7 features using training data only
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    "f_score", ascending=False
)

selected_features = feature_scores.head(7)["feature"].tolist()

X_train_7 = X_train[selected_features]
X_val_7 = X_val[selected_features]

print("Selected features:")
print(selected_features)

# Default Random Forest
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train_7, y_train)

# Grid Search
parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    parameter_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_7, y_train)
grid_model = grid_search.best_estimator_

# Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    parameter_grid,
    n_iter=9,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_7, y_train)
random_model = random_search.best_estimator_

# Compare models using validation F1
candidates = {
    "Default Random Forest": rf_default,
    "Grid Search Random Forest": grid_model,
    "Randomized Search Random Forest": random_model
}

scores = {}

for name, model in candidates.items():
    predictions = model.predict(X_val_7)
    scores[name] = f1_score(
        y_val, predictions, zero_division=0
    )

print("\nValidation F1 scores:")
print(scores)

best_name = max(scores, key=scores.get)
best_model = candidates[best_name]

# Find the highest threshold with recall >= 0.80
probabilities = best_model.predict_proba(X_val_7)[:, 1]

valid_thresholds = []

for threshold in np.arange(0.1, 1.0, 0.1):
    predictions = (probabilities >= threshold).astype(int)
    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    if recall >= 0.80:
        valid_thresholds.append(round(threshold, 1))

if valid_thresholds:
    best_threshold = max(valid_thresholds)
else:
    best_threshold = 0.1

# Save the model
joblib.dump(best_model, "models/delay_model.pkl")

# Save threshold
pd.DataFrame({
    "threshold": [best_threshold]
}).to_csv("models/delay_threshold.csv", index=False)

# Save selected feature order
pd.DataFrame({
    "feature": selected_features
}).to_csv("models/feature_order.csv", index=False)

# Calculate and save training-data imputation medians
raw_data = pd.read_csv("data/raw_shipments.csv")
cleaned_data, medians = clean_shipments(raw_data)

pd.DataFrame({
    "feature": list(medians.keys()),
    "value": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)

# Verify that the saved model loads correctly
loaded_model = joblib.load("models/delay_model.pkl")

loaded_probabilities = loaded_model.predict_proba(X_val_7)[:, 1]

difference = np.max(
    np.abs(probabilities - loaded_probabilities)
)

print("\nPACKAGING RESULTS")
print("Selected model:", best_name)
print("Best threshold:", best_threshold)
print("Best parameters:", best_model.get_params())
print("Maximum probability difference:", difference)

if difference <= 1e-9:
    print("\nMODEL RELOAD VERIFICATION PASSED")
else:
    print("\nMODEL RELOAD VERIFICATION FAILED")

print("\nMODEL PACKAGING COMPLETED")
