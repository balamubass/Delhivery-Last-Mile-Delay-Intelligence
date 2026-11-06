import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

# Load data
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["is_delayed"]

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

print("\nDelayed shipments:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Testing:", int(y_test.sum()))


# Function for classification metrics
def get_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0)
    }


# 1. Blind baseline: predict every shipment as on time
baseline_predictions = np.zeros(len(y_val), dtype=int)

baseline_metrics = get_metrics(y_val, baseline_predictions)

print("\nBLIND BASELINE")
print(baseline_metrics)


# 2. Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_metrics = get_metrics(y_val, logistic_predictions)


# 3. Decision Tree depth tuning
depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []
tree_models = {}

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        criterion="gini",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    metrics = get_metrics(y_val, predictions)

    tree_models[depth] = model

    tree_results.append({
        "max_depth": depth,
        **metrics
    })

tree_df = pd.DataFrame(tree_results)

tree_df = tree_df.sort_values(
    ["F1", "max_depth"],
    ascending=[False, True]
)

best_depth = int(tree_df.iloc[0]["max_depth"])
best_tree = tree_models[best_depth]

tree_df.to_csv(
    "outputs/part_c_tree_depth.csv", index=False
)


# 4. Random Forest for feature importance
default_forest = RandomForestClassifier(random_state=42)

default_forest.fit(X_train, y_train)

# Random Forest feature importance
rf_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": default_forest.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "importance", ascending=False
)

rf_importance.to_csv(
    "outputs/part_c_rf_importance.csv", index=False
)

noise_share = rf_importance[
    rf_importance["feature"].isin(NOISE_COLUMNS)
]["importance"].sum()

print("\nNOISE FEATURE IMPORTANCE SHARE:", round(noise_share, 4))


# 5. SelectKBest feature scores
selector_all = SelectKBest(score_func=f_classif, k="all")

selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_
})

feature_scores = feature_scores.sort_values(
    "f_score", ascending=False
)

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv", index=False
)

print("\nFEATURE SCORES")
print(feature_scores.to_string(index=False))

# Select the top 7 features
selected_features = feature_scores.head(7)["feature"].tolist()

print("\nSELECTED TOP 7 FEATURES")
print(selected_features)


# 6. Compare different numbers of selected features
k_values = [5, 7, 10, 12, 17]

k_results = []

for k in k_values:
    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_val_selected)

    k_results.append({
        "k": k,
        "F1": f1_score(y_val, predictions, zero_division=0)
    })

k_df = pd.DataFrame(k_results)

k_df.to_csv(
    "outputs/part_c_feature_selection.csv", index=False
)

print("\nFEATURE SELECTION RESULTS")
print(k_df.to_string(index=False))


# Use the selected top 7 features
X_train_7 = X_train[selected_features]
X_val_7 = X_val[selected_features]
X_test_7 = X_test[selected_features]


# 7. Default Random Forest using top 7 features
rf_default = RandomForestClassifier(random_state=42)

rf_default.fit(X_train_7, y_train)

rf_default_predictions = rf_default.predict(X_val_7)

rf_default_f1 = f1_score(
    y_val, rf_default_predictions, zero_division=0
)


# 8. Grid Search
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

grid_predictions = grid_model.predict(X_val_7)

grid_f1 = f1_score(
    y_val, grid_predictions, zero_division=0
)


# 9. Randomized Search
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

random_predictions = random_model.predict(X_val_7)

random_f1 = f1_score(
    y_val, random_predictions, zero_division=0
)


# 10. Compare Random Forest models
# Order is important for tie-breaking:
# Default, Grid Search, Randomized Search

rf_candidates = {
    "Default Random Forest": rf_default,
    "Grid Search Random Forest": grid_model,
    "Randomized Search Random Forest": random_model
}

rf_comparison = pd.DataFrame([
    {"Model": "Default Random Forest", "F1": rf_default_f1},
    {"Model": "Grid Search Random Forest", "F1": grid_f1},
    {"Model": "Randomized Search Random Forest", "F1": random_f1}
])

best_candidate_name = rf_comparison.loc[
    rf_comparison["F1"].idxmax(), "Model"
]

best_rf = rf_candidates[best_candidate_name]

rf_comparison.to_csv(
    "outputs/part_c_rf_comparison.csv", index=False
)


# 11. Compare all validation models
validation_results = [
    {"Model": "Logistic Regression", **logistic_metrics},
    {"Model": "Decision Tree", **get_metrics(
        y_val, best_tree.predict(X_val)
    )}
]

for name, model in rf_candidates.items():
    predictions = model.predict(X_val_7)

    validation_results.append({
        "Model": name,
        **get_metrics(y_val, predictions)
    })

validation_df = pd.DataFrame(validation_results)

validation_df.to_csv(
    "outputs/part_c_validation.csv", index=False
)


# 12. Choose probability threshold using validation data
probabilities = best_rf.predict_proba(X_val_7)[:, 1]

threshold_results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    predictions = (probabilities >= threshold).astype(int)

    metrics = get_metrics(y_val, predictions)

    threshold_results.append({
        "threshold": round(threshold, 1),
        **metrics
    })

threshold_df = pd.DataFrame(threshold_results)

valid_thresholds = threshold_df[
    threshold_df["Recall"] >= 0.80
]

if len(valid_thresholds) > 0:
    best_threshold = float(
        valid_thresholds["threshold"].max()
    )
else:
    best_threshold = 0.1

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv", index=False
)


# 13. Final test evaluation
# Use the selected threshold on the test set
test_probabilities = best_rf.predict_proba(X_test_7)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = get_metrics(y_test, test_predictions)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

test_results = pd.DataFrame([{
    "Best_Model": best_candidate_name,
    "Threshold": best_threshold,
    **test_metrics,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

test_results.to_csv(
    "outputs/part_c_test_results.csv", index=False
)


# 14. Save and display results
print("\nDECISION TREE RESULTS")
print(tree_df.to_string(index=False))

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(rf_importance.to_string(index=False))

print("\nFEATURE SELECTION RESULTS")
print(k_df.to_string(index=False))

print("\nRANDOM FOREST COMPARISON")
print(rf_comparison.to_string(index=False))

print("\nVALIDATION MODEL RESULTS")
print(validation_df.to_string(index=False))

print("\nTHRESHOLD RESULTS")
print(threshold_df.to_string(index=False))

print("\nSELECTED THRESHOLD:", best_threshold)

print("\nFINAL TEST RESULTS")
print(test_results.to_string(index=False))

print("\nPART C COMPLETED")