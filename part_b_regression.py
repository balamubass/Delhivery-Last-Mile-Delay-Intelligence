import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

# Load data
data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["delivery_hours"]

# Split data into training, validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


# Function to calculate regression metrics
def calculate_metrics(model, X_data, y_data):
    predictions = model.predict(X_data)

    mae = mean_absolute_error(y_data, predictions)
    mse = mean_squared_error(y_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_data, predictions)

    return mae, mse, rmse, r2


# 1. Baseline model
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_metrics = calculate_metrics(baseline, X_val, y_val)


# 2. Decision Tree depth tuning
depths = [1, 2, 3, 4, 6, 8, 12, None]

depth_results = []

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    mae, mse, rmse, r2 = calculate_metrics(model, X_val, y_val)

    depth_results.append({
        "max_depth": str(depth),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

depth_df = pd.DataFrame(depth_results)

depth_df["depth_order"] = depth_df["max_depth"].replace(
    "None", "999"
).astype(int)

depth_df = depth_df.sort_values(
    ["RMSE", "depth_order"]
)

best_depth = depth_df.iloc[0]["max_depth"]

if best_depth == "None":
    best_depth = None
else:
    best_depth = int(best_depth)

best_tree = DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)

best_tree.fit(X_train, y_train)

depth_df.drop(columns=["depth_order"]).to_csv(
    "outputs/part_b_depth_tuning.csv", index=False
)


# 3. Linear Regression with scaling
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)


# 4. Random Forest
random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)


# 5. Gradient Boosting learning rate tuning
learning_rates = [0.01, 0.1, 1.0]

gradient_results = []
gradient_models = {}

for rate in learning_rates:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    model.fit(X_train, y_train)

    mae, mse, rmse, r2 = calculate_metrics(model, X_val, y_val)

    gradient_models[rate] = model

    gradient_results.append({
        "learning_rate": rate,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

gradient_df = pd.DataFrame(gradient_results)

gradient_df = gradient_df.sort_values(
    ["RMSE", "learning_rate"]
)

best_rate = float(gradient_df.iloc[0]["learning_rate"])
best_gradient = gradient_models[best_rate]

gradient_df.to_csv(
    "outputs/part_b_learning_rates.csv", index=False
)


# Compare all models on validation data
models = {
    "Baseline": baseline,
    "Decision Tree": best_tree,
    "Linear Regression": linear_model,
    "Random Forest": random_forest,
    "Gradient Boosting": best_gradient
}

validation_results = []

for name, model in models.items():
    mae, mse, rmse, r2 = calculate_metrics(model, X_val, y_val)

    validation_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values("RMSE")

validation_df.to_csv(
    "outputs/part_b_regression_validation.csv", index=False
)


# Cross-validation on the top two models
top_two = validation_df[
    validation_df["Model"] != "Baseline"
].head(2)["Model"].tolist()

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name in top_two:
    model = models[name]

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2"
    )

    cv_results.append({
        "Model": name,
        "CV_R2_Mean": scores.mean(),
        "CV_R2_Std": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cross_validation.csv", index=False
)


# Select the best model using validation RMSE
best_model_name = validation_df.iloc[0]["Model"]
best_model = models[best_model_name]

# Evaluate the winning model on the test set only once
best_model.fit(X_train, y_train)

test_mae, test_mse, test_rmse, test_r2 = calculate_metrics(
    best_model, X_test, y_test
)

test_df = pd.DataFrame([{
    "Best_Model": best_model_name,
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

test_df.to_csv(
    "outputs/part_b_test_results.csv", index=False
)


# Display results
print("\nDECISION TREE DEPTH RESULTS")
print(depth_df.drop(columns=["depth_order"]).to_string(index=False))

print("\nGRADIENT BOOSTING LEARNING RATE RESULTS")
print(gradient_df.to_string(index=False))

print("\nVALIDATION RESULTS")
print(validation_df.to_string(index=False))

print("\nCROSS VALIDATION RESULTS")
print(cv_df.to_string(index=False))

print("\nFINAL TEST RESULT")
print(test_df.to_string(index=False))

print("\nPART B COMPLETED")