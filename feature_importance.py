
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# LOAD DATASET

df = pd.read_csv("data/new_car_price_dataset.csv")

df = df.drop(
    columns=["Unnamed: 0", "Date"],
    errors="ignore"
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])
# Standardize price to Million LKR
df["Price"] = df["Price"] / 10.0

X = df.drop(columns=["Price"])
y = df["Price"]


# IDENTIFY FEATURES


numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]


# PREPROCESSING

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])



# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TRAIN RANDOM FOREST

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest for feature importance...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"\nRandom Forest Test MAE: {mae:.4f}")


# PERMUTATION FEATURE IMPORTANCE


print("\nCalculating permutation importance...")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=2
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(importance_df.to_string(index=False))



# SAVE RESULTS


importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance saved to feature_importance.csv")


# VISUALIZE TOP 10 FEATURES

top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Decrease in Model Performance (MAE)")
plt.ylabel("Vehicle Features")
plt.title("Random Forest - Permutation Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\nFeature importance analysis completed!")
