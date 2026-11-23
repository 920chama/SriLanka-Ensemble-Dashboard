import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Custom Hybrid Regressor class for seamless joblib serialization
class RF_ANN_HybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rf_pipeline=None, ann_pipeline=None, rf_weight=0.6):
        self.rf_pipeline = rf_pipeline
        self.ann_pipeline = ann_pipeline
        self.rf_weight = rf_weight

    def fit(self, X, y):
        if self.rf_pipeline is not None:
            self.rf_pipeline.fit(X, y)
        if self.ann_pipeline is not None:
            self.ann_pipeline.fit(X, y)
        return self

    def predict(self, X):
        rf_pred = self.rf_pipeline.predict(X)
        ann_pred = self.ann_pipeline.predict(X)
        return self.rf_weight * rf_pred + (1.0 - self.rf_weight) * ann_pred


def train_and_evaluate_all():
    os.makedirs("models", exist_ok=True)
    print("=" * 60)
    print("1. LOADING & PREPROCESSING SRI LANKAN VEHICLE DATASET")
    print("=" * 60)

    df = pd.read_csv("data/new_car_price_dataset.csv")

    # Drop unnecessary index & date
    df = df.drop(columns=["Unnamed: 0", "Date"], errors="ignore")

    TARGET_COLUMN = "Price"
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    
    # Standardize target price to Million LKR (1 Million = 10 Lakhs = 1,000,000 LKR)
    df[TARGET_COLUMN] = df[TARGET_COLUMN] / 10.0

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    print(f"Total instances: {len(df)}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Build Preprocessors
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(dtype=np.float64, handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    ann_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(dtype=np.float64, handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training split: {X_train.shape[0]} samples")
    print(f"Testing split:  {X_test.shape[0]} samples")

    print("\n" + "=" * 60)
    print("2. HYPERPARAMETER TUNING DEMONSTRATION (RandomForest)")
    print("=" * 60)
    param_distributions = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [None, 20, 30],
        "model__min_samples_split": [2, 5]
    }
    rf_base_pipe = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    search = RandomizedSearchCV(
        rf_base_pipe,
        param_distributions=param_distributions,
        n_iter=3,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1
    )
    search.fit(X_train, y_train)
    print(f"Best RF Params from CV: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")

    best_rf_estimator = search.best_estimator_

    print("\n" + "=" * 60)
    print("3. BUILDING ENSEMBLE SUITE & BASELINE MODELS")
    print("=" * 60)

    # 1. Bagging: Random Forest
    rf_model = best_rf_estimator

    # 2. Boosting: XGBoost
    xgb_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 3. Boosting: LightGBM
    lgb_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ))
    ])

    # 4. Boosting: CatBoost
    cat_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", CatBoostRegressor(
            iterations=200,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            verbose=0
        ))
    ])

    # 5. Boosting: Gradient Boosting (sklearn)
    gb_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ))
    ])

    # 6. Deep Learning Paradigm: ANN (MLPRegressor)
    ann_model = Pipeline([
        ("preprocessor", ann_preprocessor),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=250,
            early_stopping=True,
            random_state=42
        ))
    ])

    # 7. Stacking Regressor
    stacking_base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=1)),
        ("lgb", LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1, n_jobs=1)),
        ("cat", CatBoostRegressor(iterations=100, learning_rate=0.08, depth=5, random_seed=42, verbose=0))
    ]
    stacking_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", StackingRegressor(
            estimators=stacking_base_estimators,
            final_estimator=Ridge(alpha=10.0),
            cv=3,
            n_jobs=1
        ))
    ])

    models = {
        "Random Forest (Bagging)": rf_model,
        "XGBoost (Boosting)": xgb_model,
        "LightGBM (Boosting)": lgb_model,
        "CatBoost (Boosting)": cat_model,
        "Gradient Boosting (Boosting)": gb_model,
        "Stacking Ensemble": stacking_model,
        "Artificial Neural Network (ANN)": ann_model
    }

    results = []
    prediction_data = pd.DataFrame({"Actual": y_test.to_numpy()})
    trained_predictions = {}

    print("\n" + "=" * 60)
    print("4. TRAINING & EVALUATING ALL MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--> Training: {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_predictions[name] = preds

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "Paradigm": (
                "Bagging" if "Bagging" in name else
                "Boosting" if "Boosting" in name else
                "Stacking" if "Stacking" in name else
                "Deep Learning"
            ),
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })
        prediction_data[name] = preds
        print(f"    MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # 8. Hybrid Model: RF + ANN Blended
    print("\n--> Evaluating Hybrid Model: Random Forest + ANN Blended...")
    hybrid_preds = (
        0.65 * trained_predictions["Random Forest (Bagging)"] +
        0.35 * trained_predictions["Artificial Neural Network (ANN)"]
    )
    h_mae = mean_absolute_error(y_test, hybrid_preds)
    h_rmse = np.sqrt(mean_squared_error(y_test, hybrid_preds))
    h_r2 = r2_score(y_test, hybrid_preds)

    results.append({
        "Model": "RF + ANN Hybrid",
        "Paradigm": "Hybridization",
        "MAE": h_mae,
        "RMSE": h_rmse,
        "R2 Score": h_r2
    })
    prediction_data["RF + ANN Hybrid"] = hybrid_preds
    print(f"    MAE: {h_mae:.4f} | RMSE: {h_rmse:.4f} | R²: {h_r2:.4f}")

    # Build and wrap hybrid model object for serialization
    hybrid_pipeline = RF_ANN_HybridRegressor(
        rf_pipeline=models["Random Forest (Bagging)"],
        ann_pipeline=models["Artificial Neural Network (ANN)"],
        rf_weight=0.65
    )

    # Save models to disk
    print("\n" + "=" * 60)
    print("5. SAVING TRAINED MODELS (.pkl) TO models/ DIRECTORY")
    print("=" * 60)

    model_filenames = {
        "Random Forest (Bagging)": "models/random_forest_model.pkl",
        "XGBoost (Boosting)": "models/xgboost_model.pkl",
        "LightGBM (Boosting)": "models/lightgbm_model.pkl",
        "CatBoost (Boosting)": "models/catboost_model.pkl",
        "Gradient Boosting (Boosting)": "models/gradient_boosting_model.pkl",
        "Stacking Ensemble": "models/stacking_model.pkl",
        "Artificial Neural Network (ANN)": "models/ann_model.pkl",
    }

    for name, filepath in model_filenames.items():
        joblib.dump(models[name], filepath)
        print(f"Saved: {filepath}")

    joblib.dump(hybrid_pipeline, "models/rf_ann_hybrid_model.pkl")
    print("Saved: models/rf_ann_hybrid_model.pkl")

    # Save metadata for frontend form dropdowns and feature schemas
    categorical_options = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_features}
    numeric_ranges = {col: {"min": float(df[col].min()), "max": float(df[col].max()), "median": float(df[col].median())} for col in numeric_features}

    metadata = {
        "features": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": categorical_options,
        "numeric_ranges": numeric_ranges,
        "target": TARGET_COLUMN
    }
    joblib.dump(metadata, "models/metadata.pkl")
    print("Saved: models/metadata.pkl")

    # 6. Save results & predictions CSV
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    results_df.to_csv("model_results.csv", index=False)
    prediction_data.to_csv("model_predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL BENCHMARK RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df

if __name__ == "__main__":
    train_and_evaluate_all()
