# 🚗 Sri Lankan Vehicle Resale Price Prediction & Ensemble Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Ensemble-Bagging%20%7C%20Boosting%20%7C%20Stacking%20%7C%20Hybrid-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Course-CM4371%20Level%204-purple.svg)](https://uom.lk)


---

## Executive Summary
This project presents an end-to-end Machine Learning research benchmark and interactive web dashboard for predicting **vehicle resale prices in Sri Lanka**. Using a curated local dataset of **9,788 real commercial listings** (from Ikman.lk / Riyasewana), we empirically compare all three major ensemble learning paradigms (**Bagging**, **Boosting**, and **Stacking**), benchmark them against an **Artificial Neural Network (ANN)**, and introduce a novel **Tabular Hybrid Model (Random Forest + ANN)** that achieves superior predictive accuracy ($R^2 = 0.9128$, $\text{MAE} = 0.47\text{ Million LKR}$).

---

## Live Interactive Dashboard
The dashboard is built using **Streamlit** and features:
1. ** Exploratory Data Analysis (EDA):** Interactive distributions of prices, top Sri Lankan vehicle brands, fuel types, transmission shares, and mileage relationships.
2. ** Model Comparison & Benchmarks:** Comprehensive evaluation table and interactive charts for all 8 trained models across MAE, RMSE, and $R^2$ scores, including actual-vs-predicted scatter plots.
3. ** Feature Importance:** Permutation importance analysis breaking down why Year of Manufacture (YOM), Brand, Engine CC, and Mileage dominate vehicle values.
4. ** Live Vehicle Resale Price Predictor:** Interactive form with dynamic model filtering by brand, allowing users to select individual models or run an **"All Models Consensus"** estimate denominated in **Million LKR** (alongside exact LKR amounts).

---

## Comprehensive Model Benchmark Leaderboard

All models were evaluated on an identical 20% hold-out test set ($N = 1,958$ listings), with all prices denominated in **Million LKR**:

| Rank | Model Name | Ensemble Paradigm | MAE (Million LKR) | RMSE (Million LKR) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **RF + ANN Hybrid** | **Hybridization** | **0.47** | **1.25** | **0.9128** |
| 2 | **Random Forest** | **Bagging** | 0.47 | 1.28 | 0.9093 |
| 3 | **Stacking Ensemble** | **Stacking** | 0.59 | 1.33 | 0.9009 |
| 4 | **Artificial Neural Network (ANN)** | **Deep Learning** | 0.61 | 1.45 | 0.8823 |
| 5 | **LightGBM** | **Boosting** | 0.72 | 1.51 | 0.8735 |
| 6 | **XGBoost** | **Boosting** | 0.70 | 1.52 | 0.8705 |
| 7 | **CatBoost** | **Boosting** | 0.78 | 1.61 | 0.8554 |
| 8 | **Gradient Boosting** | **Boosting** | 0.79 | 1.71 | 0.8367 |

*Note: All prices are denominated in **Million LKR** (1 Million = 1,000,000 LKR).*

---

## Key Research Insights & Findings

### 1. Why Did the Hybrid Model (RF + ANN) Outperform All Others?
- **Random Forest** captures discrete local decision splits and non-linear interactions across vehicle brands and condition features with high precision, but produces piecewise constant step predictions without smooth continuous gradients.
- **ANN (MLPRegressor)** learns smooth continuous representations across continuous features (Year of Manufacture, Mileage, Engine CC), mitigating leaf-partition step artifacts.
- Ensembling both ($0.65 \times \text{RF} + 0.35 \times \text{ANN}$) provides variance reduction and complementary bias cancellation, achieving the lowest test error and highest $R^2$ score ($0.9128$).

### 2. Tabular Data vs. Text/Image Data (Research Reflection)
- **Inductive Bias:** Convolutional Neural Networks (CNNs) have translational equivariance inductive biases suitable for 2D spatial pixel grids; Transformers have self-attention mechanisms suited for sequential language. Tabular datasets lack spatial or sequential coordinate structure.
- **Axis-Aligned Splits:** Neural networks are rotationally invariant, whereas tabular features have distinct semantic identities (e.g., swapping `YOM` with `Millage` destroys semantic coherence). Decision tree ensembles inherently exploit axis-aligned coordinate splits, making them faster, more robust, and less prone to hyperparameter sensitivity on tabular data.

### 3. Sri Lankan Automotive Market Dynamics
- Sri Lanka's automotive import suspension enacted in 2020 created unprecedented vehicle appreciation in used Japanese brands (Toyota Premio, Axio, Vitz, Suzuki Alto/Wagon R). The models successfully captured these localized economic anomalies.

---

## Repository Structure

```
SriLanka-Ensemble-Dashboard/
├── data/
│   └── new_car_price_dataset.csv      # Cleaned Sri Lankan vehicle listings (9,788 rows)
├── models/
│   ├── random_forest_model.pkl        # Serialized Bagging model
│   ├── xgboost_model.pkl              # Serialized XGBoost model
│   ├── lightgbm_model.pkl             # Serialized LightGBM model
│   ├── catboost_model.pkl             # Serialized CatBoost model
│   ├── gradient_boosting_model.pkl    # Serialized Gradient Boosting model
│   ├── stacking_model.pkl             # Serialized Stacking Ensemble
│   ├── ann_model.pkl                  # Serialized Neural Network
│   ├── rf_ann_hybrid_model.pkl        # Serialized RF + ANN Hybrid Model
│   └── metadata.pkl                   # Feature metadata & categorical mappings
├── notebooks/
│   └── ensemble_vehicle_price_prediction.ipynb  # End-to-end clean execution notebook
├── app.py                             # Full interactive Streamlit Dashboard application
├── train_models.py                    # Complete model training & pipeline export script
├── feature_importance.py              # Permutation feature importance generator
├── feature_importance.csv             # Exported importance metrics
├── model_results.csv                  # Benchmark metrics table
├── model_predictions.csv              # Hold-out test predictions
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Comprehensive project documentation
```

---

## Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/920chama/SriLanka-Ensemble-Dashboard.git
cd SriLanka-Ensemble-Dashboard
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-Train Models
```bash
python train_models.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.


---

## AI Tools Policy Acknowledgment

In accordance with the **CM4371 Academic Integrity & GenAI Policy**:

- **Tools Utilized:** ChatGPT and Claude.
- **Where and How AI Was Used:**
  1. Syntactic scaffolding and widget structuring for the interactive Streamlit dashboard (`app.py`).
  2. Formatting of KaTeX LaTeX expressions and markdown tables for the documentation.
  3. Scikit-learn pipeline boilerplate composition and Joblib serialization helpers.
- **Author's Intellectual Contribution:**
  - Selected the Sri Lankan automotive domain and curated the local commercial dataset.
  - Formulated the hybrid modeling strategy (Random Forest + ANN) and configured hyperparameter tuning.
  - Validated and interpreted all model comparison metrics, feature importance rankings, and tabular vs. deep learning theoretical reflections.
  - Built, verified, and tested all pipelines and scripts end-to-end.
