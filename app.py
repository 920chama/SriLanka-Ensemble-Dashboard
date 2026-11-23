import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.base import BaseEstimator, RegressorMixin
import __main__

# HYBRID MODEL WRAPPER DEFINITION

class RF_ANN_HybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rf_pipeline=None, ann_pipeline=None, rf_weight=0.65):
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

# Ensure pickle can locate the class under both __main__ and local namespace
__main__.RF_ANN_HybridRegressor = RF_ANN_HybridRegressor


# PAGE CONFIGURATION & CACHING

st.set_page_config(
    page_title="Sri Lanka Vehicle Price Ensemble Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    df = pd.read_csv("data/new_car_price_dataset.csv")
    df = df.drop(columns=["Unnamed: 0", "Date"], errors="ignore")
    # Convert target price to Million LKR (1 Million = 1,000,000 LKR)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce") / 10.0
    return df.dropna(subset=["Price"])


@st.cache_data
def load_results_and_predictions():
    results_df = pd.read_csv("model_results.csv")
    predictions_df = pd.read_csv("model_predictions.csv")
    feature_imp_df = pd.read_csv("feature_importance.csv")
    return results_df, predictions_df, feature_imp_df


@st.cache_resource
def load_backend_models():
    model_files = {
        "RF + ANN Hybrid": "models/rf_ann_hybrid_model.pkl",
        "Random Forest (Bagging)": "models/random_forest_model.pkl",
        "Stacking Ensemble": "models/stacking_model.pkl",
        "LightGBM (Boosting)": "models/lightgbm_model.pkl",
        "XGBoost (Boosting)": "models/xgboost_model.pkl",
        "CatBoost (Boosting)": "models/catboost_model.pkl",
        "Gradient Boosting (Boosting)": "models/gradient_boosting_model.pkl",
        "Artificial Neural Network (ANN)": "models/ann_model.pkl"
    }
    loaded_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)

    metadata = joblib.load("models/metadata.pkl") if os.path.exists("models/metadata.pkl") else None
    return loaded_models, metadata


df = load_dataset()
results_df, predictions_df, feature_imp_df = load_results_and_predictions()
backend_models, metadata = load_backend_models()


# SIDEBAR NAVIGATION

st.sidebar.title("🚗 Navigation")

app_section = st.sidebar.radio(
    "Select Section:",
    [
        "📊 Exploratory Data Analysis (EDA)",
        "🏆 Model Comparison & Benchmarks",
        "🔍 Feature Importance",
        "🔮 Live Price Prediction"
    ]
)

st.sidebar.markdown("---")


# HEADER & SUMMARY METRICS

st.markdown('<div class="main-title">🚗 Sri Lankan Vehicle Price Ensemble Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Comparative Analysis of Bagging, Boosting, Stacking, and Hybrid Models on Sri Lankan Automotive Resale Data</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Average Price", f"{df['Price'].mean():.2f} Million LKR")
with col3:
    st.metric("Unique Brands", f"{df['Brand'].nunique()}")
with col4:
    best_model = results_df.iloc[0]
    st.metric("Top Model", f"{best_model['Model']}")
with col5:
    st.metric("Top R² Score", f"{best_model['R2 Score']:.4f}")

st.divider()


# EXPLORATORY DATA ANALYSIS (EDA)

if app_section == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    st.write("Visual exploration of vehicle resale prices, top brands, fuel distributions, and market characteristics.")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df,
            x="Price",
            nbins=50,
            title="Distribution of Vehicle Prices (Million LKR)",
            labels={"Price": "Price (Million LKR)"},
            color_discrete_sequence=["#2563EB"]
        )
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        top_brands = df["Brand"].value_counts().head(10).reset_index()
        top_brands.columns = ["Brand", "Count"]
        fig_brands = px.bar(
            top_brands,
            x="Brand",
            y="Count",
            text="Count",
            title="Top 10 Vehicle Brands in Sri Lanka",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig_brands.update_layout(template="plotly_white")
        st.plotly_chart(fig_brands, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fuel_dist = df["Fuel Type"].value_counts().reset_index()
        fuel_dist.columns = ["Fuel Type", "Count"]
        fig_fuel = px.pie(
            fuel_dist,
            names="Fuel Type",
            values="Count",
            title="Market Share by Fuel Type",
            hole=0.45
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

    with c4:
        gear_dist = df["Gear"].value_counts().reset_index()
        gear_dist.columns = ["Gear Type", "Count"]
        fig_gear = px.pie(
            gear_dist,
            names="Gear Type",
            values="Count",
            title="Market Share by Transmission (Gear)",
            hole=0.45
        )
        st.plotly_chart(fig_gear, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig_yom = px.box(
            df[df["YOM"] >= 2000],
            x="YOM",
            y="Price",
            title="Price Distribution Across Year of Manufacture (>= 2000)",
            labels={"YOM": "Year of Manufacture", "Price": "Price (Million LKR)"},
            color_discrete_sequence=["#3B82F6"]
        )
        fig_yom.update_layout(template="plotly_white")
        st.plotly_chart(fig_yom, use_container_width=True)

    with c6:
        sampled_df = df.sample(min(1200, len(df)), random_state=42)
        fig_mileage = px.scatter(
            sampled_df,
            x="Millage(KM)",
            y="Price",
            color="Fuel Type",
            title="Price vs. Mileage (Sampled 1,200 records)",
            labels={"Millage(KM)": "Mileage (KM)", "Price": "Price (Million LKR)"},
            opacity=0.65
        )
        fig_mileage.update_layout(template="plotly_white")
        st.plotly_chart(fig_mileage, use_container_width=True)



# MODEL COMPARISON & BENCHMARKS

elif app_section == "🏆 Model Comparison & Benchmarks":
    st.header("🏆 Ensemble Model Comparison & Benchmarks")
    st.write(
        "Empirical benchmark comparing all three required ensemble paradigms (**Bagging**, **Boosting**, **Stacking**), a **Hybrid Ensemble**, and a **Single-Model Baseline (ANN)**."
    )

    st.subheader("📋 Leaderboard Comparison")
    styled_df = results_df.copy()
    styled_df["MAE (Million LKR)"] = styled_df["MAE"].map("{:.4f}".format)
    styled_df["RMSE (Million LKR)"] = styled_df["RMSE"].map("{:.4f}".format)
    styled_df["R² Score"] = styled_df["R2 Score"].map("{:.4f}".format)
    styled_df["Ensemble Type"] = styled_df["Paradigm"].map({
        "Hybridization": "Hybrid Ensemble (RF + ANN)",
        "Bagging": "Bagging Ensemble (150 Trees)",
        "Stacking": "Stacking Meta-Ensemble (4 Base Models)",
        "Boosting": "Gradient Boosted Tree Ensemble",
        "Deep Learning": "Single Baseline (Non-Ensemble)"
    })
    st.dataframe(styled_df[["Model", "Ensemble Type", "MAE (Million LKR)", "RMSE (Million LKR)", "R² Score"]], use_container_width=True)

    st.subheader("📊 Performance Visualizations")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_r2 = px.bar(
            results_df.sort_values(by="R2 Score", ascending=True),
            x="R2 Score",
            y="Model",
            orientation="h",
            color="Paradigm",
            title="R² Score (Higher is Better)",
            text=results_df.sort_values(by="R2 Score", ascending=True)["R2 Score"].apply(lambda x: f"{x:.3f}")
        )
        fig_r2.update_layout(template="plotly_white", xaxis_range=[0.8, 0.95])
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_rmse = px.bar(
            results_df.sort_values(by="RMSE", ascending=False),
            x="RMSE",
            y="Model",
            orientation="h",
            color="Paradigm",
            title="RMSE in Million LKR (Lower is Better)",
            text=results_df.sort_values(by="RMSE", ascending=False)["RMSE"].apply(lambda x: f"{x:.2f}")
        )
        fig_rmse.update_layout(template="plotly_white")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c3:
        fig_mae = px.bar(
            results_df.sort_values(by="MAE", ascending=False),
            x="MAE",
            y="Model",
            orientation="h",
            color="Paradigm",
            title="MAE in Million LKR (Lower is Better)",
            text=results_df.sort_values(by="MAE", ascending=False)["MAE"].apply(lambda x: f"{x:.2f}")
        )
        fig_mae.update_layout(template="plotly_white")
        st.plotly_chart(fig_mae, use_container_width=True)

    st.divider()

    st.subheader("🎯 Actual vs. Predicted Resale Price")
    selected_eval_model = st.selectbox(
        "Select Model to Inspect:",
        results_df["Model"].tolist()
    )

    if selected_eval_model in predictions_df.columns:
        c1, c2 = st.columns([2, 1])
        with c1:
            eval_data = predictions_df[["Actual", selected_eval_model]].dropna()
            fig_scatter = px.scatter(
                eval_data.sample(min(800, len(eval_data)), random_state=42),
                x="Actual",
                y=selected_eval_model,
                opacity=0.6,
                labels={"Actual": "Actual Price (Million LKR)", selected_eval_model: f"Predicted Price ({selected_eval_model})"},
                title=f"Actual vs. Predicted Prices ({selected_eval_model})"
            )
            min_val = min(eval_data["Actual"].min(), eval_data[selected_eval_model].min())
            max_val = max(eval_data["Actual"].max(), eval_data[selected_eval_model].max())
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash", color="red", width=2),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            st.markdown(f"#### Model Snapshot: **{selected_eval_model}**")
            row = results_df[results_df["Model"] == selected_eval_model].iloc[0]
            st.info(f"**Paradigm:** {row['Paradigm']}")
            st.metric("R² Score", f"{row['R2 Score']:.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{row['MAE']:.2f} Million LKR")
            st.metric("Root Mean Squared Error (RMSE)", f"{row['RMSE']:.2f} Million LKR")



# FEATURE IMPORTANCE
elif app_section == "🔍 Feature Importance":
    st.header("🔍 Permutation Feature Importance")
    st.write(
        "Permutation feature importance measures the increase in model prediction error when a feature is shuffled."
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_feat = px.bar(
            feature_imp_df.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            error_x="Std",
            title="Feature Permutation Importance (MAE Increase in Million LKR)",
            color="Importance",
            color_continuous_scale="Viridis",
            labels={"Importance": "Drop in MAE when Shuffled (Million LKR)"}
        )
        fig_feat.update_layout(template="plotly_white")
        st.plotly_chart(fig_feat, use_container_width=True)

    with c2:
        st.subheader("Key Drivers of Vehicle Price")
        st.markdown("""
        1. **Year of Manufacture (YOM - Importance: ~1.09 Million LKR):**
           - Highest impact factor. Vehicles manufactured more recently carry substantial scarcity premiums in Sri Lanka.
        2. **Brand & Model (Importance: ~0.96 Million LKR):**
           - Japanese brands (Toyota, Suzuki, Nissan, Honda) hold high market liquidity and superior value retention.
        3. **Engine Capacity (cc - Importance: ~0.73 Million LKR):**
           - Directly linked to vehicle size category, fuel efficiency, and excise duty tiers.
        4. **Mileage (KM - Importance: ~0.63 Million LKR):**
           - Primary indicator of mechanical usage and condition.
        5. **Transmission (Gear - Importance: ~0.58 Million LKR):**
           - Automatic transmission commands a persistent price premium over Manual in urban centers.
        """)

    st.dataframe(feature_imp_df.style.format({"Importance": "{:.4f}", "Std": "{:.4f}"}), use_container_width=True)


# LIVE VEHICLE PRICE PREDICTION
elif app_section == "🔮 Live Price Prediction":
    st.header("🔮 Live Vehicle Resale Price Prediction")
    st.write("Enter vehicle specifications to estimate its market resale value in **Million LKR** using trained ensemble models.")

    if metadata is None:
        st.error("Metadata not found. Please train models first using train_models.py.")
    else:
        with st.form("prediction_form"):
            col_b1, col_b2, col_b3 = st.columns(3)

            with col_b1:
                brand_options = metadata["categorical_options"]["Brand"]
                popular_brands = ["TOYOTA", "SUZUKI", "NISSAN", "HONDA", "MITSUBISHI", "MERCEDES-BENZ", "BMW", "HYUNDAI", "KIA"]
                sorted_brands = [b for b in popular_brands if b in brand_options] + [b for b in brand_options if b not in popular_brands]
                selected_brand = st.selectbox("Vehicle Brand", sorted_brands, index=0)

                filtered_models = sorted(df[df["Brand"] == selected_brand]["Model"].dropna().unique().tolist())
                if not filtered_models:
                    filtered_models = metadata["categorical_options"]["Model"]
                selected_model = st.selectbox("Vehicle Model", filtered_models, index=0)

                selected_yom = st.slider(
                    "Year of Manufacture (YOM)",
                    min_value=1990,
                    max_value=2024,
                    value=2018,
                    step=1
                )

                selected_mileage = st.number_input(
                    "Mileage (KM)",
                    min_value=500.0,
                    max_value=400000.0,
                    value=45000.0,
                    step=1000.0
                )

            with col_b2:
                selected_engine = st.number_input(
                    "Engine Capacity (cc)",
                    min_value=650.0,
                    max_value=5000.0,
                    value=1500.0,
                    step=100.0
                )

                selected_gear = st.selectbox(
                    "Gear / Transmission",
                    metadata["categorical_options"]["Gear"],
                    index=0
                )

                selected_fuel = st.selectbox(
                    "Fuel Type",
                    metadata["categorical_options"]["Fuel Type"],
                    index=0
                )

                town_options = metadata["categorical_options"]["Town"]
                major_towns = ["Colombo", "Gampaha", "Kandy", "Kurunegala", "Kalutara", "Galle", "Matara", "Negombo"]
                sorted_towns = [t for t in major_towns if t in town_options] + [t for t in town_options if t not in major_towns]
                selected_town = st.selectbox("Location / Town", sorted_towns, index=0)

            with col_b3:
                selected_condition = st.selectbox(
                    "Condition",
                    metadata["categorical_options"]["Condition"],
                    index=0
                )

                selected_leasing = st.selectbox(
                    "Leasing Status",
                    metadata["categorical_options"]["Leasing"],
                    index=0
                )

                selected_ac = st.selectbox("Air Conditioning", ["Available", "Not_Available"], index=0)
                selected_ps = st.selectbox("Power Steering", ["Available", "Not_Available"], index=0)
                selected_pm = st.selectbox("Power Mirror", ["Available", "Not_Available"], index=0)
                selected_pw = st.selectbox("Power Window", ["Available", "Not_Available"], index=0)

            st.markdown("---")
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                selected_model_name = st.selectbox(
                    "Select Model for Prediction:",
                    ["All Models Consensus (Ensemble View)"] + list(backend_models.keys()),
                    index=0
                )
            with col_m2:
                st.write("")
                st.write("")
                predict_button = st.form_submit_button("🚀 Run Live Price Prediction", use_container_width=True)

        if predict_button:
            input_dict = {
                "Brand": selected_brand,
                "Model": selected_model,
                "YOM": selected_yom,
                "Engine (cc)": selected_engine,
                "Gear": selected_gear,
                "Fuel Type": selected_fuel,
                "Millage(KM)": selected_mileage,
                "Town": selected_town,
                "Leasing": selected_leasing,
                "Condition": selected_condition,
                "AIR CONDITION": selected_ac,
                "POWER STEERING": selected_ps,
                "POWER MIRROR": selected_pm,
                "POWER WINDOW": selected_pw
            }
            input_df = pd.DataFrame([input_dict])

            if selected_model_name == "All Models Consensus (Ensemble View)":
                consensus_results = []
                for name, model in backend_models.items():
                    try:
                        pred_val = model.predict(input_df)[0]
                        consensus_results.append({"Model": name, "Predicted Price (Million LKR)": pred_val})
                    except Exception:
                        pass

                c_df = pd.DataFrame(consensus_results)
                mean_price = c_df["Predicted Price (Million LKR)"].mean()

                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="margin:0; color:white;">Ensemble Consensus Valuation</h3>
                    <h1 style="font-size: 2.8rem; margin: 10px 0; color:white;">{mean_price:.2f} Million LKR</h1>
                    <p style="font-size: 1.15rem; color:#E0E7FF; margin:0;">
                        Approx. <b>LKR {mean_price * 1000000:,.0f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    fig_consensus = px.bar(
                        c_df.sort_values(by="Predicted Price (Million LKR)"),
                        x="Predicted Price (Million LKR)",
                        y="Model",
                        orientation="h",
                        title="Model Breakdown for this Vehicle (Million LKR)",
                        color="Predicted Price (Million LKR)",
                        color_continuous_scale="Blues"
                    )
                    fig_consensus.update_layout(template="plotly_white")
                    st.plotly_chart(fig_consensus, use_container_width=True)

                with col_res2:
                    st.markdown("#### Model Predictions:")
                    c_df["Price (Million LKR)"] = c_df["Predicted Price (Million LKR)"].map("{:.2f} Million".format)
                    c_df["Exact LKR"] = (c_df["Predicted Price (Million LKR)"] * 1000000).map("Rs. {:,.0f}".format)
                    st.dataframe(c_df[["Model", "Price (Million LKR)", "Exact LKR"]], use_container_width=True)

            else:
                target_model = backend_models[selected_model_name]
                predicted_val = target_model.predict(input_df)[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="margin:0; color:white;">{selected_model_name} Prediction</h3>
                    <h1 style="font-size: 3rem; margin: 10px 0; color:white;">{predicted_val:.2f} Million LKR</h1>
                    <p style="font-size: 1.15rem; color:#E0E7FF; margin:0;">
                        Approx. <b>LKR {predicted_val * 1000000:,.0f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"**Vehicle:** {selected_yom} {selected_brand} {selected_model} | {selected_gear} | {selected_fuel} | {selected_mileage:,.0f} KM | {selected_town}")
