# app.py
import streamlit as st
import pandas as pd
from subterra_model import (
    load_and_preprocess_data, split_data,
    train_models, evaluate_models,
    get_feature_importance, predict_single_sample,
    feature_names
)

st.set_page_config(page_title="SubTerra AI Classifier", layout="wide")
st.title("🔬 SubTerra: Geothermal Site Classifier")

# Sidebar
st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Batch Prediction (CSV)", "Single Prediction (Manual Input)"])
model_choice = st.sidebar.selectbox("Model", ["decision_tree", "random_forest", "xgboost"])

if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

# Upload and Train Section
st.header("1️⃣ Upload and Train Model")
uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type=["csv"])

if uploaded_file:
    with st.spinner("Processing and training..."):
        try:
            X, y = load_and_preprocess_data(uploaded_file)
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = split_data(X, y)
            train_models(X_train, y_train, X_train_scaled)
            results, best_model = evaluate_models(X_test, y_test, X_test_scaled)
            st.success(f"Models trained! Best model: {best_model} with {results[best_model]*100:.2f}% accuracy")
            st.session_state.models_trained = True
        except Exception as e:
            st.error(f"Error: {e}")

# Batch Prediction
if mode == "Batch Prediction (CSV)":
    st.header("📦 Batch Prediction")
    pred_file = st.file_uploader("Upload CSV for Prediction", key="pred_upload")

    if pred_file and st.session_state.models_trained:
        try:
            df = pd.read_csv(pred_file)
            X_input = df[feature_names].copy()
            results = []
            for _, row in X_input.iterrows():
                prediction = predict_single_sample(row.to_dict(), model_choice)
                results.append(prediction)
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)
            st.download_button("Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Manual Input
elif mode == "Single Prediction (Manual Input)":
    st.header("🧪 Single Site Input")
    if st.session_state.models_trained:
        user_input = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            user_input[feat] = cols[i % 3].number_input(f"{feat}", value=0.0)

        if st.button("Predict"):
            result = predict_single_sample(user_input, model_choice)
            st.metric(label="Prediction", value=result['label'])
            if result['probabilities'] is not None:
                st.write("Probability Distribution:")
                st.write({f"Class {i}": round(p, 4) for i, p in enumerate(result['probabilities'])})
    else:
        st.warning("Please upload training data and train the models first.")

# Feature Importance
if st.session_state.models_trained:
    st.header("📊 Feature Importance")
    importance = get_feature_importance(model_choice)
    if importance:
        top_k = pd.Series(importance).sort_values(ascending=False).head(15)
        st.bar_chart(top_k)
    else:
        st.info("Selected model does not support feature importance.")

