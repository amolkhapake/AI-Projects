"""
Streamlit UI for Loan Default Prediction.
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from train import load_data, build_model

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦")


@st.cache_resource(show_spinner="Training model (one-time, ~20s)...")
def get_trained_model():
    df = load_data()
    X = df.drop(columns="default")
    y = df["default"]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = build_model(X_scaled.shape[1])
    model.fit(X_scaled, y, epochs=15, batch_size=64, verbose=0)
    return model, scaler, list(X.columns)


st.title("🏦 Loan Default Risk Predictor")
st.caption("ANN trained on applicant financial data. Model trains once and is cached.")

model, scaler, feature_names = get_trained_model()

with st.form("applicant_form"):
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual income ($)", 15000, 300000, 55000, step=1000)
        credit_score = st.slider("Credit score", 300, 850, 650)
        debt_to_income = st.slider("Debt-to-income ratio", 0.0, 0.8, 0.3, step=0.01)
    with col2:
        loan_amount = st.number_input("Loan amount ($)", 1000, 100000, 20000, step=500)
        employment_years = st.slider("Years employed", 0, 30, 5)
        num_late_payments = st.number_input("Late payments (last 2 yrs)", 0, 20, 0)

    submitted = st.form_submit_button("Predict default risk", use_container_width=True)

if submitted:
    row = pd.DataFrame([[income, credit_score, debt_to_income, loan_amount,
                          employment_years, num_late_payments]], columns=feature_names)
    scaled = scaler.transform(row)
    prob = float(model.predict(scaled, verbose=0)[0][0])

    st.metric("Predicted default probability", f"{prob:.1%}")
    if prob > 0.5:
        st.error("High risk — recommend manual underwriting review.")
    elif prob > 0.25:
        st.warning("Moderate risk — consider adjusted interest rate or additional collateral.")
    else:
        st.success("Low risk — eligible for standard approval terms.")

    st.progress(min(prob, 1.0))

st.divider()
st.caption(
    "Note: trained on synthetic data for demo purposes. In production, this "
    "connects to your bank's loan origination system and feature store, and "
    "predictions are logged with SHAP explanations for regulatory compliance."
)
