import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load saved model & scaler ──
model    = joblib.load('churn_model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

# ── Page Config ──
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# ── Header ──
st.title("🔄 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will **churn or stay**.")
st.divider()

# ── Input Form ──
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    st.subheader("📱 Services")
    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Account Info")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total          = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly))

st.divider()

# ── Predict Button ──
if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):

    # Encode inputs same way as training
    encode = {
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0,
        'Month-to-month': 0, 'One year': 1, 'Two year': 2,
        'DSL': 0, 'Fiber optic': 1, 'No': 0,
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3,
        'No phone service': 2, 'No internet service': 2
    }

    def enc(val):
        return encode.get(val, 0)

    input_data = pd.DataFrame([[
        enc(gender), enc(senior), enc(partner), enc(dependents),
        tenure, enc(phone_service), enc(multiple_lines),
        enc(internet), enc(online_sec), enc(online_backup),
        enc(device_prot), enc(tech_support), enc(streaming_tv),
        enc(streaming_mov), enc(contract), enc(paperless),
        enc(payment), monthly, total
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(f"⚠️ This customer is **LIKELY TO CHURN**")
        else:
            st.success(f"✅ This customer is **LIKELY TO STAY**")

    with res_col2:
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.progress(float(probability))

    # Risk level
    if probability >= 0.7:
        st.warning("🔴 **High Risk** — Immediate action recommended!")
    elif probability >= 0.4:
        st.warning("🟡 **Medium Risk** — Monitor this customer closely.")
    else:
        st.info("🟢 **Low Risk** — Customer looks stable.")