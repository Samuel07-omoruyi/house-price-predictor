import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path('models')
MODEL_FILE = MODEL_DIR / 'model.joblib'
META_FILE = MODEL_DIR / 'meta.json'

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

if not MODEL_FILE.exists() or not META_FILE.exists():
    st.warning("Model or metadata not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_FILE)
meta = json.loads(META_FILE.read_text())

st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["California (form)", "CSV (upload row)"])

# Ensure prediction is always positive
def sanitize_pred(pred, min_price=1):
    """Replace zero or negative predictions with min_price."""
    return max(pred, min_price)

if mode == "California (form)":
    st.header("Predict using features")

    numeric_cols = meta.get('numeric_columns', [])
    user_input = {}
    for col in numeric_cols:
        val = st.number_input(col, value=0.0)
        user_input[col] = val

    X_user = pd.DataFrame([user_input])

    if st.button("Predict"):
        raw_pred = model.predict(X_user)[0]
        pred = sanitize_pred(raw_pred, min_price=1)  # always positive
        st.success(f"Predicted House Price: ${pred:,.2f}")

else:
    st.header("Upload CSV")
    uploaded = st.file_uploader("CSV file", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)

        if st.button("Predict from CSV"):
            raw_preds = model.predict(df)
            preds = [sanitize_pred(p, min_price=1) for p in raw_preds]
            df['Predicted_Price'] = preds
            st.success("Predicted House Prices:")
            st.dataframe(df)
