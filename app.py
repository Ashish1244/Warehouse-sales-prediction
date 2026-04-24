# app.py
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}
h1,h2,h3 {
    color: #f8fafc;
}
div[data-testid="stMetricValue"] {
    color: #22c55e;
}
.stButton>button {
    background: linear-gradient(to right,#06b6d4,#3b82f6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
}
.stButton>button:hover {
    transform: scale(1.02);
}
section[data-testid="stSidebar"] {
    background: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (UPDATED FILE NAME ONLY)
# ---------------------------------------------------
MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Put model.joblib in same folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("📈 Smart Sales Prediction Dashboard")
st.write("Predict future sales instantly using your trained ML model.")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("📥 Enter Product Details")

year = st.sidebar.slider("Year", 2020, 2030, 2025)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
supplier = st.sidebar.number_input("Supplier ID", min_value=0, value=1)
item_code = st.sidebar.number_input("Item Code", min_value=0, value=1001)
item_desc = st.sidebar.number_input("Item Description ID", min_value=0, value=10)
item_type = st.sidebar.number_input("Item Type ID", min_value=0, value=1)
retail_transfers = st.sidebar.number_input("Retail Transfers", min_value=0.0, value=50.0)
warehouse_sales = st.sidebar.number_input("Warehouse Sales", min_value=0.0, value=100.0)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
if st.sidebar.button("🔮 Predict Sales"):

    input_data = np.array([[
        year,
        month,
        supplier,
        item_code,
        item_desc,
        item_type,
        retail_transfers,
        warehouse_sales
    ]])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction Completed ✅")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Sales", f"{prediction:,.2f}")

        with col2:
            if prediction > 500:
                st.metric("Category", "High 🚀")
            elif prediction > 100:
                st.metric("Category", "Medium 📈")
            else:
                st.metric("Category", "Low ⚠️")

        with col3:
            st.metric("Growth Chance", "Strong")

        chart_df = pd.DataFrame({
            "Metric": ["Retail Transfers", "Warehouse Sales", "Prediction"],
            "Value": [retail_transfers, warehouse_sales, prediction]
        })

        st.subheader("📊 Comparison Chart")
        st.bar_chart(chart_df.set_index("Metric"))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.write("Built with ❤️ using Streamlit")