import streamlit as st
import numpy as np
import joblib

# ✅ Load model and scaler
model = joblib.load("expresso_model.pkl")
scaler = joblib.load("expresso_scaler.pkl")

st.title("📱 Expresso Churn Prediction App")

# 🎯 Input fields (in the exact order of model training)
region = st.selectbox("Region (encoded)", options=range(0, 20))  # adjust range as needed
tenure = st.number_input("Tenure (months)", min_value=0)
montant = st.number_input("Recharge Amount", min_value=0)
frequence_rech = st.number_input("Recharge Frequency", min_value=0)
revenue = st.number_input("Monthly Revenue", min_value=0)
arpu_segment = st.number_input("ARPU over 90 Days", min_value=0)
frequence = st.number_input("Call Frequency", min_value=0)
data_volume = st.number_input("Data Volume Used", min_value=0)
on_net = st.number_input("On-net Calls", min_value=0)
orange = st.number_input("Calls to Orange", min_value=0)
tigo = st.number_input("Calls to Tigo", min_value=0)
zone1 = st.number_input("Calls to Zone 1", min_value=0)
zone2 = st.number_input("Calls to Zone 2", min_value=0)
mrg = st.selectbox("Is Client Pre-Churn (MRG)?", [0, 1])
regularity = st.number_input("Activity Regularity", min_value=0)
top_pack = st.selectbox("Top Pack (encoded)", options=[0, 1, 2, 3, 4, 5])  # adjust as needed
freq_top_pack = st.number_input("Top Pack Frequency", min_value=0)

# 🧮 Form the input array (match training order)
input_data = np.array([[region, tenure, montant, frequence_rech, revenue,
                        arpu_segment, frequence, data_volume, on_net, orange,
                        tigo, zone1, zone2, mrg, regularity, top_pack, freq_top_pack]])

# 🔍 Prediction button
if st.button("🔎 Predict Churn"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Prediction: {'❌ Churn' if prediction == 1 else '✅ Active'}")
