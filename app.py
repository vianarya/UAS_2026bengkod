import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediksi Customer Churn",
    page_icon="📊",
    layout="centered"
)

# ── CSS Custom ─────────────────────────────────────────────
st.markdown("""
<style>
.churn-box {
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 10px 0;
}
.tidak-churn-box {
    background: linear-gradient(135deg, #00c853, #1e8449);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 10px 0;
}
.churn-box h1, .tidak-churn-box h1 {
    font-size: 2.5rem;
    margin: 0;
}
.churn-box p, .tidak-churn-box p {
    font-size: 1.1rem;
    margin: 10px 0 0 0;
    opacity: 0.9;
}
.prob-churn {
    background-color: #ffe0e0;
    border-left: 5px solid #e74c3c;
    padding: 12px 18px;
    border-radius: 8px;
    margin: 6px 0;
    font-weight: bold;
    color: #c0392b;
}
.prob-aman {
    background-color: #e0ffe0;
    border-left: 5px solid #27ae60;
    padding: 12px 18px;
    border-radius: 8px;
    margin: 6px 0;
    font-weight: bold;
    color: #1e8449;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model          = joblib.load('best_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_cols   = joblib.load('feature_columns.pkl')
    scaler         = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
    return model, label_encoders, feature_cols, scaler

model, label_encoders, feature_cols, scaler = load_artifacts()

# ── Header ─────────────────────────────────────────────────
st.title("📊 Prediksi Customer Churn")
st.caption("Universitas Dian Nuswantoro — Bengkel Koding Data Science")
st.markdown("Masukkan data pelanggan untuk memprediksi apakah pelanggan akan **Churn** atau **Tidak Churn**.")
st.divider()

# ── Input Form ─────────────────────────────────────────────
st.subheader("📋 Data Pelanggan")

age                        = st.number_input("Age", min_value=18, max_value=100, value=28)
satisfaction_score         = st.slider("Satisfaction Score", 1, 10, 4)
total_spent                = st.number_input("Total Spent", min_value=0.0, value=1000.0)
support_tickets            = st.number_input("Support Tickets", min_value=0, max_value=100, value=1)
device_type                = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
lifetime_value             = st.number_input("Lifetime Value", min_value=0.0, value=5000.0)
marketing_spend_per_user   = st.number_input("Marketing Spend Per User", min_value=0.0, value=500.0)
email_open_rate            = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
avg_session_time           = st.number_input("Average Session Time", min_value=0.0, value=30.0)
avg_order_value            = st.number_input("Average Order Value", min_value=0.0, value=200.0)
country                    = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "Germany", "France", "India", "Other"])
is_premium_user            = st.selectbox("Premium User", [0, 1], format_func=lambda x: f"{x} - {'Ya' if x==1 else 'Tidak'}")
pages_per_session          = st.number_input("Pages Per Session", min_value=0.0, value=5.0)
payment_method             = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"])

# Nilai default untuk fitur tambahan (tidak ditampilkan ke user)
gender                     = "Male"
city                       = "New York"
acquisition_channel        = "Email"
subscription_type          = "Basic"
total_visits               = 50
email_click_rate           = 0.3
discount_used              = 0
coupon_code                = "NONE"
refund_requested           = 0
delivery_delay_days        = 2
nps_score                  = 30
last_3_month_purchase_freq = 3
signup_year                = 2020
signup_month               = 6
purchase_year              = 2023
purchase_month             = 1
st.divider()

# ── Predict ────────────────────────────────────────────────
if st.button("🔍 Prediksi Churn", use_container_width=True, type="primary"):

    input_dict = {
        'age': age,
        'satisfaction_score': satisfaction_score,
        'total_spent': total_spent,
        'support_tickets': support_tickets,
        'device_type': device_type,
        'lifetime_value': lifetime_value,
        'marketing_spend_per_user': marketing_spend_per_user,
        'email_open_rate': email_open_rate,
        'avg_session_time': avg_session_time,
        'avg_order_value': avg_order_value,
        'country': country,
        'is_premium_user': is_premium_user,
        'pages_per_session': pages_per_session,
        'payment_method': payment_method,
        'gender': gender,
        'city': city,
        'acquisition_channel': acquisition_channel,
        'subscription_type': subscription_type,
        'total_visits': total_visits,
        'email_click_rate': email_click_rate,
        'discount_used': discount_used,
        'coupon_code': coupon_code,
        'refund_requested': refund_requested,
        'delivery_delay_days': delivery_delay_days,
        'nps_score': nps_score,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
        'signup_year': signup_year,
        'signup_month': signup_month,
        'purchase_year': purchase_year,
        'purchase_month': purchase_month,
    }

    input_df = pd.DataFrame([input_dict])

    # Encode
    for col, le in label_encoders.items():
        if col in input_df.columns:
            val = str(input_df[col].iloc[0])
            input_df[col] = le.transform([val])[0] if val in le.classes_ else 0

    # Reindex sesuai fitur model
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Tampilkan data input
    st.markdown("**📄 Data yang diinput:**")
    st.dataframe(input_df, use_container_width=True)

    # Scaling
    input_final = scaler.transform(input_df) if scaler else input_df.values

    # Prediksi dengan threshold 0.3 (lebih sensitif mendeteksi churn)
    prob = model.predict_proba(input_final)[0] if hasattr(model, 'predict_proba') else None
    THRESHOLD = 0.3
    if prob is not None:
        pred = 1 if prob[1] >= THRESHOLD else 0
    else:
        pred = model.predict(input_final)[0]

    st.divider()
    st.subheader("🎯 Hasil Prediksi")

    # ── Kotak hasil utama ──────────────────────────
    if pred == 1:
        st.markdown("""
        <div class='churn-box'>
            <h1>🚨 Pelanggan Churn</h1>
            <p>Pelanggan ini <b>BERPOTENSI BERHENTI</b> menggunakan layanan!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='tidak-churn-box'>
            <h1>✅ Pelanggan Tidak Churn</h1>
            <p>Pelanggan ini <b>MASIH AKTIF</b> dan loyal terhadap layanan.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:grey; font-size:12px;'>
    UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro<br>
    Customer Churn Prediction menggunakan Machine Learning
</div>
""", unsafe_allow_html=True)