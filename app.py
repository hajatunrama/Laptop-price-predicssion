import streamlit as st
import pandas as pd
import joblib

# Load model dan encoders
model = joblib.load('laptop_price_prediction_model.pkl')
encoders = joblib.load('label_encoders.pkl')

st.title("Prediksi Harga Laptop 💻")

# Sidebar untuk input user
brand = st.selectbox("Brand", encoders['brand'].classes_)
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
rom = st.number_input("ROM/Storage (GB)", min_value=32, max_value=2048, value=512)
display = st.number_input("Display Size (inch)", min_value=11.0, max_value=18.0, value=15.6)
processor = st.selectbox("Processor", encoders['processor'].classes_)

# Tombol Prediksi
if st.button("Prediksi Harga"):
    # Siapkan input data
    input_data = pd.DataFrame({
        'brand': [encoders['brand'].transform([brand])[0]],
        'processor': [encoders['processor'].transform([processor])[0]],
        'Ram': [ram],
        'ROM': [rom],
        'display_size': [display],
        # Tambahkan kolom lain sesuai dengan yang dilatih di model
        # pastikan urutan kolom sesuai dengan df_ml.drop(columns=["price"])
    })
    
    # Prediksi
    prediction = model.predict(input_data)
    st.success(f"Estimasi Harga Laptop: Rp {prediction[0]:,.2f}")
