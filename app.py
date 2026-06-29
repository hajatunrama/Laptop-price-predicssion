import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. Load Model & Encoders ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'laptop_price_prediction_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

model, encoders = load_assets()

# --- 2. Antarmuka Ringkas ---
st.title("💻 Laptop Price Prediction")
st.write("Silakan masukkan spesifikasi untuk melihat estimasi harga.")

# Menggunakan expander agar form tidak memakan tempat terlalu banyak
with st.expander("Klik untuk mengatur spesifikasi laptop", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", encoders['brand'].classes_)
        processor = st.selectbox("Processor", encoders['processor'].classes_)
        ram = st.select_slider("RAM (GB)", options=[4, 8, 12, 16, 24, 32, 64])
        rom = st.select_slider("Penyimpanan (GB)", options=[64, 128, 256, 512, 1024, 2048])
    
    with col2:
        # Mengelompokkan pilihan yang kurang kritis
        with st.expander("Pengaturan Lanjutan (GPU, OS, Resolusi)"):
            gpu = st.selectbox("GPU", encoders['GPU'].classes_)
            os = st.selectbox("Sistem Operasi", encoders['OS'].classes_)
            res_width = st.selectbox("Resolusi Width", [1366, 1920, 2240, 2560, 3840])
            res_height = st.selectbox("Resolusi Height", [768, 1080, 1400, 1600, 2160])
        
        display_size = st.number_input("Ukuran Layar (Inch)", 11.0, 18.0, 15.6)
        warranty = st.radio("Garansi (Tahun)", [0, 1, 2, 3], horizontal=True)

# --- 3. Proses Prediksi ---
if st.button("🔍 Cek Estimasi Harga", type="primary"):
    try:
        input_data = pd.DataFrame({
            'brand': [encoders['brand'].transform([brand])[0]],
            'spec_rating': [69.38], # Nilai rata-rata dataset
            'processor': [encoders['processor'].transform([processor])[0]],
            'CPU': [0], # Sesuaikan jika kolom CPU masih digunakan
            'Ram': [int(ram)],
            'Ram_type': [0], # Sesuaikan dengan encoder jika perlu
            'ROM': [int(rom)],
            'ROM_type': [0], # Sesuaikan dengan encoder jika perlu
            'GPU': [encoders['GPU'].transform([gpu])[0]],
            'display_size': [float(display_size)],
            'resolution_width': [float(res_width)],
            'resolution_height': [float(res_height)],
            'OS': [encoders['OS'].transform([os])[0]],
            'warranty': [int(warranty)]
        })
        
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimasi Harga: Rp {prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
