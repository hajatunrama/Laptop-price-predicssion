import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Harga Laptop", page_icon="💻", layout="wide")

# --- 2. Load Model & Encoders ---
# BASE_DIR membantu aplikasi menemukan file di server manapun dia dijalankan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'laptop_price_prediction_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        # Menampilkan pesan error spesifik jika file hilang
        st.error(f"File .pkl tidak ditemukan di {BASE_DIR}. Pastikan file sudah di-upload ke GitHub.")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

# Panggil fungsi load
model, encoders = load_assets()

# --- 3. Antarmuka (UI) ---
st.title("💻 Laptop Price Prediction App")
st.markdown("Masukkan spesifikasi laptop untuk mendapatkan estimasi harga.")
st.divider()

# Layout 3 kolom
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏢 Spesifikasi Utama")
    brand = st.selectbox("Brand Laptop", encoders['brand'].classes_)
    processor = st.selectbox("Tipe Processor", encoders['processor'].classes_)
    cpu = st.selectbox("Konfigurasi CPU", encoders['CPU'].classes_)
    gpu = st.selectbox("Kartu Grafis (GPU)", encoders['GPU'].classes_)
    os = st.selectbox("Sistem Operasi (OS)", encoders['OS'].classes_)

with col2:
    st.subheader("💾 Memori & Garansi")
    ram = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64])
    ram_type = st.selectbox("Tipe RAM", encoders['Ram_type'].classes_)
    rom = st.selectbox("Penyimpanan (GB)", [64, 128, 256, 512, 1024, 2048])
    rom_type = st.selectbox("Tipe Penyimpanan", encoders['ROM_type'].classes_)
    warranty = st.selectbox("Garansi (Tahun)", [0, 1, 2, 3])

with col3:
    st.subheader("🖥️ Layar & Rating")
    display_size = st.number_input("Ukuran Layar (Inch)", min_value=11.0, max_value=18.0, value=15.6, step=0.1)
    res_width = st.selectbox("Resolusi Width (Pixel)", [1366.0, 1920.0, 2240.0, 2560.0, 2880.0, 3200.0, 3840.0])
    res_height = st.selectbox("Resolusi Height (Pixel)", [768.0, 900.0, 1080.0, 1200.0, 1400.0, 1600.0, 1800.0, 2160.0])
    spec_rating = st.slider("Skor Rating Spesifikasi", min_value=60.0, max_value=89.0, value=69.38)

st.divider()

# --- 4. Proses Prediksi ---
if st.button("🔍 Hitung Estimasi Harga", use_container_width=True, type="primary"):
    try:
        # Mengubah input menjadi data frame sesuai urutan pelatihan di Notebook
        # URUTAN WAJIB SAMA DENGAN df_ml.drop(columns=["price"]) di notebook
        input_data = pd.DataFrame({
            'brand': [encoders['brand'].transform([brand])[0]],
            'spec_rating': [float(spec_rating)],
            'processor': [encoders['processor'].transform([processor])[0]],
            'CPU': [encoders['CPU'].transform([cpu])[0]],
            'Ram': [int(ram)],
            'Ram_type': [encoders['Ram_type'].transform([ram_type])[0]],
            'ROM': [int(rom)],
            'ROM_type': [encoders['ROM_type'].transform([rom_type])[0]],
            'GPU': [encoders['GPU'].transform([gpu])[0]],
            'display_size': [float(display_size)],
            'resolution_width': [float(res_width)],
            'resolution_height': [float(res_height)],
            'OS': [encoders['OS'].transform([os])[0]],
            'warranty': [int(warranty)]
        })

        # Prediksi
        prediction = model.predict(input_data)[0]

        # Tampilkan hasil
        st.success("### Estimasi Harga Laptop:")
        st.metric(label="Harga", value=f"Rp {prediction:,.2f}")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
