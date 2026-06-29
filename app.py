import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Prediksi Harga Laptop", 
    page_icon="💻", 
    layout="wide"
)

# 2. Load Model & Encoders (menggunakan cache agar web tidak lemot)
@st.cache_resource
def load_assets():
    model = joblib.load("laptop_price_prediction_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file model (.pkl). Pastikan kedua file .pkl berada di folder yang sama! Error: {e}")
    st.stop()

st.title("💻 Laptop Price Prediction App")
st.markdown("Aplikasi estimasi harga laptop berbasis **Gradient Boosting Regressor**.")
st.divider()

# 3. Form Input User (Dibagi menjadi 3 kolom agar rapi)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏢 Identitas Utama")
    brand = st.selectbox("Brand Laptop", encoders['brand'].classes_)
    processor = st.selectbox("Tipe Processor", encoders['processor'].classes_)
    cpu = st.selectbox("Konfigurasi CPU", encoders['CPU'].classes_)
    gpu = st.selectbox("Kartu Grafis (GPU)", encoders['GPU'].classes_)
    os = st.selectbox("Sistem Operasi (OS)", encoders['OS'].classes_)

with col2:
    st.subheader("💾 Kapasitas & Memori")
    ram = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64], index=1)
    ram_type = st.selectbox("Tipe RAM", encoders['Ram_type'].classes_)
    rom = st.selectbox("Kapasitas Penyimpanan / ROM (GB)", [64, 128, 256, 512, 1024, 2048], index=3)
    rom_type = st.selectbox("Tipe Penyimpanan", encoders['ROM_type'].classes_)
    warranty = st.selectbox("Garansi (Tahun)", [0, 1, 2, 3], index=1)

with col3:
    st.subheader("🖥️ Layar & Performa")
    display_size = st.number_input("Ukuran Layar (Inch)", min_value=11.0, max_value=18.0, value=15.6, step=0.1)
    res_width = st.selectbox("Resolusi Width (Pixel)", [1366.0, 1920.0, 2240.0, 2560.0, 2880.0, 3200.0, 3840.0], index=1)
    res_height = st.selectbox("Resolusi Height (Pixel)", [768.0, 900.0, 1080.0, 1200.0, 1400.0, 1600.0, 1800.0, 2160.0], index=2)
    spec_rating = st.slider("Skor Rating Spesifikasi", min_value=60.0, max_value=89.0, value=69.38)

st.divider()

# 4. Tombol Prediksi
if st.button("🔍 Hitung Prediksi Harga", use_container_width=True, type="primary"):
    try:
        # PENTING: Urutan 14 kolom di bawah ini SUDAH DIKUNCI sama persis dengan X_train di Notebook!
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

        # Jalankan Prediksi
        prediction = model.predict(input_data)[0]

        # Tampilkan Output
        st.balloons()
        st.success("### Hasil Estimasi Harga Laptop:")
        st.metric(label="Rekomendasi Harga Jual/Beli", value=f"Rp {prediction:,.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan kalkulasi: {e}")
