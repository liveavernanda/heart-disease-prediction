import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar Input
st.sidebar.title("Input Data Pasien")
age = st.sidebar.slider("Usia (age)", 20, 100, 50)
sex = st.sidebar.selectbox("Jenis Kelamin (sex)", [0, 1])
cp = st.sidebar.slider("Tipe nyeri dada (cp)", 0, 3, 1)
trestbps = st.sidebar.number_input("Tekanan darah istirahat (trestbps)", 80, 200, 120)
chol = st.sidebar.number_input("Kolesterol (chol)", 100, 600, 200)
fbs = st.sidebar.selectbox("Gula darah puasa > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.selectbox("Hasil EKG (restecg)", [0, 1, 2])
thalach = st.sidebar.slider("Detak jantung maksimal (thalach)", 70, 210, 150)
exang = st.sidebar.selectbox("Angina Induced by Exercise (exang)", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.sidebar.slider("Slope", 0, 2, 1)
ca = st.sidebar.slider("ca (Jumlah pembuluh besar)", 0, 4, 0)
thal = st.sidebar.slider("thal (Kondisi thalium scan)", 0, 3, 2)

# Masukkan gambar besar
st.image("199095433.jpg", use_container_width=True)  # Gunakan gambar yang sudah kamu upload

# Judul
st.markdown("<h1 style='text-align: center;'>❤️ Prediksi Penyakit Jantung</h1>", unsafe_allow_html=True)
st.markdown("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

# Prediksi
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
if prediction == 1:
    st.success("Pasien berisiko mengidap penyakit jantung 💓")
else:
    st.success("Pasien tidak berisiko penyakit jantung 💚")

# Akurasi Model
st.caption("Model: Decision Tree Classifier | Akurasi: 81.97%")
