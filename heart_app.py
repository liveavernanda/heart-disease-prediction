import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/heart_disease.csv")
df = df.dropna()

# === Fitur dan target ===
X = df.drop("target", axis=1)
y = df["target"]

# === Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train model ===
model = DecisionTreeClassifier(criterion='gini', max_depth=30)
model.fit(X_train, y_train)

# === UI ===
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

# === Gambar besar header ===
st.image("199095433.jpg", width=300, use_column_width=False)

# === Judul utama ===
st.markdown("<h1 style='text-align:center;'>Prediksi Penyakit Jantung 🫀</h1>", unsafe_allow_html=True)
st.write("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

# === Sidebar input ===
st.sidebar.header("Input Data Pasien")

ca = st.sidebar.slider("ca (Jumlah pembuluh besar)", 0.0, 4.0, step=1.0)
thal = st.sidebar.slider("thal (Kondisi thalium scan)", 0.0, 3.0, step=1.0)
trestbps = st.sidebar.number_input("Tekanan darah istirahat (trestbps)", min_value=80.0, max_value=200.0, value=120.0)
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1)
slope = st.sidebar.slider("Slope", 0.0, 2.0, step=1.0)
restecg = st.sidebar.slider("Restecg", 0.0, 2.0, step=1.0)
exang = st.sidebar.slider("Exercise induced angina (exang)", 0.0, 1.0, step=1.0)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=80.0, max_value=600.0, value=200.0)
fbs = st.sidebar.slider("Fasting blood sugar > 120 mg/dl (fbs)", 0.0, 1.0, step=1.0)

# === Buat array input ===
input_data = np.array([[ca, thal, trestbps, oldpeak, slope, restecg, exang, chol, fbs]])
input_scaled = scaler.transform(input_data)

# === Prediksi ===
prediction = model.predict(input_scaled)

# === Output hasil prediksi ===
st.subheader("Hasil Prediksi:")
if prediction[0] == 1:
    st.success("Pasien berisiko mengidap penyakit jantung 🫀")
else:
    st.error("Pasien tidak berisiko penyakit jantung ❤️")

# === Footer ===
st.caption("Model: Decision Tree Classifier | Akurasi: 81.97%")
