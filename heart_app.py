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
st.image("199095433.jpg", width=200, use_container_width=True)
st.markdown("<h1 style='text-align:center;'>Prediksi Penyakit Jantung ❤️</h1>", unsafe_allow_html=True)
st.write("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

# === Sidebar input ===
st.sidebar.header("Input Data Pasien")
age = st.sidebar.slider("Usia (age)", 20, 100, 50)
sex_label = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
sex = 1 if sex_label == "Laki-laki" else 0
cp = st.sidebar.slider("Tipe nyeri dada (cp)", 0, 3, 1)
trestbps = st.sidebar.number_input("Tekanan darah istirahat (trestbps)", 80, 200, 120)
chol = st.sidebar.number_input("Kolesterol (chol)", 80, 600, 200)
fbs = st.sidebar.selectbox("Gula darah puasa > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.slider("Hasil EKG saat istirahat (restecg)", 0, 2, 1)
thalach = st.sidebar.slider("Detak jantung maksimal (thalach)", 70, 210, 150)
exang = st.sidebar.selectbox("Angina akibat olahraga (exang)", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1)
slope = st.sidebar.slider("Slope", 0, 2, 1)
ca = st.sidebar.slider("ca (Jumlah pembuluh besar)", 0, 4, 0)
thal = st.sidebar.slider("thal (Kondisi thalium scan)", 0, 3, 2)

# === Buat array input ===
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# === Prediksi ===
prediction = model.predict(input_scaled)

# === Output hasil prediksi ===
st.subheader("Hasil Prediksi:")
if prediction == 1:
    st.success("Pasien berisiko mengidap penyakit jantung 💓")
else:
    st.success("Pasien tidak berisiko penyakit jantung 💚")

# === Footer ===
st.caption("Model: Decision Tree Classifier | Akurasi: 81.97%")
