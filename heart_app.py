import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
dataset_url = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
df = pd.read_csv(dataset_url)
df.dropna(inplace=True)

# Fitur dan target
X = df.drop(columns='target')
y = df['target']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier(criterion='gini', max_depth=30)
model.fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

# Header dengan logo dan judul
st.image("199095433.jpg", width=300, use_column_width=False)
with col2:
    st.markdown("<h1 style='margin-bottom: 5px;'>Prediksi Penyakit Jantung 🫀</h1>", unsafe_allow_html=True)
    st.write("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

# Input data pasien
st.subheader("Input Data Pasien")
ca = st.sidebar("ca (Jumlah pembuluh besar)", 0.0, 4.0, 1.0)
thal = st.sidebar("thal (Kondisi thalium scan)", 0.0, 3.0, 2.0)
trestbps = st.number_input("Tekanan darah istirahat (trestbps)", 80.0, 200.0, 120.0)
oldpeak = st.sidebar("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.sidebar("Slope (Kemiringan ST)", 0.0, 2.0, 1.0)
restecg = st.sidebar("restecg (Hasil ECG istirahat)", 0.0, 2.0, 1.0)
exang = st.sidebar("exang (Angina saat olahraga)", 0.0, 1.0, 1.0)
chol = st.sidebar("Kolesterol (chol)", 100.0, 600.0, 200.0)
fbs = st.sidebar("fbs (Gula darah puasa > 120mg/dl)", 0.0, 1.0, 1.0)

input_data = np.array([[ca, thal, trestbps, oldpeak, slope, restecg, exang, chol, fbs]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Hasil prediksi
st.subheader("Hasil Prediksi")
if prediction[0] == 1:
    st.error("⚠️ Pasien berisiko terkena penyakit jantung.")
else:
    st.success("✅ Pasien tidak berisiko terkena penyakit jantung.")

# Gambar ilustrasi tambahan
st.image("199095433.jpg", caption="Ilustrasi gejala serangan jantung", use_column_width=True)
