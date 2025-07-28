import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    url = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
    data = pd.read_csv(url).dropna()
    return data

data = load_data()
X = data.drop('target', axis=1)
y = data['target']

# PCA for training pipeline only
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=30)
model.fit(X_train, y_train)

st.title("Prediksi Penyakit Jantung 🫀")
st.write("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

ca = st.slider("ca (Jumlah pembuluh besar)", 0.0, 4.0, 1.0)
thal = st.slider("thal (Kondisi thalium scan)", 0.0, 3.0, 2.0)
trestbps = st.number_input("Tekanan darah istirahat (trestbps)", 80.0, 200.0, 120.0)
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Kemiringan ST (slope)", [0.0, 1.0, 2.0])
restecg = st.selectbox("Hasil EKG istirahat (restecg)", [0.0, 1.0, 2.0])
exang = st.radio("Angina Induced oleh olahraga (exang)", [0.0, 1.0])
chol = st.number_input("Kolesterol (chol)", 100.0, 600.0, 250.0)
fbs = st.radio("Gula darah > 120 mg/dl (fbs)", [0.0, 1.0])

if st.button("Prediksi Sekarang"):
    input_data = np.array([[ca, thal, trestbps, oldpeak, slope, restecg, exang, chol, fbs]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    if pred[0] == 1:
        st.error("⚠️ Hasil Prediksi: Pasien Berisiko Penyakit Jantung")
    else:
        st.success("✅ Hasil Prediksi: Tidak Berisiko Penyakit Jantung")