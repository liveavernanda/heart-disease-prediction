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
col1, col2, col3 = st.columns([1, 2, 1])  # layout 3 kolom, kolom tengah lebih lebar
with col2:
    st.image("199095433.jpg", width=400)
st.markdown("<h1 style='text-align:center;'>Prediksi Penyakit Jantung ❤️</h1>", unsafe_allow_html=True)
st.write("Masukkan data pasien untuk mengetahui apakah berisiko penyakit jantung.")

# === Sidebar input ===
st.sidebar.header("Input Data Pasien")

age = st.sidebar.slider("Usia (age)", 20, 100, 50, help="Usia pasien dalam tahun")

sex_label = st.sidebar.selectbox("Jenis Kelamin", ["1", "0"], help="0 = Perempuan, 1 = Laki-laki")
sex = 1 if sex_label == "Laki-laki" else 0

cp = st.sidebar.slider("Tipe nyeri dada (cp)", 0, 3, 1, help="""
0 = Typical angina  
1 = Atypical angina  
2 = Non-anginal pain  
3 = Asymptomatic
""")

trestbps = st.sidebar.number_input("Tekanan darah istirahat (trestbps)", 80, 200, 120, help="Tekanan darah saat istirahat dalam mmHg")

chol = st.sidebar.number_input("Kolesterol (chol)", 100, 600, 200, help="Jumlah kolesterol serum dalam mg/dl")

fbs = st.sidebar.selectbox("Gula darah puasa > 120 mg/dl (fbs)", [0, 1], help="1 jika gula darah puasa > 120 mg/dl, jika tidak maka 0")

restecg = st.sidebar.selectbox("Hasil EKG saat istirahat (restecg)", [0, 1, 2], help="""
0 = Normal  
1 = Kelainan gelombang ST-T  
2 = Hipertrofi ventrikel kiri
""")

thalach = st.sidebar.slider("Detak jantung maksimal (thalach)", 70, 210, 150, help="Detak jantung maksimal yang dicapai saat latihan")

exang = st.sidebar.selectbox("Angina akibat olahraga (exang)", [0, 1], help="1 = ya, 0 = tidak")

oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1, help="Depresi segmen ST akibat latihan dibandingkan dengan saat istirahat")

slope = st.sidebar.selectbox("Kemiringan segmen ST (slope)", [0, 1, 2], help="""
0 = Turun  
1 = Datar  
2 = Naik
""")

ca = st.sidebar.slider("Jumlah pembuluh utama (ca)", 0, 4, 0, help="Jumlah pembuluh darah besar yang terlihat dari fluoroskopi")

thal = st.sidebar.selectbox("Hasil thalassemia test (thal)", [0, 1, 2, 3], help="""
0 = Tidak diketahui  
1 = Normal  
2 = Fixed defect  
3 = Reversable defect
""")

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
