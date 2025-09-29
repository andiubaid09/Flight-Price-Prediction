# 🧠 Regression Models Repository

Repositori ini berisi kumpulan eksperimen dan implementasi berbagai **algoritma regresi** untuk memprediksi harga tiket pesawat dan dataset serupa. Setiap model dibangun menggunakan **Scikit-learn Pipeline** (atau wrapper lain) untuk menjamin preprocessing konsisten dan mencegah data leakage.  

## 🚀 Tujuan
- Membandingkan performa berbagai model regresi.
- Mendokumentasikan preprocessing, hyperparameter tuning, dan evaluasi.
- Menyediakan model siap pakai untuk prediksi harga tiket pesawat.

---

## 📂 Struktur Proyek
├── models/
│ ├── random_forest/
│ │ ├── flight_rf_notebook.ipynb # Notebook training & evaluasi
│ │ ├── best_flight_rf.pkl # Model terlatih (via Git LFS / Release)
│ │ └── README.md # Penjelasan khusus Random Forest
│ ├── linear_regression/
│ ├── xgboost/
│ └── ...
├── data/ # Dataset mentah / sample
├── notebooks/ # Eksperimen umum
└── README.md # Dokumentasi utama

---

## 🔧 Setup Lingkungan
Pastikan Anda sudah menginstal dependensi berikut:

```bash
pip install pandas numpy scikit-learn joblib
```

## 📊 Model yang Tersedia
1. 🌲 Random Forest Regressor
  - Pipeline dengan TransformedTargetRegressor (log-transform target).
  - Hyperparameter tuning via GridSearchCV.
  - Kinerja:
    - R²: 0.9537
    - MAE: 2808.1478
    - RMSE: 4882.4682

2. 📈 Linear Regression (coming soon)

## 🛠️ Cara Menggunakan Model
Contoh untuk Random Forest:
```bash
import pandas as pd
import joblib

# Load Model 
model = joblib.load("RandomForest Regressor/Model/rfr-flight_price_prediction.pkl")

# Data Baru
new_data = pd.DataFrame({
    'source_city' : ['Delhi'],
    'departure_time' : ['Evening'],
    'stops' : ['zero'],
    'arrival_time' : ['Night'],
    'destination_city' : ['Mumbai'],
    'class' : ['Economoy'],
    'days_left' : 1 
})

# Prediksi
prediksi = model.predict(new_data)[0]
print(f"Prediksi Harga Tiket : {prediksi:,.2f}")
```

## 🧑‍💻 Kontributor
- Muhammad Andi Ubaidillah