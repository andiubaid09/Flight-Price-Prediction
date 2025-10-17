# 🧠 Regression Models Repository

Repositori ini berisi kumpulan eksperimen dan implementasi berbagai **algoritma regresi** untuk memprediksi harga tiket pesawat dengan dataset serupa. Setiap model dibangun menggunakan **Scikit-learn Pipeline** (atau wrapper lain) untuk menjamin preprocessing konsisten dan mencegah data leakage.  

## 🚀 Tujuan
- Membandingkan performa berbagai model regresi.
- Mendokumentasikan preprocessing, hyperparameter tuning, dan evaluasi.
- Menyediakan model siap pakai untuk prediksi harga tiket pesawat.

---

## 🔧 Setup Lingkungan
Pastikan Anda sudah menginstal dependensi berikut:

```bash
pip install pandas numpy scikit-learn joblib
```

## 📊 Model yang Tersedia
1. 🌲 Random Forest Regressor
  - Terdiri dari banyak pohon keputusan *(Decision Tree)* yang dibangun secara acak dari dataset. Setiap pohon akan membuat prediksi sendiri, lalu hasil akhirnya diambil sebagai rata-rata dari semua pohon untuk regresi.
  - Pipeline dengan TransformedTargetRegressor (log-transform target).
  - Hyperparameter tuning via GridSearchCV.
  - Kinerja:
    - R²: 0.9537
    - MAE: 2808.1478
    - RMSE: 4882.4682

2. 📈 Linear Regression 
  - Model paling sederhana yang digunakan untuk memprediksi nilai kontinu (angka) berdasarkan hubungan linear antara input (fitur) dan output (target).
  - Pipeline dengan TransformedTargetRegressor (log-transform target).
  - Menerapkan logaritma pada inputan fitur numerik (X) sebelum distandarisasi.
  - Kinerja:
    - R²: 0.8489
    - MAE: 4912.04
    - RMSE: 8826.69

3. 🧱 Ridge Regression
  - Ridge adalah variasi dari Linear Regression yang menambahkan regularisasi L2 untuk menghindari overfitting dan menstabilkan model ketika fitur-fitur memiliki korelasi tinggi *(multikolinearitas)*. Tidak menghapus fitur, tapi mengecilkan nilai koefisien agar tidak ekstrem.
  - Pipeline dengan TransformedTargetRegressor (log-transform target)
  - Menerapkan logaritma pada inputan fitur numerik (X) sebelum distandarisasi. Artinya fitur numerik mengambil logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler().
  - Tujuan dari menerapkan logaritma pada inputan fitur numerik (X) adalah mengurangi efek *skewness* (kemencengan) data. Membuat outlier tidak terlalu berpengaruh besar dan membantu model linear belajar hubungan yang lebih proporsional.
  - Misal data days_left : [1, 2, 3, 5, 10, 20, 50, 100, 200]. Grafiknya miring ke kanan (kebanyakan kecil, sedikit besar). Kalau diambil lognya maka : [0.69, 1.10, 1.39, 1.79, 2.39, 3.04, 3.93, 4.61, 5.30]. Sekarang datanya lebih seimbang dan halus. Nilai besar diperkecil, tapi urutan tetap sama. Model jadi tidak "tertipu" oleh data ekstrem seperti 200 hari.
  - Kinerja :
    - R² :
    - MAE :
    - RMSE :
4. 🪢 Lasso Regression
5. ⚖️ ElasticNet 
6. Decision Tree Regressor

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