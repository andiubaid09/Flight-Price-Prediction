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

## 