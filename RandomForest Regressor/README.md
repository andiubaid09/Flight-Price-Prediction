
# ✈️ Prediksi Harga Tiket Pesawat (Flight Price Prediction)

## 📝 Deskripsi Proyek
Proyek ini bertujuan untuk membangun model *Machine Learning* yang kuat dan akurat untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan.  

Model dikembangkan menggunakan arsitektur **Scikit-learn Pipeline** untuk memastikan konsistensi *preprocessing* dan mencegah *data leakage*.  
Model utama yang digunakan adalah **Random Forest Regressor** yang dibungkus dengan **TransformedTargetRegressor** (TTR) agar target harga dapat ditransformasi secara logaritmik sehingga prediksi lebih stabil.

---

## 🚀 Fitur Utama & Metodologi

### 1. Arsitektur Pipeline
- Semua tahap preprocessing data dan modeling dikapsulasi ke dalam `Pipeline`.
- Menjamin data latih dan data baru diproses **identik**.

### 2. Transformasi Target Otomatis
- Target variabel (`price`) ditransformasi dengan fungsi logaritmik `np.log1p` sebelum pelatihan.
- Hasil prediksi dikembalikan ke skala asli dengan `np.expm1`.
- Membantu mengurangi efek distribusi harga yang *skewed* (tidak normal).

### 3. Pemrosesan Fitur (Preprocessing)

| Tipe Fitur              | Contoh Fitur        | Teknik Transformasi    |
|--------------------------|---------------------|-------------------------|
| **Kategorikal Ordinal** | `class` (Economy, Business) | `OrdinalEncoder` |
| **Kategorikal Nominal** | `source_city`, `departure_time`, `destination_city` | `OneHotEncoder` |
| **Numerik**             | `days_left`, `duration` | `StandardScaler` |

### 4. Model & Optimasi
- **Model**: Random Forest Regressor
- **Optimasi**: `GridSearchCV` dengan parameter:
  - `n_estimators` (jumlah pohon)
  - `max_depth` (kedalaman pohon)
  - `min_samples_split` (jumlah minimum sampel untuk split)

---

## 📈 Hasil Kinerja (Pada Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.9422 | Model menjelaskan lebih dari 94% variasi harga |
| **MAE (Mean Absolute Error)** | 3043.62 | Rata-rata prediksi meleset sekitar Rp 3.043 |
| **RMSE (Root Mean Squared Error)** | 5452.52 | Kesalahan rata-rata, sensitif terhadap outlier |

---

## 🛠️ Panduan Penggunaan

### 1. Prasyarat
Pastikan pustaka berikut sudah terpasang:

```bash
pip install pandas numpy scikit-learn joblib
