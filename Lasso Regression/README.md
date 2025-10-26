# 🪢 Prediksi Harga Tiket Pesawat dengan Lasso Regression

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan pada datasheet. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **Lasso Regressor** sebagai model utama.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.


### 📖 Penjelasan Tentang Ridge Regression
Lasso Regression *(Least Absolute Shrinkage and Selection Operator)* adalah jenis regresi linear yang menggunakan reguralisasi L1, dengan tujuan utamannya adalah mencegah overfitting dan memilih fitur *(feature selection)* dengan menghilangkan variabel yang tidak penting (koefisien jadi nol). Karakteristik dari Lasso ini adalah L1 Regularization yang menggunakan penalti absolute coef yaitu menekankan koefisien mendekati 0. Kemudian, Automatic feature selection yang bisa membuat beberapa koefisien = 0 dimana fitur tidak dianggap oleh Lasso. Mencegah overfitting karena model menjadi sederhana. Lasso ini bisa menjadi keras. jika alpha terlalu besar sehingga semua koefisien bisa jadi nol *(underfitting)*.
Lasso akan cocok jika banyak fitur, sehingga membuat cari fitur penting saja, berbeda dengan Ridge dimana fitur multikolinear dan semua fitur harus relevan. Berikut adalah kelebihan dari Lasso Regression:
1. Bisa pilih fitur penting otomatis
2. Mengurangi overfitting
3. Interpretasi mudah

Serta berikut adalah kekurangan dari Lasso Regression:
1. Bisa menghilangkan fitur yang sebenarnya relevan jika alpha terlalu besar
2. Tidak bekerja dengan baik jika semua fitur penting
3. Tidak stabil jika fitur sangat saling berkorelasi.

Kapan Lasso digunakan? Saat jumlah fitux (x) banyak, saat ingin tahu fitur mana yang paling penting, saat ada hubungan multikolinear (fitur saling berkorelasi) dan saat ingin mencegah overfitting, tapi tetap sederhana.

**Contoh kasus Sederhana Menggunakan Lasso**, misal kita ingin memprediksi harga rumah berdasarkan fitur (x) seperti luas tanah, jumlah kamar, tahun bangun, warna cat rumah (tidak relevan). Jika menggunakan Lasso Regression fitur tidak penting seperti cat akan menjadikan koefisien =0, dihilangkan dari model. Hasilnya, model lebih sederhana dan lebih fokus pada fitur relevan. Singkatnya Lasso = Linear Regression (base model) + Regularisasi L1 dan mampu membuat beberapa koefisien = 0 serta berguna untuk *feature selection*.

---

## 🚀 Fitur Utama

### 1. Arsitektur Pipeline
- Seluruh preprocessing (encoding, scaling, ordinal) dan model dibungkus dalam `Pipeline`.
- Menjamin data latih dan data baru diproses identik → mencegah *data leakage*.

### 2. Transformasi Target
- Target `price` ditransformasi dengan `np.log1p` saat training.
- Hasil prediksi dikembalikan ke skala asli dengan `np.expm1`.
- Membantu model menghadapi distribusi harga yang sangat bervariasi.

### 3. Preprocessing Fitur

| Jenis Fitur              | Contoh               | Teknik Transformasi  |
|---------------------------|----------------------|----------------------|
| **Kategorikal Ordinal**  | `class` (Economy/Business) | `OrdinalEncoder` |
| **Kategorikal Nominal**  | `source_city`, `departure_time`, `destination_city`,`stops`, `arrival_time` | `OneHotEncoder` |
| **Numerik**              | `days_left` | `StandardScaler` |

### 4. Optimasi Hyperparameter
Dilakukan dengan **GridSearchCV** pada parameter utama Random Forest:
- `Alpha (λ) ` → Sebuah kekuatan reguralisasi pada Lasso. Berapa kuat Lasso mendorong koefisien ke nol.
- Hyperparameter menggunakan **GridSearchCV** ditemukan nilai Lasso adalah 0.01.

Alpha mengatur kekuatan regularisasi (hukuman terhadap besar kecilnya koefisien model). Nilai alpha jika 0 itu artinya sangat kecil atau sama saja dengan linear regression (base model). Berikut efek nilai aplha saat melakukan hyperparameter tuning pada model.
|Nilai alpha        | Efek pada model                                 |
|-------------------|-------------------------------------------------|
|Sangat kecil (0)   |Mirip Linear Regression (tanpa reguralisasi), resiko overfitting|
|Sedang (0.001, 0.1)|Reguralisasi cukup, model seimbang (bias vs variance optimal)|
|Besar (10,100)     |Koefisien ditekan mendekati 0, banyak fitur dihapus, resiko underfitting|

Ketika **GridSearchCV** menemukan alpha = 0.01 sebagai yang terbaik, artinya reguralisasi ringan cukup untuk mengurangi overfitting, tidak terlalu besar sehingga model masih bisa belajar dengan baik. Model ini tidak butuh regularisasi yang terlalu kuat. Fitur-fitur yang digunakan cukup relevan karena penalti ringan (0.01) sudah mampu menurunkan error tanpa membuang banyak fitur. Berarti model ini tidak terlalu kompleks, tidak overfitting, dan tidak butuh penyederhanaan ekstrem.

Lasso() masih punya beberapa parameter lain yang opsional untuk di tuning:
|Parameter          | Fungsi                  | Umumnya di tuning         |
|-------------------|-------------------------|---------------------------|
|alpha              | Kekuatan reguralisasi L1, semakin tinggi akan semakin banyak koefisien yang ditekan ke nol| Sangat penting untuk di tuning|
|fit_intercept      | Apakah model menghitung intercept(bias)|Opsional|
|max_iter           |Jumlah maksimal iterasi untuk optimasi|Penting jika model tidak konvergen|
|tol                |Toleransi untuk berhenti iterasi(semakin kecil akan lebih akurat tapi lebih lama)|Kadang diperlukan|
|selection          |Cara update koefisien:`cyclic` atau `random`|Biasanya tidak diubah|
|warm_start         | Gunakan hasil sebelumnya sebagai starting point|Jarang dituning|
|positive           | Jika True, semua koefisien harus positif| Hanya jika data memang mengharuskan|

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.8569 | Model menjelaskan 85,69% variabilitas dari data target |
| **MAE** | 4947.23 | Rata-rata selisih absolut antara nilai prediksi dan nilai aktual = 4947.23 |
| **RMSE** | 8587.39 | Beberapa titik data jauh dari prediksi model yaitu 8587.39 |

**R2 *(coefficient of determination)*** mengukur proporsi varians target yang dapat dijelaskan oleh model.
**MAE *(Mean Absolute Error)*** rata-rata nilai absolut selisih prediksi dan aktual.
**RMSE *(Root Mean Squared Error)*** Akar dari rata-rata kuadrat error. Lebih sensitif terhadap kesalahan besar (outlier).

**Interpretasi Angka**
- R2 = 0.8569 menunjukkan seberapa bear variasi data aktual yang bisa dijelaskan oleh model. Artinya model mampu menjelaskan sekitar 85.69% pola hubungan antara fitur (X) dan target (y). Sisanya 14.31% variasi tidak berhasil dijelaskan oleh model (mungkin disebabkan oleh noise, variabel yang belum digunakan, atau ketidaktepatan model). Model cukup kuat karena >80% pola data dapat diprediksi.
- MAE = 4947.23 menunjukkan rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. Berarti rata-rata model melakukan kesalahan prediksi sebesar + 4.947 satuan. Kesalahan rata-rata masih cukup moderat, tapi belum sangat kecil.
- RMSE = 8587.39 menunjukkan akar dari rata-rata selisih kuadrat, karena dikuadratkan kesalahan besar (outlier) lebih besar diperhatikan/dihukum lebih keras. Nilai RMSE menunjukkan ada beberapa data yang kesalahannya sangat besar. Saat model salah, terkadang salahnya bisa sampai +8.587 atau lebih ini menandakan adanya outlier/error tinggi di sebagian point.

Interpretasi angka ini sedikit lebih baik dari Ridge, namun perbandingan dengan RandomForest masih kalah jauh, terutama di error dan R2.

---

## 📊 Visualisasi Data
### 1. Prediksi vs Nilai Aktual
![Prediksi vs Harga Nilai Aktual](Assets/Harga%20Aktual%20vs%20Harga%20Prediksi.png)<br>
Visualisasi di atas merupakan Scatter Plot Prediksi vs Nilai Aktual, visualisasi membantu untuk mengerti perilaku model secara intuitif hal yang sering tidak terlihat hanya dari angka metrik. Plot ini menunjukkan seberapa dekat hasil prediksi dengan kenyataan. Semacam uji keakuratan visual, melengkapi metrik Numerik(MAE, RMSE, R2). Jika R2 mendekati 1 dan titik berjejer di sekitar garis merah -> Model bagus.
Interpretasi hasil:
- Kalau titik-titik biru banyak yang menempel di garis merah, berarti model prediksi sangat akurat.
- Titik yang jauh dari garis merah = error prediksi yang lebih besar
- Semakin rapat titik ke garis merah -> semakin tinggi nilai R2 (koefisien determinasi)
- Menambahkan nilai R2 di plot, jadi pembaca bisa langsung lihat seberapa baik model menjelaskan variasi data.

### 2. Distribusi Residual Error
![Distribusi Residual Error](Assets/Distribusi%20Residual%20Error.png)<br>
Residual adalah selisih antara nilai aktual dan nilai prediksi. Artinya, seberapa jauh prediksi model dari nilai sebenarnya. Tujuannya adalah mengecek apakah error model terdistribusi secara normal (simetris, tanpa bias besar).
Elemen visualisasi:
1. Histogram (batang warna coral/oranye)
 - Menunjukkan sebaran nilai residual.
 - Semakin tinggi batang -> semakin banyak residual yang nilainya berada pada rentang tersebut.
 - Kalau batang lebih banyak di sekitar 0 -> berarti error kecil dan model cukup baik.
2. Kurva KDE (garis halus melengkung)
 - Fungsinya membantu melihat bentuk distribusi residual. Mirip dari histogram
 - Kalau kurvanya simetris di sekitar 0 -> model tidak bias
 - Kalau melenceng ke kiri/kanan -> model bias (overestimate atau underestimate)
3. Garis Tegak Lurus Merah Putus-Putus(axvline)
 - Titik acuan di 0 (residual = 0)
 - Kalau semua residual jatuh tepat di garis ini, model prediksi = nilai aktual(perfect).
 - Tapi realitannya residual pasti menyebar di sekitar garis ini
 - Garis merah ini dipakai untuk menilai: apakah distribusi residual condong ke kiri/kanan dan seberapa jauh penyebarannya dari nol.

Model yang baik biasanya menghasilkan residual:
- Tersebar di sekitar nol,
- Tidak membentuk pola tertentu,
- Tidak terlalu menyebar jauh
Jadi, dengan grafik ini, kita bisa menilai model sudah cukup baik atau masih perlu perbaikan (dengan feature engineering atau tuning parameter)

Interpretasi hasil:
- Distribusi simetris & berpusat di 0 -> model cenderung tidak bias, prediksi mendekati nilai aktual.
- Distribusi condong ke kiri atau kanan -> ada bias, misalnya model sering overestimate (Residual Negatif atau prediksi terlalu tinggi) atau underestimate (Residual Positif atau prediksi terlalu rendah).
- Distribusi menyebar lebar -> Error prediksi besar, model kurang akurat.
- Distribusi sempit di sekitar 0 -> Prediksi model sangat dekat dengan nilai aktual, model bagus

### 3. Residual vs Nilai Prediksi
![Residual vs Nilai Prediksi](Assets/Residual%20vs%20Nilai%20Prediksi.png)<br>
Visualisasi ini menampilkan Residual Plot (Residual vs Predicted Values). Tujuannya untuk mengecek apakah error model terdistribusi secara acak atau ada pola tertentu. Membantu mendeteksi, bias sistematis (model selalu meleset ke satu arah), Heteroskedatisitas (variasi error meningkat pada nilai prediksi besar) dan Nonlinearitas (model tidak cukup fleksibel untuk pola data). Ini adalah visualisasi evaluasi model regresi yang digunakan untuk menilai kualitas prediksi dengan melihat pola error(residual) terhadap nilai prediksi.

Elemen visualisasi:
1. Scatterplot titik hijau (prediksi vs residual)
 - Sumbu X = harga prediksi (y_pred) -> Nilai yang diperkirakan model.
 - Sumbu Y = residual (y_test - y_pred) -> selisih antara nilai aktual dan prediksi
 - Setiap titik = satu data.
2. Garis horizontal merah putus-putus (residual = 0)
 - Menjadi titik acuan:
  - Kalau residual = 0 -> prediksi tepat sama dengan aktual
  - Kalau titik di atas garis -> model underestimate (prediksi terlalu rendah).
  - Kalau titik di bawah garis -> model overestimate (prediksi terlalu tinggi).

Penyebaran acak di sekitar garis 0 -> artinya error tidak bergantung pada nilai prediksi berarti model cukup baik. Pola tertentu (misalnya residual makin besar saat harga makin tinggi)-> ada masalah:
 - Heteroskedastisitas : error semakin besar saat nilai meningkat -> model tidak stabil di data besar
 - Nonlinearitas: Model tidak cukup fleksibel menangkap pola -> hubungan non linear.
 - Bias Sistematis: Model cenderung selalu overestimate atau underestimate.

Interpretasi hasil:
 - Kalau titik-titik terlihat acak, menyebar merata di sekitar 0 -> model sudah oke
 - Kalau titik cenderung membentuk pola busur/ kurva -> mungkin model perlu metode lain (misalnya boosting atau menambah fitur)
 - Kalau sebaran makin melebar di kanan (harga tinggi) -> model kesulitan memprediksi harga 
 
### 4. Fitur Penting (Gini Importance)
![Gini Importance](Assets/Feature%20Importances.png)<br>
Visualisasi ini adalah Feature Importance Plot dari Lasso Regressor, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

## 🛠️ Cara Menggunakan

### 1. Prasyarat
Install pustaka berikut:
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Muat & Gunakan Model
```bash
import pandas as pd
import joblib

# Muat model
best_model = joblib.load("Lasso_regressor_predictions_flights_price.pkl")

# Data baru
data_baru = pd.DataFrame({
    "source_city": ["Mumbai"],
    "departure_time": ["Aftenoon"],
    "stops": [zero],
    "arrival_time": ["late_night"],
    "destination_city": ["Delhi"],
    "class": ["Business"],
    "days_left": [7],
    "duration": [2.5]
})

# Prediksi harga
prediksi = best_model.predict(data_baru)[0]
print(f"Prediksi Harga Tiket:  {prediksi:,.2f}")
```

## 🔮 Potensi Pengembangan
- Coba alpha lebih kecil lagi (0.005, 0.001) untuk melihat apakah bisa lebih baik dari 0.01
- Gunakan LassoCV() otomatis untuk mencari alpha dari banyak nilai tanpa manual grid
- Bandingkan *coefficients* untuk lihat fitur mana yang jadi 0 (tidak dipakai)
