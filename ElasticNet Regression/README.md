# ⚖️ Prediksi Harga Tiket Pesawat dengan ElasticNet Regression

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **ElasticNet Regressor** sebagai model utama.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.

---

## 📖 Penjelasan Tentang ElasticNet Regression
ElasticNet adalah model regresi yang menggabungkan kelebihan dari Ridge Regression (L2 Regularization) dimana mengecilkan koefisien untuk menghindari overfitting dan Lasso Regression (L1 Regularization) yang bisa menghapus (memaksa 0) koefisien fitur yang tidak penting. Jadi ElasticNet = Lasso + Ridge dalam satu model dengan tujuan dari model ElasticNet adalah mencari koefisien terbaik dengan meminimalkan. ElasticNet adalah hybrid dari Ridge dan Lasso yang bisa mengurangi overfitting sekaligus melakukan seleksi fitur dengan lebih stabil. Berikut adalah kelebihan dan kekurangan dari ElasticNet:
|Kelebihan                                 |Kekurangan                                    |
|------------------------------------------|----------------------------------------------|
|Mengatasi multicollinearity               |Perlu tuning 2 parameter (alpha & l1_ratio)   |
|Bisa melakukan seleksi fitur (koefisien 0)|Lebih kompleks dibandingkan Ridge/Lasso saja  |
|Lebih stabil dibanding Lasso saat fitur berkorelasi| Interpretasi lebih sulit jika fitur sangat banyak|
|Cegah overfitting seperti Ridge           |Perlu scaling fitur agar hasil optimal        |

Kapan ElasticNet digunakan? GUnakan ElastiNet jika:
1. Banyak fitur yang berkorelasi satu sama lain
2. Jika ingin kombinasi kelebihan Ridge & Lasso
3. Jika ingin mencegah overfitting namun tetap bisa seleksi fitur

Kesimpulannya adalah ElasticNet Regression adalah model linear yang menggunakan reguralisasi kombinasi L1 (Lasso) dan L2 (Ridge). Model ini efektif digunakan ketika dataset memiliki banyak fitur, terdapat multicollinearity dan tujuan model bukan hanya prediksi tetapi juga seleksi fitur yang penting.

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
Dilakukan dengan **GridSearchCV** pada parameter utama ElasticNet:
- `alpha` → kontrol kekuatan penalti
- `l1_ratio` → mengatur kombinasi antara Lasso dan Ridge
- Hyperparameter menggunakan **GridSearchCV** ditemukan nilai alpha= 0.01 dan serta l1_ratio= 0.5

**Interpretasi Angka dari Hyperparameter**
Alpha adalah parameter yang menentukan seberapa kuat regularisasi diterapkan dalam model, nilai 0.01 termasuk sangat kecil, berarti regularisasi yang diterapkan ringan. Model tetap mempertahankan fleksibilitas dan tidak terlalu "tekan". Artinya, fitur-fitur yang ada masih dianggap penting oleh model dan tidak terlalu banyak dikecilkan atau dihapus. Kalau nilai alpha besar (misalnya 1 atau 10), model akan lebih memaksa koefisien mendekati nol dan bisa mengapus fitur-tapi tidak disini tidak. Model ini dengan nilai alpha 0.01 artinya tidak memerlukan penalti yang besar, artinya data cukup stabil dan tidak terlalu overfitting saat tanpa regularisasi berat.

l1_ratio mengontrol kombinasi antara L1 (Lasso) dan L2 (Ridge). l1_ratio = 0 -> murni Ridge (L2). Berarti nilai 0.5 adalah kombinasi L1 & L2 seimbang. Ada upaya mengurangi nilai koefisien (Ridge) agar model lebih stabil. Ada juga upaya menghapus fitur yang tidak penting (Lasso), tapi tidak terlalu agresif. Jadi model ini tidak terlalu keras seperti Lasso murni, tapi juga tidak hanya mengecilkan koefisien seperti Ridge murni. Model ini memilih jalan tengah, sebagian fitur mungkin dikecilkan , sebagian bisa jadi dihapus jika tidak penting, tapi tetap menjaga stabilitas seperti Ridge. Berikut tabel interpretasi kombinasi Alpha & l1_ratio untuk dataset ini.
|Parameter          | Makna                     | Implikasi                       |
|-------------------|---------------------------|---------------------------------|
|alpha = 0.01       | Regularisasi ringan       | Fitur tetap penting, tidak banyak yang dihapus|
|l1_ratio = 0.5     | Kombinasi Lasso + Ridge seimbang| Model menyeimbangkan seleksi fitur & stabilitas|

Artinya, model tidak membutuhkan penalti besar karena datanya cukup bersih/stabil, tidak menghapus banyak fitur (tidak agresif seperti Lasso tinggi), tetap menjaga stabilitas saat fitur saling berkorelasi, memberikan performa yang lebih baik dari linear murni tapi lebih sederhana dibanding model kompleks seperti Random Forest.

**Parameter Penting pada ElasticNet()**
|Parameter                |  Fungsi                                            |
|-------------------------|----------------------------------------------------|
|alpha                    | Kontrol kekuatan penalti. Semakin besar -> semakin banyak koefisien ditekan|
|l1_ratio                 | Mengatur kombinasi antara Lasso dan Ridge. 0= Ridge murni dan 1= Lasso murni. 0.5 = kombinasi seimbang|
|max_iter                 | Jumlah iterasi maksimum untuk mencapai konvergensi |
|tol                      | Toleransi error, menentukan kapan algoritma berhenti|                       
|selection                | cara update koefisien: `cyclic` (urutan) atau `random`|

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.8569 | Model menjelaskan >85% variasi harga tiket |
| **MAE** | 4932.41| Rata-rata selisih absolut antara prediksi dan nilai sebenarnnya|
| **RMSE** | 8587.98 | Akar dari rata-rata kuadrat error, berapa kesalahan prediksi model |

**Interpretasi Angka**
- RMSE = 8587.98 merupakan nilai rata-rata kesalahan prediksi sekitar +8.587 satuan. Ini lebih sensitif terhadap kesalahan besar. Saat model salah, terkadang salahnya bisa sampai +8.587 atau lebih ini menandakan adanya outlier/error tinggi di sebagian point.
- MAE = 4932.42 menunjukkan rata-rata selisih absolut antara prediksi dan nilai sebenarnya adalah +4.932 satuan target. Ini menggambarkan kesalahan tipikal yang lebih stabil karena tidak dipengaruhi outlier.
- R2 = 0.8569, artinya 85.69% variasi data target dapat dijelaskan oleh model ElasticNet. Sisanya 14.31% tidak dapat dijelaskan(mungkin karena faktor luar, noise, atau fitur yang belum digunakan).

---

## 📊 Visualisasi Data
### 1. Outlier Datasheet
![Outlier Data](Assets/Outlier.png)<br>
Ditemukan adanya **outlier** yaitu nilai yang jauh berbeda dari mayoritas data. Outlier ini dapat menyebabkan:
- 📉 **Model bias** → prediksi rata-rata jadi terlalu tinggi/rendah
- 📊 **Distribusi miring (skewed)** → membuat error lebih besar pada harga normal
- ⚡ **Training tidak stabil** → terutama untuk algoritma sensitif terhadap distribusi target
Oleh karena itu, dilakukan **Transformasi logaritmik pada target (price)** menggunakan `TranformedTargetRegressor`. Hal ini membuat distribusi lebih normal dan model lebih mudah belajar.

### 2. Prediksi vs Nilai Aktual
![Prediksi vs Harga Nilai Aktual](Assets/Prediksi%20vs%20Harga%20Aktual.png)<br>
Visualisasi di atas merupakan Scatter Plot Prediksi vs Nilai Aktual, visualisasi membantu untuk mengerti perilaku model secara intuitif hal yang sering tidak terlihat hanya dari angka metrik. Plot ini menunjukkan seberapa dekat hasil prediksi dengan kenyataan. Semacam uji keakuratan visual, melengkapi metrik Numerik(MAE, RMSE, R2). Jika R2 mendekati 1 dan titik berjejer di sekitar garis merah -> Model bagus.
Interpretasi hasil:
- Kalau titik-titik biru banyak yang menempel di garis merah, berarti model prediksi sangat akurat.
- Titik yang jauh dari garis merah = error prediksi yang lebih besar
- Semakin rapat titik ke garis merah -> semakin tinggi nilai R2 (koefisien determinasi)
- Menambahkan nilai R2 di plot, jadi pembaca bisa langsung lihat seberapa baik model menjelaskan variasi data.

### 3. Distribusi Residual Error
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

### 4. Residual vs Nilai Prediksi
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
 
### 5. Fitur Penting (Gini Importance)
![Gini Importance](Assets/Fitur%20Penting.png)<br>
Visualisasi ini adalah Feature Importance Plot dari Random Forest, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

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
best_model = joblib.load("rfr-flight_price_prediction.pkl")

# Data baru
data_baru = pd.DataFrame({
    "source_city": ["Jakarta"],
    "departure_time": ["Malam"],
    "stops": [0],
    "arrival_time": ["Pagi"],
    "destination_city": ["Bali"],
    "class": ["Business"],
    "days_left": [7],
    "duration": [2.5]
})

# Prediksi harga
prediksi = best_model.predict(data_baru)[0]
print(f"Prediksi Harga Tiket:  {prediksi:,.2f}")
```

## 🔮 Potensi Pengembangan
- Tambah fitur Airlines karena setiap maskapai memiliki harga berbeda.
- Validasi dengan K-Fold CV untuk hasil yang lebih stabil.
- Analisis error per segmen (mis. kelas Business vs Economy).
