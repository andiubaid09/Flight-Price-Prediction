# 🐱 Prediksi Harga Tiket Pesawat dengan CatBoost (Categorical Boosting)

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **LightGBM Regressor** sebagai model utama.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.

---

## 📖 Penjelasan Tentang LightGBM
LightGBM adalah algoritma *gradient boosting* yang dikembangkan oleh Microsoft Research. Dirancang untuk lebih cepat daripada XGBoost, terutama untuk dataset besar, lebih efisien dalam penggunaan memori, mampu menagani fitur numerik dan kategorikal secara langsung tanpa perlu *one-hot encoding*. Tujuan utama LGBM adalah menyediakan algoritma boosting yang sangat efisien, cepat dan menggunakan memori lebih sedikit dibandingkan algoritma boosting tradisional (seperti XGBoost versi awal) terutama ketika berhadapan dengan dataset yang sangat besar.

Perbedaan fundamental antara XGBoost dan LightGBM terletak pada startegi mereka dalam menumbuhkan pohon keputusan dan teknik pengoptimalan komputasi yang mereka gunakan untuk mencapai efisiensi. XGBost mengimplementasikan strategi pertumbuhan pohon Level-wise (berdasarkan level) yang seragam, di mana ia membelah semua node pada level kedalaman yang sama sebelum melanjutkan ke level berikutnya. Pendekatan ini secara eksplisit mengontrol kedalaman pohon dan cenderung mengurangi risiko overfitting, tetapi mungkin lebih lambat karena harus memeriksa split yang tidak penting. Sebaliknya, LightGBM mengadopsi strategi Leaf-wise (berdasarkan daun) yang secara proaktif memilih untuk membelah daun yang menghasilkan pengurangan loss (kerugian) terbesar. Strategi ini memungkinkan LightGBM  membangun pohon yang lebih dalam dan lebih kompleks dengan lebih cepat karena berfokus pada area yang paling membutuhkan perbaikan, namum bisa lebih rentan terhadap overfitting pada dataset yang lebih kecil sehingga memerlukan kontrol parameter kedalaman yang lebih ketat.

Selain itu, LightGBM memperkenalkan dua inovasi utama untuk meningkatkan kecepatan secara drastis pada dataset besar: **Gradien-based One-Side Sampling (GOSS)** dan **Exclusive Feature Bundling (EFB)**. GOSS secara cerdas mengurangi jumlah data pelatihan dengan hanya menggunakan semua sampel dengan gradien besar (kesalahan besar) dan mengambil sampel secara acak dari data dengan gradien kecil, sehingga menjaga akurasi sambil mengurangi waktu komputasi. Sementara itu, EFB mengelompokkan fitur-fitur yang saling eksklusif (jarang terjadi bersamaan) menjadi satu fitur gabungan yang secara efektif mengurangi dimensi fitur (jumlah kolom) dan mempercepat proses pencarian split. Meskipun XGBoost juga sangat teroptimasi dan mendukung paralelisasi, optimalnya berfokus pada proses split (pembelahan) pada setiap node untuk efisiensi, sedangkan LightBGM berfokus pada mengurangi data yang perlu diproses dan jumlah fitur, membuatnya secara umum lebih cepat dan menggunakan memori yang lebih sedikit ketika dihadapkan pada dataset yang berukuran sangat besar.

Bagaimana cara kerja dari LightGBM? sama seperti XGBoost, LightGBM membangun sekumpulan pohon keputusan (decision tree) secara bertahap (boosting). Namum perbedaan utamanya ada di bagaimana pohon dibangun:
1. XGBoost -> Level-wise growth, pohon dibangun lebar dulu, tiap level dikembangkan serentak. Lebih stabil, tapi bisa lambat dan boros memori.
2. LightGBM -> Leaf-wise growth (Best-first search), pohon tumbuh ke arah daun dengan penurunan error terbesar (loss reduction), lebih efisien dan akurat, tapi jika tidak dikontrol bisa overfitting.

Berikut contoh bagaimana analogi LightGBM bekerja dengan memprediksi harga tiket:
1. Pohon pertama menebak berdasarkan kota asal & tujuan
2. Pohon kedua menyesuaikan berdasarkan waktu keberangkatan
3. Pohon ketiga memperbaiki lagi berdasarkan jumlah hari tersisa dan seterusnya

Setiap pohon belajar memperbaiki kesalahan pohon sebelumnya. Hasil akhirnya: model yang sangat presisi dan cepat.

Berikut adalah kelebihan LightGBM:
|Kelebihan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Cepat & Efisien                            |Menggunakan histogram-based learning dan leaf-wise growth, training jauh lebih cepat daripada XGBoost|
|Dapat menangani data besar                 |Bisa memproses jutaan data tanpa bottleneck memori|
|Ramah memori                               |Tidak perlu onehot encoding eksplisit, mendukung fitur kategorikal langsung|
|Handle missing value                       |Secara otomatis, tanpa imputasi manual           |
|Presisi tinggi                             |Leaf-wise membuat model fokus dibagian yang paling sulit diprediksi|
|Mendukung paralelisme dan GPU              |Dapat mempercepat pelatihan secara signifikan |

Berikut adalah kelemahan LightGBM:
|Kelemahan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Cenderung overfitting            |Karena pertumbuhan pohon bersifat leaf-wise, model bisa terlalu fokus pada noise|
|Kurang stabil di dataset kecil   |Untuk data kecil atau tidak seimbang, hasilnya tidak konsisten|
|Sulit diinterpretasi                       |Karena banyak pohon kompleks, interpretasi fitur tidak sederhana regresi linear|

Kapan LightGBM digunakan? Gunakan LightGBM jika:
1. Data tabular (CSV, excel, sensor, log, dsb)
2. Dataset besar (>1 juta baris)
3. Ingin waktu yang lebih cepat dan presisi tinggi

Kesimpulannya adalah LightGBM algoritma yang dikembangkan microsoft research, dirancang untuk kecepatan tinggi, efisiensi memori, dan performa prediksi yang sangat baik.

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
Dilakukan dengan **GridSearchCV** pada parameter utama LightGBM:
- `n_estimators` → Jumlah pohon
- `learning_rate` → Kecepatan belajar
- `max_depth` → Kedalaman pohon
- Hyperparameter menggunakan **GridSearchCV** ditemukan:
  1. n_estimators = 500
  2. learning_rate = 0.2
  3. max_depth = 20

**Interpretasi Angka dari Hyperparameter**
- n_estimators = 500, ini berarti LightGBM akan mencoba membangun 500 pohon berturut-turut (500 boosting rounds). Memberi model kapasitas belajar yang besar dapat menangkap pola kompleks.
- learning_rate = 0.2, kontribusi tiap pohon akan dikalikan 0.2 sebelum dijumlahkan ke model keseluruhan. Nilai 0.2 termasuk cukup agresif/relatif tinggi (umumnya 0.01-0.1 sering dipakai untuk model stabil). Model akan konvergen lebih cepat (mungkin cukup dengan <500 pohon), tetapi ada risiko overfitting dan generalisasi menurun jika tidak diimbangi regulasi/early stopping.
- max_depth = 20, Artinya tiap pohon boleh hingga 20 level kedalaman-cukup dalam.Tiap pohon bisa menangkap interaksi fitur yang kompleks. Meningkatkan kemampuan model untuk menangkap pola non-linear kuat.

Dengan kombinasi ini, model berkapasitas sangat besar dan belajar cepat. Jika tidak ada early stopping dan regularisasi, model kemungkinan besar overfit ke data training (meski di beberapa kasus- jika data sangat banyak & bersih kombinasi ini masih bisa baik). Dalam kasus saya ini, memiliki banyak data dan sangat bersih dengan melakukan banyak preprocessing yang walaupun semuanya sudah bisa ditangani oleh LightGBM secara native seperti fitur kategorikal (onehot encoding).

**Parameter Penting pada LightGBMRegressor()**
|Parameter                |  Fungsi                               | Dampak                              |
|-------------------------|---------------------------------------|-------------------------------------|
|n_estimators|Jumlah total pohon(round boosting)|Banyak pohon maka lebih akurat tapi lama; berpotensi overfit tanpa early stopping|
|max_depth|Kedalaman maksimum setiap pohon|Nilai tinggi, membuat model bisa tangkap pola kompleks tapi resiko overfit, kecil bisa underfit|
|learning_rate|Mengontrol seberapa besar pembaruan bobot tiap pohon berkontribusi ke model akhir|Kecil (0.01 - 0.1) -> stabil tapi lambat; besar (0.2-0.3) cepat tapi rentan overfit|
|min_child_weight|Total bobot minimum untuk satu daun (alternatif kontrol overfit)|Mirip fungsinya dengan *min_child_samples*|                       
|min_child_samples|Jumlah minimum data dalam satu daun|Besar: model lebih halus(mencegah overfit), <kecil:lebih detail tapi berisiko overfit|
|num_leaves|Jumlah daun dalam pohon (kompleksitas cabang)|Disarankan lebih kecil untuk menghindari overfitting|
|random_state|Seed untuk membuat hasil split tetap konsisten|Penting agar hasil reproducible|
|colsample_bytree|Fraksi fitur yang digunakan tiap pohon|Biasanya 0.6 - 0.9|
|reg_alpha (L1 regularization)|Menambahkan penalti terhadap nilai absolut bobot|Membuat model lebih sparse (fitur yang tidak penting diabaikan)|
|reg_lambda (L2 regularization)|Penalti terhadap kuadrat bobot|Membuat model lebih stabil dan mengurangi overfitting|
|boosting_type|Jenis booting|Umum : `gbdt` (default), `dart` (dropout boosting -> mengurangi overfit), `gross` (faster for large data)|
|early_stopping_rounds|Berhenti otomatis jika tidak ada peningkatan dalam beberapa iterasi|Penting untuk efisiensi dan mencegah overfit|
|objective|Jenis tugas: `regression`,`binary`,`multiclass`|       |
|metric|Metrik evaluasi:`rmse`,`mae`,`r2`,`logloss`,dll|          |

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.9552 | Model menjelaskan >95% variasi harga tiket |
| **MAE** | 2809.01| Rata-rata selisih absolut antara prediksi dan nilai sebenarnnya|
| **RMSE** | 4861.22 | Akar dari rata-rata kuadrat error, berapa kesalahan prediksi model |

**Interpretasi Angka**
- RMSE = 4861.22, merupakan nilai untuk mengukur rata-rata kesalahan prediksi model terhadap nilai aktual sekitar +4861 satuan harga. RMSE sensitif terhadap error besar karena RMSE sensitif terhadap error besar (karena dikuadratkan), jadi semakin kecil nilainya, semain baik model dalam menghindari kesalahan ekstrem.
- MAE = 2809.01 menunjukkan rata-rata selisih jarak absolut antara nilai prediksi dan nilai aktual adalah 2809. Artinya secara umum, prediksi harga model menyimpang sekitar 2800 satuan harga dari nilai sebenarnya. Menunjukkan akurasi cukup tinggi untuk kasus regresi harga.
- R2 = 0.9552, artinya 95.52% model LightGBM mampu menjelaskan variasi nilai target (y) pada data uji. Dengan kata lain, hampir seluruh perubahan pada harga dapat dijelaskan oleh fitur yang digunakan. Hanya sekitar 4.5% variasi yang tidak bisa dijelaskan model.

Berdasarkan hasil evaluasi, model LightGBM yang telah dioptimasi melalui hyperparameter tuning menunjukkan performa prediksi yang sangat baik. Model ini bisa dikatakan stabil, akurat dan generalisasi baik dan sangat cocok digunakan sebagai model utama (baseline terbaik) untuk dataset harga tiket penerbangan ini.

---

## 📊 Visualisasi Data
### 1. Prediksi vs Nilai Aktual
![Prediksi vs Harga Nilai Aktual](Assets/Prediksi%20vs%20Nilai%20Aktual.png)<br>
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
 
### 5. Fitur Penting (Gini Importance)
![Gini Importance](Assets/Top%2015%20Feature.png)<br>
Visualisasi ini adalah Feature Importance Plot dari LightGBM, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

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
best_model = joblib.load("LGBM_Prediction_Ticket_Price.pkl")

# Data baru
data_baru = pd.DataFrame({
    "source_city": ["Delhi"],
    "departure_time": ["Morning"],
    "stops": ["zero"],
    "arrival_time": ["Aftenoon"],
    "destination_city": ["Mumbai"],
    "class": ["Economy"],
    "days_left": [7],
    "duration": [2.5]
})

# Prediksi harga
prediksi = best_model.predict(data_baru)[0]
print(f"Prediksi Harga Tiket:  {prediksi:,.2f}")
```

## 🔮 Potensi Pengembangan
- Hyperparameter tuning lebih jauh, eksplore parameter lain dengan **GridSearchCV** yang lebih luas, temukan konfigurasi yang paling optimal dan tambkah early_stopping_rounds
- Buat fitur baru yang relevan (feature engineering).
- Ensembling dengan model lain (XGBoost + RandomForest atau Voting Regressor)
- Regularisasi dan overfitting control, LightGBM punya 2 parameter L1 dan L2