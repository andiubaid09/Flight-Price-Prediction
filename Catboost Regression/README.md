# 🐱 Prediksi Harga Tiket Pesawat dengan CatBoost (Categorical Boosting)

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **CatBoost Regressor** sebagai model utama.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.

---

## 📖 Penjelasan Tentang CatBoost
CatBoost singkatan dari "Categorical Boosting" adalah salah satu algoritma **gradient boosting** modern buatan Yandex (mesin pencari Rusia) yang dirancang untuk menangani fitur kategorikal secara otomatis, mencegah overfitting lewat teknik matematis canggih (ordered boosting), lebih cepat dan efisien dibanding XGBoost dan LightGBM di banyak kasus. Catboost dibangun di atas konsep gradient boosting yaitu membuat banyak pohon keputusan kecil (weak learners) secara berurutan, di mana tiap pohon memperbaiki kesalahan dari pohon sebelumnya. Catboost dibangun di atas prinsip Gradient Boosting Decision Tree (GBDT). Secara intuitif:
1. Buat pohon pertama -> prediksi awal masih jelek.
2. Hitung error (residual)
3. Buat pohon kedua yang fokus belajar dari error tadi
4. Ulangi ratusan-ribuan kali hingga error makin kecil.
5. Gabungkan semua pohon jadi satu model kuat (boosted ensemble)

Catboost banyak digunakan dalam kasus regresi, klasifikasi dan ranking karena kemampuannya yang luar biasa dalam menemukan pola non-linear yang kompleks dari data mentah tanpa preprocessing yang rumit.
Algoritma ini punya keunggulan besar untuk data categorical features (fitur non-numerik seperti gender, kota, produk, dsb). Di algoritma lain seperti XGBoost dan LightGBM, fitur kategorikal harus di-encode manual menggunakan onehotencoder atau label encoding. Sementara di CatBoost bisa langsung memproses kategori tanpa encoding manual menggunakan konsep statistik internal bernama **target statistics** yang menjaga supaya tidak terjadi data leakage.

Alih-alih melakukan one-hot encoding, CatBoost menggunakan metode Target Encoding yang aman yaitu mengubah kategori menjadi nilai statistik berdasarkan target. Namun, CatBoost menghitung nilai encoded_value secara bertahap sehingga tidak terjadi kebocoran informasi dari target keseluruhan. Dengan metode ini, catboost lebih efisien daripada one-hot encoding, dapat menangani fitur dengan ratusan kategori unik dan menghindari overfitting akibat target leakage.

CatBoost membangun model sebagai kombinasi dari banyak Decision Tress yang dangkal di mana setiap pohon memperbaiki kesalahan dari pohon sebelumnya.Proses training-nya meliputi:
1. Membagi dataset menjadi beberapa subset acak
2. Melatih pohon keputusan pada residual (selisih antara target aktual dan prediksi sebelumnya)
3. Menggabungkan hasil semua pohon untuk memperbarui prediksi
4. Mengulang hingga mencapai jumlah iterations yang ditentukan atau early stopping

Berikut adalah kelebihan CatBoost:
|Kelebihan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Ordered Boosting                          |Mencegah target leakage dan overfitting dengan melatih model secara berurutan menggunakan subset data yang berbeda|
|Handling Categorical Features             |Dapat menangani fitur kategorikal secara otomatis tanpa perlu onehotencoder atau label encoding|
|Cepat dan Efisien                          |Optimized untuk CPU dan GPU bahkan pada dataset besar|
|Performa Stabil                            |Tidak mudah overfit dan lebih konsisten dibanding XGBoost atau LightBGM pada dataset kecil-menengah|
|Interoperability dengan Scikit-Learn   |Bisa langsung dipakai dalam Pipeline dan GridSearch/RandomizedSearchCV|


Berikut adalah kelemahan CatBoost:
|Kelemahan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Training Time Lebih Lama |Meskipun cepat dibanding boosting klasik, CatBoost cenderung lebih lambat dari model linear atau random forest, terutama saat tuning hyperparameter seperti GridSearchCV dan RandomizedSearchCV|
|Ukuran Model Cukup Besar    |Karena model terdiri dari banyak decision trees, hasil akhir bisa memakan memori yang signifikan (ratusan MB untuk dataset besar)|
|Kurang Fleksibel pada Produksi Ringan (Embedded/IoT)|CatBoost tidak cocok dijalankan langsung pada perangkat seperti Raspberry Pi kecil karena memerlukan memori besar dan dependensi Python|
|Tidak Selalu Unggul di Semua Kasus     | Untuk datased dengan relasi linear sederhana, regresi linear atau ridge regression bisa lebih cepat dan hampir sama akuratnya|

Kesimpulannya adalah CatBoost algoritma yang dikembangkan Yandex, dirancang untuk menangani data tabular secara efisien terutama yang mengandung banyak fitur kategorikal tanpa perlu preprocessing kompleks seperti one-hot encoding. Catboost termasuk dalam keluarga ensemble learning, di mana banyak model decision tree baru berfokus memperbaiki kesalahan yang dibuat oleh tree sebelumnya.

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
Dilakukan dengan **RandomizedSearchCV** pada parameter utama CatBoost:
- `iterations` → Jumlah pohon (round boosting)
- `learning_rate` → Kecepatan belajar
- `max_depth` → Kedalaman pohon
- `l2_leaf_reg` →  Regularisasi L2
- Hyperparameter menggunakan **RandomizedSearchCV** ditemukan:
  1. iterations = 1500
  2. learning_rate = 0.15
  3. max_depth = 10
  4. l2_leaf_reg = 15

**Interpretasi Angka dari Hyperparameter**
- iterations = 1500, ini berarti jumlah total boosting rounds atau decision trees yang dibangun secara bertahap oleh model. Model membangun 1500 pohon keputusan secara berurutan. Semakin banyak iterasi, semakin kompleks modelnya tapi juga berisiko overfitting jika terlalu tinggi.
- learning_rate = 0.15, menentukan seberapa besar kontribusi setiap tree baru dalam memperbaiki kesalahan model sebelumnya. Nilai kecil membuat model belajar perlahan tapi lebih stabil; nilai besar mempercepat konvergensi tapi bisa menyebabkn overshooting (overfitting atau tidak stabil). Nilai 0.15 sedikit lebih tinggi dari default (0.1) artinya model belajar dengan laju agak cepat. Karena jumlah iterasi juga cukup tinggi (1500), kombinasi ini masih seimbang dan cepat belajar tapi tidak terlalu agresif.
- max_depth = 10, Kedalaman maksimum setiap decision tree. Semakin besar max_depth, semakin kompleks pola yang bisa ditangkap tapi risiko overfitting juga meningkat. Nilai 10 termasuk dalam kategori menengah - tinggi. Artinya setiap pohon mampu membentuk aturan yang cukup kompleks untuk menangkap interaksi antar fitur. Karena dataset bisa dibilang cukup besar (240 ribu rows) dan mungkin memiliki hubungan non-linear antar variabel, kedalaman 10 adalah pilihan yang masuk akal. Model bisa memahami pola kompleks tanpa terlalu berlebihan.
- l2_leaf_reg = 15, parameter regularisasi L2 (mirip dengan ridge regression) yang menambahkan penalti terhadap bobot besar pada leaf value dari tree. Tujuannya adalah mengontrol kompleksitas model agar tidak overfitting. Nilai ini lebih tinggi dari default (3), artinya model diberi penalti lebih kuat agar tidak terlalu mengikuti noise pada data training. Ini membuat model lebih konservatif dalam mempelajari pola yang terlalu spesifik.

Dengan kombinasi ini menghasilkan model seimbang antara akurasi tinggi dan generalisasi baik, ideal untuk dataset besar dan kompleks.

**Parameter Penting pada CatBoostRegressor()**
|Parameter                |  Fungsi                               | Dampak                              |
|-------------------------|---------------------------------------|-------------------------------------|
|iterations|Menentukan jumlah total boosting iterations(jumlah pohon)|Semakin besar nilainya, semakin kompleks model|
|max_depth|Kedalaman maksimum setiap pohon|Nilai tinggi, membuat model bisa tangkap pola kompleks tapi resiko overfit, kecil bisa underfit|
|learning_rate|Mengontrol laju pembelajaran model|Nilai kecil (0.01-0.1) membuat model belajar lebih lambat tapi stabil, nilai besar (0.15-0.3) mempercepat konvergensi namun bisa tidak stabil. Biasanya diimbangi dengan jumlah iterations|
|l2_leaf_reg|Regularisasi L2 pada nilai daun (leaf value)|Mengontrol kompleksitas model, nilai besar (10-30) mencegah overfitting, nilai kecil (1-5) meningkatkan fleksibilitas model|                       
|bagging_temperature|Mengatur kekuatan randomness pada bootstrap sampling| Nilai 0 - deterministik, nilai tinggi (>1) = lebih acak, mencegah overfitting. Biasanya 0-1 untuk regresi|
|border_count|Jumlah borders (batas binning) untuk fitur numerik|Mengatur seberapa halus fitur numerik di-split. Default 254, nilai kecil mempercepat training tapi mengurangi akurasi|
|random_strength|Kontrol tingkat keacakan dalam pemilihan split|Nilai besar menambah regularization, mencegah overfitting. Biasanya 1-20|
|grow_policy|Strategi pertumbuhan pohon|`SymnetricTree` (default, cepat dan stabil) atau `Depthwise`/`Lossguide` (lebih fleksibel tapi lambat)|
|min_data_in_leaf|Minimum jumlah sampel di setiap daun|Nilai besar membuat model lebih konservatif. Cocok untuk dataset besar|
|subsample|Fraksi data yang digunakan per iterasi (mirip bagging)|Nilai 0.7 - 0.9, umum digunakan untuk mempercepat training dan menambah regularisasi|
|colsample_bylevel|Persentase fitur yang digunakan di tiap level pohon|Membantu mencegah overfitting pada dataset dengan fitur banyak|
|early_stopping_rounds|Berhenti otomatis jika tidak ada peningkatan dalam beberapa iterasi|Penting untuk efisiensi dan mencegah overfit|
|rsm|Random subspace method - proporsi fitur yang digunakan per split|Mirip  `colsample_bylevel`, umumnya antara 0.8 - 1.0|
|loss_function|Menentukan fungsi kehilangan yang dioptimalkan|Untuk regresi : `RMSE`,`MAE`,`r2`, dll dan untuk Classification: `accuracy_score`,`precision`,`recall`|
|eval_metric|Metrik evaluasi untuk pemantauan training| Misalnya:`RMSE`,`R2`. Tidak mempengaruhi training tapi membantu memilih model terbaik|
|random_seed|Nilai acak untuk reproduksibilitas|Pastikan konsisten agar hasil model bisa diulang. Biasanya 42 sama seperti random_state|
|verbose| Mengatur tingkat log output selama training|Nili integer (misal 100) untuk menamppilkan progress setiap 100 iterasi|

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