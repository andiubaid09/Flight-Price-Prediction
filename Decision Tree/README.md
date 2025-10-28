# 🌳 Prediksi Harga Tiket Pesawat dengan Decision Tree Regressor

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **Decision Tree Regressor** sebagai model utama.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.

---

## 📖 Penjelasan Tentang Decision Tree Regressor
Decision Tree adalah algoritma Machine Learning yang bekerja seperti struktur pohon keputusan berbasis aturan if-else. Analoginnya seperti data akan "ditanyakan" pertanyaan sederhana, misalnya: "Apakah harga < 5000?" -> Jika iya ke kiri, kalau tidak ke kanan. Pertanyaan seperti ini terus terjadi sampai model mencapai keputusan akhir (leaf node), jika digunakan untuk prediksi harga -> Gunakan *Decision Tree Regressor* dan jika digunakan untuk prediksi kategori -> Gunakan *Decision Tree Classifier*.

Bagaimana cara kerja dari Decision Tree?
1. Akar (root) -> Node pertama, memilih fitur terbaik untuk melakukan pemisahan
2. Cabang (branch) -> Hasil dari jawaban "if/else (yes/no)" dari suatu fitur
3. Daun (leaf) -> Tempat hasil prediksi akhir
4. Proses pemecahan data ini disebut splitting yang disusun oleh aturan if-else.

Contoh aturan if-else pada regresi harga tiket ini:
```bash
if (stops <= 1):
    if (days_left > 30):
        Prediksi harga = 3500
    else:
        Prediksi = 5000
else:
    if (class == "Business"):
        Prediksi = 12000
    else:
        Prediksi = 7000
```

Kenapa disebut if-else? Karena setiap node menghasilkan pertanyaan logika sederhana dan model hanya membagi data berdasarkan batas nilai fitur serta sangat mudah dipahami dan interpretable model.

Berikut adalah kelebihan dan kekurangan dari Decision Tree:
|Kelebihan                                 |Kekurangan                                    |
|------------------------------------------|----------------------------------------------|
|Mudah dipahami (mirip alur logika manusia)|Mudah overfitting (kalau tanpa pengaturan parameter)|
|Tidak perlu feature scaling (standarisasi/normalisasi)|Sensitif terhadap perubahan kecil pada data|
|Bisa menangani data numerik & kategorikal| Tidak sebaik RandomForest                     |
|Interpretasi mudah, bisa diambil feature importances|                                              |

Kapan Decision Tree digunakan? Gunakan Decision Tree jika:
1. Butuh model yang mudah dipahami (interpretable)
2. Data memiliki hubungan non-linear
3. Data gabungan: Numerik + Kategorikal
4. Saat dataset tidak terlalu besar & ingin cepat
5. Untuk mengetahui fitur mana yang paling penting
6. Sebagai dasar model lebih kuat (RandomForest, XGBoost, Lightboost, dll)

Kapan Decision Tree tidak cocok untuk digunakan?
1. Data sangat besar, gunakan yang lebih powerful seperti RandomForest atau XGBoost
2. Butuh prediksi sangat akurat
3. Data linear sederhana, gunakan Linear, Ridge, Lasso atau ElasticNet
4. Ingin model stabil

Jangan gunakan jika dataset besar atau butuh akurasi maksimal, lebih baik pake RandomForest/XGBoost.

Kesimpulannya adalah Decision Tree adalah model berbasis logika if-else yang membagi data secara hierarkis sampai mencapai prediksi akhir. Model ini sangat intuitif dan fleksibel, tapi perlu pengaturan hyperparameter untuk menghindari overfitting.

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
Dilakukan dengan **GridSearchCV** pada parameter utama Decision Tree:
- `max_depth` → Kedalaman pohon
- `min_samples_split` → Minimum data sebelum node boleh di split
- `min_samples_leaf` → Minimum data dalam leaf node
- Hyperparameter menggunakan **GridSearchCV** ditemukan:
  1. max_depth = 20
  2. min_samples_split = 2
  3. min_samples_leaf = 10

**Interpretasi Angka dari Hyperparameter**
- max_depth = 20, Artinya pohon keputusan boleh tumbuh hingga 20 level kedalaman (20 aturan if-else bertingkat), semakin besar kedalaman, semakin detail model mempelajari data. Depth = 20 menunjukkan bahwa model cukup kompleks tetapi tidak dibiarkan bebas sampai overfitting parah (karena masih ada batasan leaf)
- min_samples_split = 2, Ini adalah jumlah minimum sample yang diperlukan sebelum node boleh dibagi. Nilai 2 adalah nilai default (paling permisif) artinya selama ada 2 data dalam node, dia tetap boleh di split. Model berusaha mempelajari pola sekecil mungkin. Namun, resiko overfitting tetap diminimalkan karena ada batasan di min_samples_leaf.
- min_samples_leaf = 10, Ini adalah jumlah minimum sample di setiap leaf (ujung pohon). Artinya: satu keputusan akhir harus berbasis setidaknya 10 data, tidak boleh hanya 1-2 data saja. Dampaknya dapat mengurangi overfitting, membuat keputusan lebih general, tidak terlalu mengikuti noise dan cocok untuk dataset real-world yang mengandung variasi dan outliers

**Parameter Penting pada DecisionTreeRegressor()**
|Parameter                |  Fungsi                               | Dampak                              |
|-------------------------|---------------------------------------|-------------------------------------|
|max_depth|Batas maksimal kedalaman (tingat) pohon|Mencegah pohon terlalu dalam  dan overfitting|
|min_samples_split|Minimum jumlah sampel di node sebelum dibagi|Mengontrol kapan node boleh displit|
|min_samples_leaf|Minimum jumlah sampel di leaf/daun (akhir cabang)|Membuat model lebih general dan hindari leaf dengan data sedikit|
|max_features|Jumlah fitur yang boleh dipakai untuk split di setiap node|Mengontrol kompleksitas dan kecepatan training|                       
|criterion|Fungsi untuk mengukur kualitas pemisahan node|Untuk regresi default adalah `squared_error` (MSE). Ada juga `friedman_mse`, `absolute_error`, `poisson`|
|max_leaf_nodes|Jumlah maksimum daun(leaf) dalam pohon|Alternatif untuk mengontrol ukuran pohon|
|random_state|Seed untuk membuat hasil split tetap konsisten|Penting agar hasil reproducible|
|min_weight_fraction_leaf|Minimum presentase sampel untuk setiap leaf berdasarkan bobot|Dipakai kalau dataset punya weighting|
|splitter|Starategi memilih split:`best` atau `random`|`best` cari split terbaik, `random` lebih cepat tapi kurang akurat|

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.9517 | Model menjelaskan >85% variasi harga tiket |
| **MAE** | 2822.91| Rata-rata selisih absolut antara prediksi dan nilai sebenarnnya|
| **RMSE** | 4991.25 | Akar dari rata-rata kuadrat error, berapa kesalahan prediksi model |

**Interpretasi Angka**
- RMSE = 4991.25 merupakan nilai untuk mengukur besar kesalahan prediksi, tapi lebih menekankan pada error besar (karena dikuadratkan). Nilai 4991.25 menunjukkan rata-rata penyimpangan prediksi sekitar 4.9 ribu unit dari nilai sebenarnya. Karena RMSE > MAE, berarti ada beberapa error besar (outlier) yang masih memengaruhi model tapi secara umum masih sangat bagus. Model stabil, tapi sesekali ada prediksi yang meleset jauh
- MAE = 2822.91 menunjukkan rata-rata kesalahan prediksi model sebesar +2822.91. Artinya, setiap prediksi meleset sekitar 2.8 ribuan dari nilai sebenarnya. Nilai ini relatif kecil dibanding skala target. Prediksi rata-rata model hanya salah sekitar +2822 dari nilai asli.
- R2 = 0.9517, artinya 95.17% variasi data target dari data uji yang dapat dijelaskan oleh model Decision Tree. Nilai ini sangat tinggi, menandakan model sangat akurat dan mampu menangkap pola hubungan fitur dengan target dengan baik. Hanya sekitar 4.83% variasi data yang tidak bisa dijelaskan oleh model (noise/error). Model sangat kuat, prediksi harga/tiket mendekati nilai sebenarnya.

Model Decision Tree secara keseluruhan berperforma sangat baik dan generalisasinya kuat.

---

## 📊 Visualisasi Data
### 1. Prediksi vs Nilai Aktual
![Prediksi vs Harga Nilai Aktual](Assets/Harga%20Prediksi%20vs%20Nilai%20Aktual.png)<br>
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
![Gini Importance](Assets/Top%2015%20Features%20Importances%20DT.png)<br>
Visualisasi ini adalah Feature Importance Plot dari Decision Tree, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

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
best_model = joblib.load("ElasticNet_pipeline_predictions_price.pkl")

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
- Menambahkan fitur lebih relevan (feature engineering) 
- Scaling ulang data (standardization).
- COba tuning l1_ratio lebih halus (0.1, 0.3, 0.7, 0.9)