# ⚔️ Prediksi Harga Tiket Pesawat dengan Support Vector Machine (Training no completed 🚫)

## 📝 Deskripsi Proyek
Proyek ini membangun model **Machine Learning** untuk memprediksi harga tiket pesawat berdasarkan berbagai fitur penerbangan. Model dikembangkan menggunakan **Scikit-learn Pipeline** dengan **Support Vector Regression (SVR)** sebagai model utama. Namun, eksperimen ini dengan kernel RBF untuk memprediksi harga tiket pesawat berjalan selama lebih dari 11 jam tanpa menyelesaikan proses pelatihan karena kompleksitas kernel non-linear pada dataset berskala besar.  

Untuk meningkatkan akurasi pada data harga yang memiliki distribusi miring (*skewed*), target variabel (`price`) ditransformasi secara logaritmik menggunakan **TransformedTargetRegressor (TTR)**.

---

## 📖 Penjelasan Tentang Support Vector Machine
Support Vector Machine (SVM) adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regressi, tapi lebih terkenal untuk klasifikasi. Tujuan SVM adalah mencari garis atau bidang yang memisahkan data dari dua kelas sebaik mungkin. Garis ini disebut sebagai **Hyperplane**. SVM bekerja dengan mencari sebuah hyperplane yang memisahkan data menjadi dua kelas berbeda. Hyperplane ini dapat diangap sebagai "garis batas keputusan (decision boundary)" yang dapat diupayakan agar memiliki jarak maksimum terhadap titik-titik data terdekat dari masing-masing kelas. Titik-titik yang berada paling dekat dengan hyperplane disebut support vectors dan mereka berperan penting dalam menentukan posisi serta orientasi hyperplane.

```bash
🔴🔴🔴      |      🔵🔵🔵
🔴🔴🔴   <--|-->   🔵🔵🔵
🔴🔴🔴      |      🔵🔵🔵

```

Tanda **|** adalah hyperplane. Support vector = titik paling dekat di sisi masing-masing kelas. SVM berusaha meminmalkan dua hal yaitu kesalahan klasifikasi dan ukuran margin yang kecil. Dalam konteks regressi (Support Vector Regression/SVR), SVM berusaha menemukan fungsi regresi optimal yang memiliki penyimpangan minimum (error) dari nilai aktual, sambil tetap mempertahankan margin toleransi (ε-insensitive zone) agar model tidak terlalu sensitif terhadap noise.

SVM memiliki dua pendekatan utama yaitu Linear SVM, digunakan jika data dapat dipisahkan secara linear, artinya terdapat garis lurus atau bidang datar yang bisa memisahkan kelas-kelas data dengan jelas dan Non-linear SVM digunakan jika data tidak bisa dipisahkan secara linear. Untuk menangani hal ini, SVM menggunakan pendekatan yang disebut *kernel trick*, yaitu mentransformasikan data dari ruang berdimensi rendah ke ruang berdimensi tinggi agar dapat dipisahkan secara linear di ruang tersebut.

Kernel merupakan fungsi yang memetakan data ke ruang fitur berdimensi lebih tinggi tanpa harus menghitung koordinatnya secara eksplisit. Pendekatan ini memungkinkan SVM menangani hubungan non-linear dengan efisien. Beberapa kernel yang umum digunakan antara lain:
|Jenis Kernel             | Fungsi                  | Kegunaan              |
|-------------------------|-------------------------|-----------------------|
|Linear                   |K(xi​,xj​)=xi​⋅xj​           | Untuk data yang dapat dipisahkan secara linear|
|Polynomial               |K(xi​,xj​)=(xi​⋅xj​+1)d      | Menangani hubungan polinomial antar fitur|
|RBF (Radial Basis Function)| K(xi​,xj​)=exp(−γ∥xi​−xj​∥2)| Digunakan untuk data non-linear|
|Sigmoid                  |K(xi​,xj​)=tanh(αxi​⋅xj​+c)  | Mirip fungsi aktivasi pada jaringan saraf tiruan|

Kernel RBF merupakan paling populer karena kemampuannya dalam menangani berbagai bentuk distribusi data yang kompleks dan non-linear


Berikut adalah kelebihan Support Vector Machine:
|Kelebihan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Akurat pada data berdimensi tinggi        |SVM tetap stabil dan akurat walaupun jumlah fitur sangat banyak (high-dimensional space), seperti pada teks, gambar atau genomik.|
|Efektif pada data non-linear              |Dengan penggunaan *kernel trick* (misalnya RBF), SVM mampu menangani pola hubungan antar fitur yang kompleks dan tidak linear|
|Bekerja baik pada dataset kecil hingga menengah|Karena berfokus pada titik-titik support vector, SVM tidak memerlukan seluruh data untuk membangun model, sehingga performanya lebih tetap baik pada data terbatas|
|Robust terhadap outlier ringan |Titik data ekstrem yang sedikit tidak terlalu memengaruhi posisi hyperplane, asalkan jumlahnya tidak berlebihan|
|Memiliki dasar teoretis kuat|SVM didasarkan pada teori optimisasi konveks, sehingga solusi yang diperoleh bersifat global optinum, bukan local minimum|
|Dapat digunakan untuk klasifikasi dan regressi| Varian SVM (yaitu Support Vector Regression/SVR) memungkinkan algoritma ini digunakan juga untuk tugas regresi kontinu|


Berikut adalah kelemahan Support Vector Machine:
|Kelemahan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Waktu komputasi tinggi pada dataset besar |Kompleksitas komputasi SVM meningkat drastis dengan jumlah sampel, sehingga tidak efisien untuk data besar (ratusan ribu hingga jutaan baris)|
|Sulit dalam pemilihan kernel yang tepat|Pemilihan kernel yang tidak sesuai (linear, polynomial, RBF, sigmoid) dapat menyebabkan performa model menurun signifikan|
|Memerlukan tuning parameter yang sensitif|Parameter seperti C, gamma, dan epsilon sangat mempengaruhi hasil akhir, sehingga perlu proses pencarian paramter (GridSearch/RandomSearch) yang mahal waktu|
|Kurang interpretatif| Hasil model berupa vektor bobot dan support vector yang sulit dijelaskan secara intuitif, tidak seperti Decision Tree yang mudah divisualisasikan|
|Sensitif terhadap skala fitur|SVM mengandalkan jarak antar titik (inner product), sehingga memerlukan normalisasi atau standarisasi fitur terlebih dahulu|
|Sulit dioptimalkan pada data dengan noise tinggi|Jika data memiliki banyak outlier atau noise, margin optimal sulit dicapai dan model dapat mengalami overfitting|

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
Dilakukan dengan **RandomizedSearchCV** pada parameter utama Support Vector Machine:
- `C` → Regularization
- `epsilona` → Insensitive Zone (khusus SVR)
- `gamma` → Kernel coefficient
- Hyperparameter menggunakan **RandomizedSearchCV** tidak dapat berhasil dengan waktu training yang sangat lama yaitu 11 jam.

**Parameter Penting pada Support Vector Machine()**
|Parameter                |  Fungsi                               | Penjelasan                          |
|-------------------------|---------------------------------------|-------------------------------------|
|C|Regularization parameter|Mengontrol trade-off antara margin besar dan kesalahan klasifikasi. Nilai kecil membuat margin lebar tapi toleran terhadap kesalahan (underfitting), sedangkan nilai besar memaksa model memisahkan data dengan benar tapi bisa overfitting|
|kernel|Jenis fungsi kernel|Menentukan fungsi pemetaan ke ruang fitur berdimensi lebih tinggi|
|degree|Derajat kernel polinomial|Hanya digunakan jika `kernel=poly`, menentukan derajat polinomial. Derajat yang lebih tinggi dapat menangkap hubungan lebih kompleks tapi juga meningkatkan overfitting|
|gamma|Parameter kernel RBF, poly, sigmoid|Mengontrol seberapa jauh pengaruh satu sampel pelatihan. Nilai besar = model fokus ke data terdekat (overfitting), nilai kecil = margin lebih halus (underfitting)|                       
|coef0|Koefisien kernel poly/sigmoid|Mengatur pengaruh antara komponen linear dan non-linear dalam kernel polinomial dan sigmoid. Biasanya hanya disetel saat pakai kernel tersebut|
|shrinking|Strategi optimasi|Boolean (True/False). Jika True, algoritma menggunakan heuristik `shrinking` untuk mempercepat pelatihan tanpa anyak kehilangan akurasi. Default:True|
|probability|Estimasi probabilitas output|Jika True, SVM menghitung probabilitas prediksi dengan metode Platt scaling(predict_proba). Ini menambah waktu pelatihan. Default :False|
|class_weight|Bobot tiap kelas|Berguna untuk data tidak seimbang. Misal `class_weight=balanced` otomatis memberi bobot proporsional terhadap frekuensi kelas|
|max_iter|Batas iterasi pelatihan|Menentukan jumlah maksimum iterasi saat proses fitting. Jika -1, tidak ada batas (default). Berguna untuk menghindari pelatihan yang terlalu lama|
|decision_function_shape|Bentuk output keputusan|Mengontrol bentuk hasil fungsi keputusan: `ovr` (one-vs-rest, default) untuk multi-class. `ovo` (one-vs-one) digunakan dalam kasus|
|tol|Toleransi konvergensi|Batas toleransi untuk menghentikan iterasi optimisasi. Nilai kecil membuat hasil lebih akurat tapi memperlambat pelatihan. Default:1e-3|

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.9553 | Model menjelaskan >95% variasi harga tiket |
| **MAE** | 2737.66| Rata-rata selisih absolut antara prediksi dan nilai sebenarnnya|
| **RMSE** | 4801.70 | Akar dari rata-rata kuadrat error, berapa kesalahan prediksi model |

**Interpretasi Angka**
- RMSE = 4801.70, mengindikasin bahwa rata-rata kesalahan prediksi model berada pada kisaran +4801.70  satuan terhadap nilai aktual. Karena RMSE menghitung akar dari rata-rata kuadrat kesalahan, metrik ini memberikan penalti yang lebih besar terhadap kesalahan dengan nilai ekstrem. Oleh karena itu, nilai RMSE yang masih relatif rendah menunjukkan bahwa model tidak hanya akurat secara umum, tetapi juga mampu menjaga stabilitas performa terhadap error besar (large deviations).
- MAE = 2737.66 menggambarkan bahwa rata-rata deviasi absolut antara hasil prediksi dan nilai sebenarnya berada di sekitar 2737 stauan. Berbeda dengan RMSE, MAE menghitung rata-rata kesalahan tanpa mengkuadratkan selisih, sehingga memberikan pandangan yang lebih realistis terhadap kesalahan rata-rata tanpa mengkuadratkan selisih, sehingga memberikan pandangan yang lebih realistis terhadap kesalahan rata-rata secara umum. Perbedaan antara RMSE dan MAE yang tidak terlalu jauh (rasio 1.75) menandakan bahwa distribusi error model bersifat relatif stabil tanpa adanya penyimpangan besar (outlier errors) yang mendominasi. Hal ini menunjukkan bahwa model menghasilkan prediksi yang konsisten di seluruh rentang data.
- R2 = 0.9553, r2 mengukur seberapa besar proporsi variasi pada data target yang dapat dijelaskan oleh model, di mana nilai 0 menunjukkan bahwa model tidak mampu menjelaskan data sama sekali, sedangkan nilai 1 menunjukkan kemampuan penjelasan sempurna. Dengan nilai 0.9553 dapat diinterpretasikan bahwa sekitar 95.53% variasi pada variabel target berhasil dijelaskan model dan hanya sekitar 4.47% sisanya yang tidak dapat dijelaskan. Kemungkinan disebabkan oleh fakor acak atau variabel lain yang tidak disertakan dalam proses training.

Secara keseluruhan, hasil evaluasi ini menunjukkan bahwa model CatBoost regressor mampu mempelajari hubungan kompleks antar fitur dengan sangat baik. Dengan parameter yang sudah dijelaskan sebelumnya, model ini mampu menangkap pola non-linear dalam data tanpa kehilangan stabilitas. Oleh karena itu, dapat disimpulkan bahwa model yang dikembangkan efisien, akurat dan robust, serta layak diterapkan untuk keutuhan prediksi berbasis regresi pada dataset berukuran besar dengan karakteristik serupa.

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
Visualisasi ini adalah Feature Importance Plot dari Catboost, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

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
best_model = joblib.load("CatBoost_Prediction_Flight_Ticket.pkl")

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
- Optimasi Hyperparameter yang lebih luas meskipun model telah melalui proses tuning dengan parameter. Ruang pencarian hyperparameter dapat diperluas untuk mencakup parameter lain
- Buat fitur baru yang relevan (feature engineering), Eksperimen dengan fitur dan seleksi atribut.