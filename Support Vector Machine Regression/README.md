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

## 🧪 Catatan Eksperimen

Selama proses pelatihan model Support Vector Machine (SVM) dengan kernel RBF (Radial Basis Function) ini pelatihan tidak berhasil diselesaikan dalam waktu yang wajar. Eksperimen ini dijalankan menggunakan Google Colab dengan spesifikasi runtime standar.

### 🔍 Rincian Eksperimen
- Model yang digunakan : SVR (kernel='rbf')
- Metode Tuning : RandomizedSearchCV
- Parameter yang dicari:
  1. C : 0.1 - 300 (logspace)
  2. epsilon : 0.05 - 5.0 (linspace)
  3. gamma : ['scale','auto']
- Jumlah iterasi (n_iter) : 3
- Cross-validation (CV) : 2 folds
- Jumlah kandidat total : 6 kombinasi parameter
- Durasi berjalan       : 11 jam 29 menit (belum selesai/ tidak konvergen)

### ⚙️ Analisis Teknis
- Kompleksitas Komputasi, SVM dengan kernel RBF memiliki kompleksitas waktu 0(n2), di mana `n` adalah jumlah sampel. Dataset proyek ini memiliki ratusan ribu baris data, menyebabkan waktu training membengkak secara eksponesial.
- Kernel non-linear, dimana kernel RBF melakukan transformasi non-linear untuk memisahkan data yang tidak dapat dipisahkan secara linear. Proses ini membutuhkan banyak perhitungan jarak antar titik dalam ruang berdimensi tinggi.
- Tidak ada batas iterasi, parameter default `max_iter=-1` menyebabkan model terus mencoba mencapai konvergensi tanpa batas waktu tertentu. Karena itu, training tidak pernah berhenti meskipun sudah berjalan lebih dari 11 jam.
- Penggunaan RandomizedSearchCV walaupun menggunakan `n_iter` hanya 3 dan `cv` hanya 2. Setiap kombinasi parameter tetap harus melakukan fitting berulang kali. Ini memperbanyak total proses training dan evaluasi.

### 🧠 Insight dan Pembelajaran
- SVM (RBF) sangat kuat untuk data non-linear, namun tidak efisien untuk dataset beasr
-