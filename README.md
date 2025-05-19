# Laporan Proyek Machine Learning - Athalla Naufal Muthahhari

## Domain Proyek: Prediksi Penjualan Video Game

Dalam industri video game yang sangat kompetitif, mengetahui potensi penjualan suatu game sangat krusial bagi para pengembang dan penerbit untuk merencanakan strategi pemasaran, alokasi sumber daya, dan distribusi. Dataset penjualan video game ini memuat informasi penjualan dari berbagai wilayah seperti NA (North America), EU (Europe), JP (Japan), dan Other, serta berbagai fitur seperti genre, platform, dan publisher. Proyek ini bertujuan untuk memprediksi `Global_Sales` dari sebuah game berdasarkan informasi tersebut menggunakan pendekatan regresi.

## Business Understanding

### Problem Statements
1. Bagaimana cara memprediksi total penjualan global (`Global_Sales`) sebuah game berdasarkan atribut-atribut lainnya?
2. Algoritma machine learning mana yang memberikan hasil paling akurat dalam memprediksi penjualan game?

### Goals
1. Mengembangkan model prediksi `Global_Sales` dari data fitur yang tersedia.
2. Membandingkan performa tiga model regresi untuk menentukan model terbaik.

### Solution Statements
- Membangun tiga model prediktif menggunakan:
  - Linear Regression
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- Melakukan evaluasi performa model menggunakan metrik **R²** dan **Mean Squared Error (MSE)**.
- Visualisasi performa prediksi setiap model terhadap data aktual untuk pemilihan model terbaik.

## Data Understanding

Dataset yang digunakan berasal dari Kaggle: [Video Game Sales Dataset](https://www.kaggle.com/datasets/zahidmughal2343/video-games-sale). Dataset ini mencatat penjualan video game dari berbagai platform dan wilayah di seluruh dunia.

### Struktur Dataset
- **Jumlah entri (baris):** 16.598
- **Jumlah fitur (kolom):** 11
- **Ukuran memori:** ±1.4 MB

Berikut ringkasan struktur kolom berdasarkan output `.info()`:

| Kolom         | Non-Null Count | Tipe Data | Keterangan |
|---------------|----------------|-----------|------------|
| `Rank`        | 16.598         | int64     | Peringkat berdasarkan total penjualan |
| `Name`        | 16.598         | object    | Nama game |
| `Platform`    | 16.598         | object    | Platform tempat game dirilis |
| `Year`        | 16.327         | float64   | Tahun rilis game (memiliki 271 missing values) |
| `Genre`       | 16.598         | object    | Genre game |
| `Publisher`   | 16.540         | object    | Penerbit game (memiliki 58 missing values) |
| `NA_Sales`    | 16.598         | float64   | Penjualan di Amerika Utara (juta unit) |
| `EU_Sales`    | 16.598         | float64   | Penjualan di Eropa (juta unit) |
| `JP_Sales`    | 16.598         | float64   | Penjualan di Jepang (juta unit) |
| `Other_Sales` | 16.598         | float64   | Penjualan di wilayah lain (juta unit) |
| `Global_Sales`| 16.598         | float64   | Penjualan total (target), hasil penjumlahan regional sales |

### Kondisi Nilai Kosong
Dataset memiliki beberapa nilai kosong yang perlu ditangani:

| Kolom       | Jumlah Nilai Kosong |
|-------------|----------------------|
| `Year`      | 271                  |
| `Publisher` | 58                   |

### Insight Awal
- `Global_Sales` merupakan penjumlahan langsung dari `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales`.
- Fitur `Year` memiliki tipe `float64` karena mengandung nilai kosong (`NaN`), meskipun secara logika nilainya seharusnya berupa `int`.
- Fitur `Name`, `Platform`, `Genre`, dan `Publisher` bertipe kategorikal (`object`) dan akan memerlukan encoding saat proses machine learning.
- Tidak ada nilai kosong pada fitur target `Global_Sales` maupun fitur penjualan per wilayah.


## Data Preparation

Tahapan data preparation dilakukan secara bertahap untuk memastikan kualitas data sebelum masuk ke proses pemodelan. Berikut langkah-langkah yang diterapkan:

### 1. Pembersihan Data
- **Menghapus baris dengan nilai kosong pada `Year` dan `Publisher`**  
  Tujuannya untuk menghindari error atau bias saat model membaca data kategorikal maupun temporal. Total baris yang dihapus:
  - `Year`: 271 baris
  - `Publisher`: 58 baris

- **Menghapus baris dengan `Global_Sales` kosong**  
  Walaupun sangat jarang, langkah ini dilakukan untuk memastikan target variabel tidak memiliki nilai hilang.

- **Menghapus baris dengan nilai `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales` semuanya nol**  
  Tujuannya adalah untuk menghilangkan *noise* yang berasal dari game yang secara realistis tidak memiliki penjualan sama sekali di semua wilayah. Data semacam ini dapat mengganggu pembelajaran model karena tidak memberi informasi variasi penjualan.

### 2. Seleksi Fitur Numerik
Fokus fitur ditujukan pada data numerik untuk baseline model regresi, yaitu:
- `NA_Sales`
- `EU_Sales`
- `JP_Sales`
- `Other_Sales`

Fitur-fitur ini menjadi *predictor* untuk memprediksi `Global_Sales`, yang secara logis memang merupakan hasil penjumlahan dari keempat fitur tersebut. Tujuannya adalah membangun baseline model dengan input sederhana terlebih dahulu.

### 3. Normalisasi (Scaling)
- **StandardScaler** digunakan untuk menstandarkan fitur numerik agar memiliki distribusi dengan mean 0 dan standar deviasi 1.
- Scaling diperlukan karena beberapa algoritma regresi (seperti Support Vector Regressor) sensitif terhadap skala data.

### 4. Split Data
- Dataset dibagi menjadi dua bagian menggunakan `train_test_split`:
  - **80%** data digunakan untuk **training**
  - **20%** data digunakan untuk **testing**
- Parameter `random_state=42` digunakan untuk memastikan pembagian data konsisten dan dapat direproduksi.

---

Langkah-langkah ini dilakukan untuk memastikan data yang masuk ke model bersih, representatif, dan sesuai format yang dibutuhkan oleh algoritma machine learning.


## Modeling

### Model 1: Linear Regression
- Pendekatan sederhana namun kuat untuk data yang berhubungan linier.
- Tidak memerlukan parameter khusus.
- Kelebihan: Interpretatif dan cepat.
- Kekurangan: Kurang fleksibel terhadap non-linearitas.

### Model 2: Random Forest Regressor
- Ensemble learning berbasis decision tree.
- Parameter default digunakan.
- Kelebihan: Menangani non-linearitas dan robust terhadap outlier.
- Kekurangan: Bisa overfitting jika tidak dikontrol.

### Model 3: Support Vector Regressor (SVR)
- Kernel-based regression model.
- Digunakan kernel RBF tanpa tuning.
- Kelebihan: Baik untuk data kecil.
- Kekurangan: Kurang akurat tanpa scaling dan tuning.

## Evaluation

### Metrik Evaluasi:
- **R² (coefficient of determination)**: Seberapa baik model menjelaskan varians data.
- **Mean Squared Error (MSE)**: Rata-rata kesalahan kuadrat dari prediksi.

### Hasil Evaluasi:
| Model                   | R² Score | MSE   |
|------------------------|---------|-------|
| Linear Regression      | 1.00    | 0.00  |
| Random Forest Regressor| 0.83    | 0.71  |
| SVR                    | 0.42    | 2.50  |

**Interpretasi**:
- Linear Regression mendapat hasil sempurna karena `Global_Sales` = jumlah 4 fitur input.
- Random Forest juga cukup baik dengan MSE < 1, cocok untuk prediksi non-linear.
- SVR gagal memodelkan data dengan baik tanpa tuning dan preprocessing tambahan.

### Visualisasi:
Disertakan grafik hasil prediksi vs nilai aktual dengan garis merah sebagai referensi ideal. Linear Regression sejajar sempurna, sedangkan Random Forest dan SVR menyimpang tergantung performanya.
![Visualisasi Model](image.png)

## Kesimpulan

- Linear Regression terlalu akurat karena target adalah hasil penjumlahan dari fitur input (problem identitas).
- Model yang lebih realistis untuk masa depan adalah **Random Forest**, terutama jika input data berasal dari metadata lain.
- SVR kurang sesuai untuk data ini tanpa tuning lanjutan.

**Saran Lanjutan**:
- Gunakan fitur seperti `Genre`, `Platform`, dan `Publisher` untuk prediksi regional sales.
- Lakukan one-hot encoding atau label encoding untuk variabel kategorikal.
- Coba hyperparameter tuning pada SVR dan Random Forest.
- Evaluasi tambahan dengan fitur `Year_of_Release` sebagai prediktor.

---

