# PROJECT CONTEXT

## Judul Penelitian

Analisis dan Segmentasi Developer serta Prediksi Penggunaan AI Tools Menggunakan Machine Learning pada Dataset Stack Overflow Developer Survey 2024–2025

## Latar Belakang Singkat

Penelitian ini menggunakan dataset Stack Overflow Developer Survey untuk menganalisis karakteristik developer, melakukan segmentasi developer, dan memprediksi penggunaan AI Tools oleh developer menggunakan pendekatan machine learning.

Fokus penelitian dipersempit dari “prediksi penggunaan teknologi” menjadi “prediksi penggunaan AI Tools” agar scope penelitian lebih jelas, realistis, dan mudah diimplementasikan.

## Tujuan Project Machine Learning

Project ini bertujuan untuk:

1. Melakukan eksplorasi dan analisis awal terhadap data developer.
2. Melakukan segmentasi developer menggunakan K-Means Clustering.
3. Membangun model klasifikasi untuk memprediksi apakah developer menggunakan AI Tools atau tidak.
4. Membandingkan performa model Random Forest dan Support Vector Machine.
5. Menghasilkan output yang dapat digunakan untuk laporan penelitian BAB IV.

## Dataset

Dataset yang digunakan adalah Stack Overflow Developer Survey tahun 2024 dan 2025.

Untuk tahap awal pengembangan model, gunakan dataset 2025 terlebih dahulu agar pipeline machine learning stabil. Setelah pipeline berhasil, dataset 2024 dapat digunakan sebagai pembanding atau digabungkan hanya jika struktur kolomnya kompatibel.

Contoh file dataset:

```text
survey_results_public_2025.csv
survey_results_public_2024.csv
```

Jika nama file berbeda, sesuaikan dengan file yang tersedia di folder project.

## Fokus Penelitian

Fokus utama project ini adalah:

```text
Prediksi penggunaan AI Tools oleh developer.
```

Target supervised learning:

```text
AI_Usage
```

Nilai target:

```text
1 = Developer menggunakan AI Tools
0 = Developer tidak menggunakan AI Tools
```

Target dapat dibuat dari kolom yang berkaitan dengan penggunaan AI Tools, misalnya:

```text
AISelect
```

Namun, nama kolom harus dicek langsung dari dataset karena struktur kolom bisa berbeda antar tahun.

## Model / Algoritma yang Digunakan

Project ini wajib menggunakan minimal 3 model atau algoritma:

| Model                        | Jenis                 | Tujuan                                                    |
| ---------------------------- | --------------------- | --------------------------------------------------------- |
| K-Means Clustering           | Unsupervised Learning | Segmentasi developer                                      |
| Random Forest                | Supervised Learning   | Prediksi penggunaan AI Tools                              |
| Support Vector Machine / SVM | Supervised Learning   | Prediksi penggunaan AI Tools dan pembanding Random Forest |

Catatan:

K-Means tidak dibandingkan langsung dengan Random Forest dan SVM karena jenis tugasnya berbeda. K-Means digunakan untuk segmentasi, sedangkan Random Forest dan SVM digunakan untuk klasifikasi.

## Fitur yang Disarankan

Gunakan fitur yang berkaitan dengan karakteristik developer dan teknologi yang digunakan.

Fitur kandidat:

```text
Age
Country
EdLevel
DevType
Employment
RemoteWork
YearsCode
YearsCodePro
LanguageHaveWorkedWith
DatabaseHaveWorkedWith
PlatformHaveWorkedWith
WebframeHaveWorkedWith
ToolsTechHaveWorkedWith
```

Jika beberapa kolom tidak tersedia, gunakan hanya kolom yang tersedia di dataset.

## Aturan Penting: Hindari Data Leakage

Jangan gunakan kolom yang terlalu dekat dengan target sebagai fitur input.

Jika target dibuat dari kolom AISelect, maka kolom berikut sebaiknya tidak dipakai sebagai fitur:

```text
AISelect
AISent
AIAcc
AIBen
AIToolCurrently Using
AIToolInterested in Using
AIToolNot interested in Using
```

Tujuannya agar model benar-benar memprediksi berdasarkan karakteristik developer, bukan karena jawaban target sudah bocor ke fitur input.

## Preprocessing yang Dibutuhkan

Lakukan preprocessing berikut:

1. Load dataset menggunakan pandas.
2. Cek nama kolom yang tersedia.
3. Pilih fitur yang relevan.
4. Buat target `AI_Usage`.
5. Hapus data yang targetnya kosong.
6. Bersihkan kolom numerik seperti `YearsCode` dan `YearsCodePro`.
7. Tangani nilai seperti:

   * `Less than 1 year` menjadi `0`
   * `More than 50 years` menjadi `51`
8. Tangani missing value.
9. Encode fitur kategorikal menggunakan One-Hot Encoding.
10. Scaling fitur numerik menggunakan StandardScaler.
11. Split data menjadi training dan testing untuk supervised learning.

## Evaluasi Model

Untuk Random Forest dan SVM, gunakan metrik:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
```

Untuk K-Means, gunakan:

```text
Elbow Method
Silhouette Score
Distribusi jumlah data per cluster
Interpretasi karakteristik cluster
```

## Output yang Diharapkan

Project harus menghasilkan:

1. Notebook atau script Python yang rapi.
2. Hasil EDA sederhana.
3. Hasil segmentasi K-Means.
4. Hasil evaluasi Random Forest.
5. Hasil evaluasi SVM.
6. Tabel perbandingan Random Forest dan SVM.
7. Visualisasi sederhana untuk mendukung laporan.
8. Ringkasan hasil yang bisa digunakan untuk BAB IV.

## Struktur Folder yang Disarankan

```text
project-ml-stackoverflow/
│
├── data/
│   ├── survey_results_public_2025.csv
│   └── survey_results_public_2024.csv
│
├── notebooks/
│   └── 01_model_stackoverflow_ai_tools.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── requirements.txt
└── README.md
```

Untuk tahap awal, cukup gunakan satu notebook terlebih dahulu agar proses eksplorasi lebih mudah.

## Library yang Disarankan

Gunakan Python dengan library berikut:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Jika menggunakan notebook:

```text
jupyter
ipykernel
```

## Prinsip Implementasi

1. Jangan hardcode terlalu banyak nama kolom tanpa validasi.
2. Selalu cek apakah kolom tersedia sebelum digunakan.
3. Jika kolom tidak ditemukan, tampilkan daftar kolom yang mirip.
4. Buat kode yang mudah dijelaskan di laporan.
5. Prioritaskan model yang berjalan stabil daripada model yang terlalu kompleks.
6. Jangan melakukan tuning berlebihan sebelum baseline model berhasil.
7. Gunakan random_state agar hasil dapat direproduksi.
8. Simpan hasil evaluasi model dalam bentuk tabel.
9. Simpan visualisasi ke folder outputs/figures.
10. Tulis komentar secukupnya agar kode mudah dipahami.

## Catatan untuk Laporan

Nanti hasil project ini akan digunakan untuk BAB IV, terutama bagian:

1. Deskripsi dataset.
2. Proses preprocessing.
3. Hasil K-Means Clustering.
4. Evaluasi Random Forest.
5. Evaluasi SVM.
6. Perbandingan performa model.
7. Interpretasi hasil.
