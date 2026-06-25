# Stack Overflow AI Tools ML Project

Project ini membuat baseline machine learning untuk penelitian:

Analisis dan Segmentasi Developer serta Prediksi Penggunaan AI Tools Menggunakan Machine Learning pada Dataset Stack Overflow Developer Survey 2024-2025.

## Tujuan

- Melakukan preprocessing dataset Stack Overflow Developer Survey 2025.
- Membuat target `AI_Usage` dari kolom `AISelect`.
- Melakukan segmentasi developer dengan K-Means Clustering.
- Memprediksi penggunaan AI tools dengan Random Forest dan Linear SVM.
- Menyimpan tabel, grafik, dan ringkasan hasil untuk bahan BAB IV laporan.

## Struktur

```text
.
├── DATASET.md
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_model_stackoverflow_ai_tools.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluate_models.py
│   ├── run_pipeline.py
│   └── train_models.py
└── outputs/
    ├── figures/
    ├── tables/
    └── reports/
```

Dataset lokal tidak dimasukkan ke GitHub. Lihat [DATASET.md](DATASET.md) untuk instruksi penempatan file dataset.

## Setup

1. Clone repository.

```bash
git clone <url-repository>
cd <nama-repository>
```

2. Buat virtual environment.

```bash
python -m venv .venv
```

3. Aktifkan virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependency.

```bash
pip install -r requirements.txt
```

5. Letakkan dataset 2025 pada salah satu path berikut.

```text
Survey/packages/archive/2025/results.csv
data/survey_results_public_2025.csv
survey_results_public_2025.csv
```

## Cara Menjalankan

Jalankan melalui notebook:

```bash
jupyter notebook
```

Buka:

```text
notebooks/
  01_model_stackoverflow_ai_tools.ipynb
```

Atau jalankan pipeline langsung dari terminal:

```bash
python src/run_pipeline.py
```

## Output

Setelah notebook atau pipeline dijalankan, hasil akan tersimpan di:

- `outputs/figures`: grafik elbow, silhouette, distribusi cluster, confusion matrix, dan perbandingan model.
- `outputs/tables`: tabel ketersediaan kolom, distribusi target, ringkasan cluster, confusion matrix, classification report, dan perbandingan performa model.
- `outputs/reports`: ringkasan awal BAB IV.

Folder output disimpan di repository hanya sebagai struktur kosong. File hasil eksekusi diabaikan oleh `.gitignore`.

## Metodologi Singkat

Target:

- `AI_Usage = 1`: developer menggunakan AI tools.
- `AI_Usage = 0`: developer tidak menggunakan AI tools.

Preprocessing:

- Missing value numerik diisi median.
- Missing value kategorikal diisi `Unknown`.
- `YearsCode` dan `YearsCodePro` dikonversi ke numerik jika tersedia.
- Fitur kategorikal di-encode dengan One-Hot Encoding.
- Fitur multi-select dipisahkan berdasarkan `;` dan diubah menjadi token biner.
- Fitur numerik di-scaling dengan StandardScaler.

Model:

- K-Means Clustering untuk segmentasi developer.
- Random Forest untuk prediksi penggunaan AI tools.
- Linear SVM untuk prediksi penggunaan AI tools.

Evaluasi:

- K-Means: Elbow Method, Silhouette Score, distribusi cluster, dan interpretasi karakteristik cluster.
- Klasifikasi: accuracy, precision, recall, F1-score, confusion matrix, dan classification report.

## Catatan Dataset 2025

Dari inspeksi awal dataset 2025:

- Target tersedia: `AISelect`.
- Fitur kandidat yang tersedia mencakup `Age`, `Country`, `EdLevel`, `DevType`, `Employment`, `RemoteWork`, `YearsCode`, `LanguageHaveWorkedWith`, `DatabaseHaveWorkedWith`, `PlatformHaveWorkedWith`, dan `WebframeHaveWorkedWith`.
- Fitur kandidat yang tidak tersedia pada dataset 2025 lokal: `YearsCodePro` dan `ToolsTechHaveWorkedWith`.
- Kolom AI tidak dipakai sebagai fitur input untuk menghindari data leakage.

## Catatan GitHub

File berikut sengaja tidak dimasukkan ke repository:

- Dataset mentah di `Survey/` atau `data/`.
- File output hasil training di `outputs/`.
- Cache Python seperti `__pycache__/`.
- Virtual environment seperti `.venv/`.
