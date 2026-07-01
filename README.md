# Stack Overflow AI Tools ML Project

Project ini membuat baseline machine learning untuk penelitian:

**Analisis dan Segmentasi Developer serta Prediksi Penggunaan AI Tools Menggunakan Machine Learning pada Dataset Stack Overflow Developer Survey 2025.**

## Tujuan

- Melakukan preprocessing dataset Stack Overflow Developer Survey 2025.
- Membuat target `AI_Usage` dari kolom `AISelect`.
- **Unsupervised Learning**: Segmentasi developer dengan **K-Means** (tradisional) dan **GMM** (modern probabilistic).
- **Supervised Learning**: Prediksi penggunaan AI tools dengan **Random Forest** & **Linear SVM** (tradisional) serta **XGBoost** & **MLP Neural Network** (modern).
- Menyimpan tabel, grafik, dan ringkasan hasil untuk bahan BAB IV laporan.

## Struktur

```text
.
├── DATASET.md
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── .vscode/
│   └── settings.json          # VS Code Python path config
├── notebooks/
│   └── 01_model_stackoverflow_ai_tools.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Load, target, clean, feature engineering
│   ├── train_models.py        # K-Means, GMM, RF, SVM, XGBoost, MLP
│   ├── evaluate_models.py     # Metrics, plots, tables, reports
│   └── run_pipeline.py        # Pipeline orchestrator
├── outputs/
│   ├── figures/               # *.png (elbow, silhouette, BIC/AIC, CM, dll)
│   ├── tables/                # *.csv (metrics, CM, classification report)
│   └── reports/               # bab_iv_ringkasan_awal.md
└── data/
    └── survey_2025.csv
```

## Dataset

Dataset lokal: `data/survey_2025.csv` (Stack Overflow Developer Survey 2025).
Pipeline mencari secara otomatis di beberapa lokasi fallback.

## Alur Pipeline

```
load CSV → inspect columns → create target AI_Usage → select features → build preprocessor
    │
    ├── [Unsupervised] K-Means (k=2..10) → elbow + silhouette → k terbaik → cluster summary
    ├── [Unsupervised] GMM (k=2..8) → BIC/AIC + silhouette → k terbaik → cluster summary
    │
    └── [Supervised] Train/Test split (80:20)
        ├── Random Forest (traditional ensemble)
        ├── Linear SVM (traditional linear classifier)
        ├── XGBoost (modern gradient boosting)
        └── MLP (modern neural network)
            ↓
        Evaluasi: accuracy, precision, recall, F1, confusion matrix
        → comparison table → visualisasi
```

## Model yang Digunakan

### Unsupervised Learning (Segmentasi Developer)

| Model | Kategori | Keterangan |
|-------|----------|------------|
| **K-Means** | Tradisional | Hard clustering, inertia + silhouette untuk pilih k |
| **GMM** | Modern | Soft clustering (probabilistic), BIC/AIC + silhouette untuk pilih k |

### Supervised Learning (Klasifikasi AI_Usage)

| Model | Kategori | Keterangan |
|-------|----------|------------|
| **Random Forest** | Tradisional | 200 trees, class_weight balanced |
| **Linear SVM** | Tradisional | Linear kernel, class_weight balanced |
| **XGBoost** | Modern | Gradient boosting, scale_pos_weight untuk imbalance |
| **MLP** | Modern | Neural network 2 hidden layer (100,50), early stopping |

## Preprocessing

- Missing value numerik → median, kategorikal → "Unknown"
- `YearsCode` dikonversi ke numerik (`"Less than 1 year"` → 0, `"More than 50 years"` → 51)
- Fitur kategorikal → One-Hot Encoding
- Fitur multi-select (;) → CountVectorizer binary (min_df=20)
- Fitur numerik → StandardScaler
- **Data leakage dihindari**: semua kolom AI (`AISelect`, `AISent`, `AIAcc`, `AITool*`, `AIModels*`, dll) tidak dipakai sebagai fitur

## Output

Semua hasil tersimpan di `outputs/`:

| Folder | Isi |
|--------|-----|
| `figures/` | elbow method, silhouette, BIC/AIC, distribusi cluster, confusion matrix, perbandingan model |
| `tables/` | ketersediaan kolom, distribusi target, ringkasan cluster (K-Means + GMM), CM, classification report, perbandingan performa |
| `reports/` | `bab_iv_ringkasan_awal.md` — ringkasan untuk laporan |

## Setup

1. Clone repository.
2. Buat virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Aktifkan:
   ```bash
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
4. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
5. Letakkan dataset `survey_2025.csv` di `data/`.

## Cara Menjalankan

Via notebook:
```bash
jupyter notebook
# Buka: notebooks/01_model_stackoverflow_ai_tools.ipynb
```

Atau via terminal:
```bash
python src/run_pipeline.py
```

## Catatan

- Dataset mentah dan file output hasil training tidak disimpan di repository (lihat `.gitignore`).
- VS Code users: `.vscode/settings.json` sudah dikonfigurasi agar LSP bisa meresolve import dari folder `src/`.
