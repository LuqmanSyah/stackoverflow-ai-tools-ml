# Dataset

Dataset yang digunakan adalah Stack Overflow Developer Survey.

File dataset tidak disertakan di repository GitHub karena ukurannya besar dan berasal dari sumber eksternal. Untuk menjalankan project, letakkan file dataset 2025 pada salah satu lokasi berikut:

```text
Survey/packages/archive/2025/results.csv
data/survey_results_public_2025.csv
survey_results_public_2025.csv
```

Pipeline akan mencari dataset secara otomatis menggunakan urutan lokasi tersebut. Nama file yang digunakan adalah `survey_{tahun}.csv` (contoh: `survey_2025.csv`).

Untuk dataset lokal saat project dibuat:

- File utama 2025: `Survey/packages/archive/2025/results.csv`
- Schema 2025: `Survey/packages/archive/2025/schema.csv`
- Target: `AISelect`

Mapping target:

- `AI_Usage = 1` untuk jawaban `AISelect` yang dimulai dengan `Yes`
- `AI_Usage = 0` untuk jawaban `AISelect` yang dimulai dengan `No`
- Jawaban kosong atau tidak sesuai mapping dibuang sebelum modeling
