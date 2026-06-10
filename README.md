# 🩺 Diabetes Prediction Analysis

> Proyek Machine Learning profesional yang memprediksi risiko diabetes menggunakan data medis pasien Pima Indians. Dibangun dengan Python, Scikit-Learn, dan teknik Ensemble Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 🖼️ Visualisasi

| Heatmap Korelasi | Confusion Matrix |
|---|---|
| ![heatmap](Output%20Image/Heatmap%20Korelasi.png) | ![matrix](Output%20Image/Confusion%20Matrix%20DS.png) |

---

## ⚡ Hasil Utama

- 🎯 **Akurasi Terbaik:** 79.22%
- 🌲 **Model Terbaik:** Decision Tree Classifier (`max_depth=5`)
- 🔥 **Fitur Paling Berpengaruh:** Glucose, BMI, Age
- ⚖️ **Jumlah Data:** 768 sampel pasien
- 🗳️ **Metode Ensemble:** Majority Voting (5 model)

---

## 📌 Ringkasan Proyek

Proyek ini menggunakan **Pima Indians Diabetes Dataset** untuk memprediksi apakah seorang pasien berisiko terkena **diabetes** atau **sehat** berdasarkan data medis (glukosa, tekanan darah, BMI, insulin, dll.).

Tujuan utama proyek ini adalah mendemonstrasikan alur kerja Machine Learning end-to-end yang bersih — mulai dari eksplorasi data mentah hingga prediksi menggunakan teknik Ensemble (Majority Voting).

---

## 🌍 Dampak di Dunia Nyata

Model ini mendemonstrasikan bagaimana Machine Learning dapat mendukung:
- ⚕️ Deteksi dini risiko diabetes pada pasien
- 💰 Pengurangan biaya pemeriksaan laboratorium berulang
- 📊 Pengambilan keputusan berbasis data di industri kesehatan
- 🏥 Skrining massal yang lebih efisien di fasilitas kesehatan

---

## 📈 Pipeline Eksekusi & Hasil

Menjalankan seluruh pipeline menghasilkan output komprehensif meliputi pemrosesan data, pelatihan model, evaluasi, dan prediksi langsung. Berikut adalah rincian lengkap hasil proyek:

### 1. Import Library & Load Dataset
- **Library Utama**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (KNeighborsClassifier, DecisionTreeClassifier, VotingClassifier).
- **Dataset**: Memuat `diabetes.csv` yang berisi **768 baris** (pasien) dan 9 kolom (8 fitur medis + 1 target `Outcome`).

### 2. Exploratory Data Analysis (EDA)
- 🌡️ **Heatmap Korelasi**: Memetakan hubungan linear antar seluruh fitur medis. Fitur `Glucose` memiliki korelasi tertinggi (0.47) terhadap `Outcome` (diabetes), diikuti oleh `BMI` (0.29), `Age` (0.24), dan `Pregnancies` (0.22).

### 3. Preprocessing Data (`StandardScaler`)
- **Pembagian Fitur & Target**: 8 kolom fitur (`x`) dan 1 kolom target `Outcome` (`y`).
- **Data Splitting**: 80% data train (**614 sampel**) dan 20% data test (**154 sampel**) dengan `random_state=42`.
- **Feature Scaling**: Menggunakan `StandardScaler()` untuk menstandarisasi skala seluruh fitur medis agar performa KNN optimal.

### 4. Training KNN — 4 Distance Metrics × 3 Variasi K
Proyek ini mengevaluasi **KNN** secara menyeluruh dengan 4 metrik jarak dan 3 nilai K:

| Metrik Jarak | k=3 | k=5 | k=7 | Best K | Best Akurasi |
|---|---|---|---|---|---|
| **Euclidean** | 70.78% | 69.48% | 68.18% | k=3 | **70.78%** |
| **Manhattan** | 67.53% | 66.23% | 70.78% | k=7 | **70.78%** |
| **Chebyshev** | 61.04% | 68.83% | 67.53% | k=5 | **68.83%** |
| **Minkowski** (p=3) | 68.83% | 70.78% | 67.53% | k=5 | **70.78%** |

### 5. Training Decision Tree
- **Decision Tree Classifier** dengan `max_depth=5` mencapai akurasi **79.22%** ✅ — jauh melampaui semua varian KNN.
- Model ini dipilih sebagai **model terbaik** karena kemampuannya menangkap hubungan non-linear dalam data medis.

### 6. Ensemble — Majority Voting
- Menggabungkan **5 model** (4 KNN terbaik + Decision Tree) menggunakan `VotingClassifier` dengan strategi `hard voting`.
- Akurasi ensemble: **70.13%** — lebih rendah dari Decision Tree tunggal karena dominasi model KNN yang kurang akurat.

### 7. Perbandingan Model (Ranking Akurasi)

| Ranking | Model | Akurasi |
|---|---|---|
| 🥇 1 | **Decision Tree** | **79.22%** |
| 🥈 2 | KNN Euclidean | 70.78% |
| 🥉 3 | KNN Manhattan | 70.78% |
| 4 | KNN Minkowski | 70.78% |
| 5 | Ensemble (Gabungan) | 70.13% |
| 6 | KNN Chebyshev | 68.83% |

### 8. Evaluasi — Confusion Matrix (Decision Tree)
Berdasarkan model terbaik (Decision Tree) pada 154 data test:

|  | Prediksi: Sehat (0) | Prediksi: Diabetes (1) |
|---|---|---|
| **Aktual: Sehat (0)** | ✅ 87 (True Negative) | ❌ 12 (False Positive) |
| **Aktual: Diabetes (1)** | ❌ 20 (False Negative) | ✅ 35 (True Positive) |

- **Sensitivity (Recall):** 35 / (35 + 20) = **63.64%**
- **Specificity:** 87 / (87 + 12) = **87.88%**
- **Precision:** 35 / (35 + 12) = **74.47%**

### 9. Simulasi — Prediksi Pasien Baru
- **Data Input**: `[Pregnancies=2, Glucose=150, BP=75, ST=20, Insulin=100, BMI=33, DPF=0.5, Age=40]`
- **Hasil Voting**: Seluruh **5 algoritma** (4 KNN + Decision Tree) memprediksi **Diabetes (1)**.
- **Kesimpulan**: Pasien dinyatakan harus segera mendapat perawatan medis.

---

### 🎯 Cakupan Proyek Ini
- Exploratory Data Analysis (EDA) dengan heatmap korelasi
- Analisis korelasi dan distribusi fitur medis
- Klasifikasi biner (Sehat / Diabetes)
- Perbandingan model: KNN (4 distance metrics) vs Decision Tree
- Ensemble Learning dengan Majority Voting
- Evaluasi model dengan akurasi, confusion matrix, dan simulasi pasien baru

---

## 📊 Dataset Overview & Konteks

Proyek ini menggunakan dataset publik **Pima Indians Diabetes**, yang berisi data diagnostik medis dari perempuan keturunan Pima Indian berusia ≥ 21 tahun.

- **Sumber:** [UCI Machine Learning Repository — Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) (Smith et al., 1988).
- **Jumlah Data:** **768** sampel pasien dengan 9 kolom.
- **Integritas Data:** Dataset bersih tanpa nilai null (beberapa fitur memiliki nilai 0 yang merepresentasikan missing values).

Berikut adalah deskripsi kolom fitur medis yang memprediksi outcome diabetes:

| Fitur | Deskripsi |
|---|---|
| `Pregnancies` | Jumlah kehamilan |
| `Glucose` | Konsentrasi glukosa plasma 2 jam (tes toleransi glukosa oral) |
| `BloodPressure` | Tekanan darah diastolik (mm Hg) |
| `SkinThickness` | Ketebalan lipatan kulit trisep (mm) |
| `Insulin` | Insulin serum 2 jam (mu U/ml) |
| `BMI` | Body Mass Index (berat badan dalam kg / tinggi² dalam m) |
| `DiabetesPedigreeFunction` | Fungsi silsilah diabetes (riwayat genetik) |
| `Age` | Usia pasien (tahun) |
| `Outcome` | Target kelas: **0** = Sehat, **1** = Diabetes |

---

## ⚙️ Setup & Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/rapfii/diabetes-prediction-analysis
cd diabetes-prediction-analysis
```

### 2. Buat Virtual Environment *(Disarankan)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn jupyter
```

---

## 🚀 Cara Menjalankan

### Jalankan Notebook Jupyter
```bash
jupyter notebook "Diabetes_Prediction_Analysis.ipynb.ipynb"
```

Buka browser dan jalankan seluruh cell secara berurutan. Notebook akan:
1. Memuat dan mengeksplorasi dataset
2. Membuat heatmap korelasi fitur
3. Melatih 4 varian KNN + Decision Tree
4. Menjalankan Ensemble (Majority Voting)
5. Menampilkan perbandingan model & confusion matrix
6. Melakukan simulasi prediksi pasien baru

---

## 🧠 ML Workflow

```
Raw Data → EDA (Heatmap Korelasi) → Feature/Target Split
    → Train/Test Split (80/20) → Feature Scaling (StandardScaler)
    → Training KNN (4 Metrics × 3 K) → Training Decision Tree
    → Ensemble Voting → Evaluasi (Confusion Matrix)
    → Simulasi Prediksi Pasien Baru
```

---

## 🗂️ Struktur Proyek

```
Source Code & Dataset/
│
├── diabetes.csv                                ← Dataset Pima Indians Diabetes (768 sampel)
│
├── Diabetes_Prediction_Analysis.ipynb.ipynb     ← Notebook utama: EDA, training, evaluasi & prediksi
│
├── Output Image/
│   ├── Heatmap Korelasi.png                    ← Heatmap korelasi antar fitur medis
│   └── Confusion Matrix DS.png                 ← Confusion matrix Decision Tree (model terbaik)
│
└── README.md                                   ← Dokumentasi proyek (anda di sini!)
```

---

## 🛠️ Teknologi yang Digunakan

| Tool | Kegunaan |
|---|---|
| `pandas` | Pemuatan dan manipulasi data |
| `matplotlib` + `seaborn` | Visualisasi data (heatmap, confusion matrix) |
| `scikit-learn` — `KNeighborsClassifier` | Model KNN dengan 4 distance metrics |
| `scikit-learn` — `DecisionTreeClassifier` | Model Decision Tree |
| `scikit-learn` — `VotingClassifier` | Ensemble Majority Voting |
| `scikit-learn` — `StandardScaler` | Standarisasi fitur |
| `scikit-learn` — `accuracy_score`, `confusion_matrix` | Evaluasi model |
| `Jupyter Notebook` | Lingkungan pengembangan interaktif |

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** — bebas digunakan, dimodifikasi, dan dibagikan.

---

## 🙋 Author

**Rapfi**
- GitHub: https://github.com/rapfii
- LinkedIn: https://www.linkedin.com/in/raffi-khairan-hidayat

---

> ⭐ Jika proyek ini bermanfaat, berikan bintang di repository ini!
