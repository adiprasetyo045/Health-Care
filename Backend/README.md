# 🏥 Diabetes Prediction System - Backend

Sistem pendukung keputusan (DSS) berbasis Machine Learning untuk memprediksi risiko diabetes. Dibangun menggunakan **Flask**, **Scikit-Learn**, dan **Decision Tree Algorithm** dengan optimasi kalibrasi probabilitas.

## 🚀 Fitur Utama

- **Decision Tree Classifier**: Menggunakan Entropy & CalibratedClassifierCV untuk probabilitas yang akurat.
- **Data Balancing**: Implementasi **SMOTE** untuk menangani ketidakseimbangan kelas data.
- **Robust Preprocessing**: Konversi otomatis data kategori (teks) ke numerik.
- **RESTful API**: Endpoint JSON untuk integrasi Frontend/Mobile.
- **Prediction Logging**: Menyimpan riwayat prediksi ke CSV untuk audit.
- **Admin Dashboard**: Visualisasi performa model dan riwayat pasien.

## 📂 Struktur Proyek

```text
Diabetes-Detector/
├── Backend/                 # Source Code Utama
│   ├── config.py            # Konfigurasi Global
│   ├── app.py               # Flask App Factory
│   ├── data/                # Dataset & Logs
│   ├── models/              # Model Logic & Pickle
│   ├── routes/              # API & Web Routes
│   ├── static/              # CSS/JS Assets
│   └── templates/           # HTML Views
├── Scripts/                 # Utilitas & Training
│   ├── check_dataset.py     # Cek Integritas Data
│   ├── balance_dataset.py   # SMOTE Balancing
│   ├── train_model.py       # Training Model
│   ├── debug_algo.py        # Debugging Manual
│   └── fix_prediction.py    # Self-Healing Tool
├── run_app.py               # Entry Point Server
├── requirements.txt         # Dependencies
└── README.md                # Dokumentasi