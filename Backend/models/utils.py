import csv
import os
from datetime import datetime
from typing import Dict, Any

# Mengimpor Config dan Preprocessor
from Backend.config import Config
from Backend.models.preprocess import DiabetesPreprocessor

# Inisialisasi Preprocessor sekali saja untuk referensi urutan fitur
_preprocessor = DiabetesPreprocessor()
REQUIRED_FEATURES = _preprocessor.feature_order

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi kelengkapan data input API.
    Hanya memastikan field wajib ada dan tidak None/Kosong.
    Konversi tipe data dilakukan di preprocess.py.
    """
    errors = []
    
    for feature in REQUIRED_FEATURES:
        # 1. Cek keberadaan key
        if feature not in data:
            errors.append(f"Missing field: {feature}")
            continue
        
        val = data[feature]
        
        # 2. Cek nilai None atau String Kosong
        # Catatan: Angka 0 dianggap valid (tidak None), jadi aman.
        if val is None:
            errors.append(f"Value cannot be None for: {feature}")
        elif isinstance(val, str) and str(val).strip() == "":
            errors.append(f"Empty value for field: {feature}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

def log_prediction(input_data: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Menyimpan log prediksi ke CSV (Audit Trail).
    Menggunakan path dari Config agar sinkron dengan Dashboard.
    """
    # KOREKSI: Gunakan Config.PREDICTION_LOG agar masuk ke folder 'logs/'
    log_path = Config.PREDICTION_LOG
    
    # Pastikan folder logs ada
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Siapkan data baris dengan URUTAN FITUR YANG KONSISTEN
    row_data = []
    
    # 1. Fitur Input (Pastikan urutan sesuai kolom training)
    for feature in REQUIRED_FEATURES:
        row_data.append(input_data.get(feature, ""))

    # 2. Hasil Prediksi
    # Mengambil data dari result dictionary
    row_data.extend([
        result.get("label", "Unknown"),
        result.get("probability_percent", 0.0), # Sesuaikan dengan key di api_routes ('probability_percent')
        result.get("risk_level", "Unknown"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

    # Header CSV
    header = REQUIRED_FEATURES + ["prediction_label", "probability_percent", "risk_level", "timestamp"]

    # Tulis ke CSV
    file_exists = os.path.exists(log_path)
    
    try:
        # PENTING: newline='' mencegah baris kosong ganda di Windows
        # encoding='utf-8' mencegah error karakter aneh
        with open(log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Tulis header hanya jika file baru dibuat
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)
    except Exception as e:
        # Jangan biarkan error logging menghentikan respons API utama
        print(f"⚠️ Warning: Gagal menulis log prediksi: {e}")