import os
import json

class Config:
    # --- 1. BASE DIRECTORIES ---
    # Absolute path folder 'Backend'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Root Project (Satu level di atas Backend/)
    ROOT_DIR = os.path.dirname(BASE_DIR)
    
    # Lokasi Folder Utama (Dibuat Dinamis)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Path untuk Flask (Templates & Static)
    TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
    STATIC_DIR = os.path.join(BASE_DIR, "static")

    # --- 2. FILE PATHS ---
    RAW_DATA = os.path.join(DATA_DIR, "diabetes.csv")
    BALANCED_DATA = os.path.join(DATA_DIR, "diabetes_balanced.csv")
    PREDICTION_LOG = os.path.join(LOGS_DIR, "prediction_logs.csv")
    
    # Resource Model & Metadata
    MODEL_PATH = os.path.join(MODELS_DIR, "decision_tree_bundle.pkl")
    META_PATH = os.path.join(MODELS_DIR, "decision_tree_meta.json")
    
    # Laporan Teknis
    DATA_REPORT = os.path.join(DATA_DIR, "dataset_report.txt")
    BALANCE_REPORT = os.path.join(DATA_DIR, "balancing_report.txt")
    TRAINING_REPORT = os.path.join(DATA_DIR, "training_report.txt")

    # --- 3. DATA DEFINITIONS ---
    # Harus sesuai urutan kolom saat training
    FEATURES = [
        'age', 'gender', 'pulse_rate', 'systolic_bp', 'diastolic_bp',
        'glucose', 'height', 'weight', 'bmi', 'family_diabetes',
        'hypertensive', 'family_hypertension', 'cardiovascular_disease', 'stroke'
    ]

    # Deskripsi untuk UI
    FEATURE_DESCRIPTIONS = {
        'age': 'Usia (Tahun)',
        'gender': 'Jenis Kelamin',
        'pulse_rate': 'Nadi (bpm)',
        'systolic_bp': 'Sistolik',
        'diastolic_bp': 'Diastolik',
        'glucose': 'Gula Darah (mmol/L)',
        'height': 'Tinggi (m)',
        'weight': 'Berat (kg)',
        'bmi': 'BMI',
        'family_diabetes': 'Riwayat Diabetes Keluarga',
        'hypertensive': 'Hipertensi',
        'family_hypertension': 'Riwayat Hipertensi Keluarga',
        'cardiovascular_disease': 'Penyakit Jantung',
        'stroke': 'Riwayat Stroke'
    }

    # Ambang Batas Klinis (WHO Standard)
    THRESHOLDS = {
        'glucose': {
            'normal': {'min': 4.0, 'max': 5.6},
            'prediabetes': {'min': 5.7, 'max': 6.9},
            'diabetes': {'min': 7.0, 'max': 99.0}
        },
        'bmi': {
            'normal': {'min': 18.5, 'max': 24.9},
            'overweight': {'min': 25.0, 'max': 29.9},
            'obese': {'min': 30.0, 'max': 99.0}
        }
    }

    # --- 4. AUTO-CREATE DIRECTORIES ---
    @classmethod
    def init_app(cls):
        """Membuat folder-folder penting jika belum ada."""
        folders = [cls.MODELS_DIR, cls.LOGS_DIR, cls.DATA_DIR]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

    # --- 5. SERVER CONFIG ---
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8000
    DEBUG = True

# Menjalankan inisialisasi folder saat modul di-import
Config.init_app()

def get_system_status():
    """Fungsi utilitas untuk memeriksa status file model dan log."""
    status = {
        "directories": {
            "data": os.path.exists(Config.DATA_DIR),
            "models": os.path.exists(Config.MODELS_DIR),
            "logs": os.path.exists(Config.LOGS_DIR)
        },
        "files": {
            "model_ready": os.path.exists(Config.MODEL_PATH),
            "meta_ready": os.path.exists(Config.META_PATH),
            "dataset_exists": os.path.exists(Config.RAW_DATA)
        },
        "model_summary": None
    }

    if status["files"]["meta_ready"]:
        try:
            with open(Config.META_PATH, 'r') as f:
                meta = json.load(f)
                status["model_summary"] = {
                    "accuracy": f"{meta.get('accuracy_cv', 0) * 100:.2f}%",
                    "date": meta.get("training_date", "Unknown"),
                    "algorithm": meta.get("algorithm", "Decision Tree")
                }
        except:
            status["model_summary"] = "Error reading meta"

    return status

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🏥 DIABETES SYSTEM CONFIGURATION")
    print("="*40)
    
    st = get_system_status()
    print(f"\n📁 STATUS FOLDER:")
    for k, v in st["directories"].items():
        print(f"   [{'✅' if v else '❌'}] {k.upper()}")

    print(f"\n📄 STATUS FILE:")
    for k, v in st["files"].items():
        print(f"   [{'✅' if v else '❌'}] {k.replace('_', ' ').upper()}")

    if st["model_summary"]:
        print(f"\n🤖 MODEL INFO: {st['model_summary']}")
    else:
        print(f"\n⚠️  MODEL INFO: Belum dilatih (pkl/json tidak ditemukan)")
    print("\n" + "="*40)