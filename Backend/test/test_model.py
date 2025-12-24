"""
Backend/test/test_model.py
Unit Test untuk Model Decision Tree (Tanpa API/Flask).
Fokus: Memastikan model bisa diload dan melakukan prediksi.
"""

import sys
import os
from pathlib import Path

# 1. Setup Path Project
# Agar bisa import module 'Backend' dari root project
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from Backend.config import Config
from Backend.models.decision_tree_model import DiabetesModel

def test_model():
    print("=" * 70)
    print("🧪 MODEL INTERNAL TEST SUITE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Import & Singleton Test
    # --------------------------------------------------
    print("\n[1] Initializing Model (Singleton Pattern)")
    try:
        # Menggunakan get_instance() bukan load_bundle manual
        model = DiabetesModel.get_instance()
        print("   ✅ Model Class Initialized")
        
        if model.model_bundle is not None:
             print("   ✅ Model Bundle Loaded into Memory")
        else:
             print("   ⚠️  Model Bundle is None (Belum ditraining?)")

    except Exception as e:
        print(f"   ❌ Import/Init failed: {e}")
        return

    # --------------------------------------------------
    # 2. File Artifacts Check
    # --------------------------------------------------
    print("\n[2] Checking Model Artifacts (via Config)")
    
    files_to_check = [
        ("Model Pickle", Config.MODEL_PATH),
        ("Metadata JSON", Config.META_PATH)
    ]

    for name, path in files_to_check:
        if path.exists():
            print(f"   ✅ {name} Found: {path.name}")
        else:
            print(f"   ❌ {name} MISSING: {path}")
            print("       (Jalankan 'python Scripts/train_model.py' dulu)")

    # --------------------------------------------------
    # 3. Prediction Sanity Test
    # --------------------------------------------------
    print("\n[3] Prediction Sanity Test")
    
    # Data test dengan campuran tipe data (String/Int) untuk menguji Preprocessor juga
    sample_data = {
        "age": 55,
        "gender": "Male",         # String (akan diproses preprocessor)
        "pulse_rate": 75,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "glucose": 200,           # Nilai tinggi (seharusnya Diabetic)
        "height": 170,
        "weight": 85,
        "bmi": 0,                 # 0 biar auto-calc
        "family_diabetes": 1,
        "hypertensive": "Yes",    # String
        "family_hypertension": 1,
        "cardiovascular_disease": 0,
        "stroke": 0,
    }

    try:
        # Prediksi menggunakan method class
        result = model.predict(sample_data)

        if result.get('success', True):
            print("   ✅ Prediction Execution OK")
            print("-" * 30)
            print(f"   Input Summary : {result.get('input_summary', 'N/A')}") # Jika ada fitur summary
            print(f"   Label         : {result['label']}")
            print(f"   Probability   : {result['probability_percent']}%")
            print(f"   Risk Level    : {result['risk_level']}")
            print(f"   Interpretation: {result['interpretation']}")
            print("-" * 30)
        else:
            print(f"   ❌ Model returned error: {result.get('error')}")

    except Exception as e:
        print(f"   ❌ Prediction Crash: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ MODEL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_model()