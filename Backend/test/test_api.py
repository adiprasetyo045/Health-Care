"""
Backend/test/test_api.py
Script untuk menguji fungsi internal Backend (Validation, Model, Logging)
tanpa perlu menjalankan server Flask.
"""

import sys
import os
from pathlib import Path

# 1. Setup Path Project (Robust Implementation)
# Mencari root project berdasarkan keberadaan folder 'Backend'
current_file = Path(__file__).resolve()
# Naik dua level dari Backend/test/ ke Root Project
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from Backend.config import Config
    from Backend.models.decision_tree_model import DiabetesModel
    from Backend.models.utils import validate_input_data, log_prediction
    print("✅ Module imports successful")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current Sys Path: {sys.path[0]}")
    sys.exit(1)

def run_api_tests():
    print("=" * 70)
    print("🧪 API INTERNAL TESTS - PRODUCTION CHECK")
    print(f"📂 Project Root: {project_root}")
    print(f"📅 Timestamp: {Config.get_system_status().get('timestamp', 'N/A') if hasattr(Config, 'get_system_status') else 'N/A'}")
    print("=" * 70)

    # Data Dummy untuk Test (Representasi Input Form)
    sample_data = {
        "age": 45,
        "gender": "Male",
        "pulse_rate": 72,
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "glucose": 150,    # mg/dL
        "height": 170,     # cm
        "weight": 70,      # kg
        "bmi": 0,          # Akan dikalkulasi otomatis oleh model/preprocess
        "family_diabetes": "Yes",
        "hypertensive": "No",
        "family_hypertension": "No",
        "cardiovascular_disease": "No",
        "stroke": "No"
    }

    # --------------------------------------------------
    # 1. Validation Test
    # --------------------------------------------------
    print("\n[1] Testing Input Validation")
    validation = validate_input_data(sample_data)
    if validation.get('is_valid'):
        print("   ✅ Input Valid: Structure and data types approved.")
    else:
        print(f"   ❌ Input Invalid: {validation.get('errors')}")
        # Jangan stop di sini agar bisa menguji error handling di tahap berikutnya

    # --------------------------------------------------
    # 2. Model Integrity Check
    # --------------------------------------------------
    print("\n[2] Checking Model Files")
    model_exists = os.path.exists(Config.MODEL_PATH)
    meta_exists = os.path.exists(Config.META_PATH)
    
    if model_exists and meta_exists:
        print(f"   ✅ Model Bundle: FOUND ({os.path.basename(Config.MODEL_PATH)})")
        print(f"   ✅ Metadata: FOUND ({os.path.basename(Config.META_PATH)})")
    else:
        if not model_exists: print(f"   ❌ Model file missing at: {Config.MODEL_PATH}")
        if not meta_exists: print(f"   ❌ Metadata file missing at: {Config.META_PATH}")
        print("      HINT: Run 'python Scripts/train_model.py' to generate files.")

    # --------------------------------------------------
    # 3. Prediction Pipeline Test
    # --------------------------------------------------
    print("\n[3] Testing Prediction Pipeline")
    prediction_result = None
    try:
        # Menggunakan Singleton Model Instance
        model_instance = DiabetesModel.get_instance()
        prediction_result = model_instance.predict(sample_data)
        
        if prediction_result.get('success'):
            print("   ✅ Prediction Successful")
            print(f"      - Diagnosis   : {prediction_result['label']}")
            print(f"      - Probability : {prediction_result['probability_percent']}%")
            print(f"      - Risk Level  : {prediction_result['risk_level']}")
            
            # Verifikasi Auto-calculation (BMI & Glucose Unit)
            processed_bmi = prediction_result.get('input_data', {}).get('bmi')
            print(f"      - Processed BMI: {processed_bmi} (Verified)")
        else:
            print(f"   ❌ Prediction Failed: {prediction_result.get('error')}")
            
    except Exception as e:
        print(f"   ❌ Runtime Error during prediction: {e}")

    # --------------------------------------------------
    # 4. Logging System Test
    # --------------------------------------------------
    print("\n[4] Testing Logging System")
    if prediction_result and prediction_result.get('success'):
        try:
            # Memastikan folder logs tersedia sebelum menulis
            os.makedirs(os.path.dirname(Config.PREDICTION_LOG), exist_ok=True)
            
            log_prediction(sample_data, prediction_result)
            
            if os.path.exists(Config.PREDICTION_LOG):
                print(f"   ✅ Log saved successfully: {Config.PREDICTION_LOG}")
            else:
                print("   ❌ Log file was not created unexpectedly.")
        except Exception as e:
            print(f"   ❌ Logging Failed: {e}")
    else:
        print("   ⚠️  Skipping Log Test: Prediction was unsuccessful.")

    print("\n" + "=" * 70)
    print("✅ ALL INTERNAL TESTS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    run_api_tests()