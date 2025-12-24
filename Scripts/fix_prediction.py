"""
Scripts/fix_prediction.py
Diagnosa dan perbaikan otomatis untuk masalah prediksi/model.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 1. Setup Path Project
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# 2. Import Module
try:
    from Backend.config import Config
    from Backend.models.preprocess import DiabetesPreprocessor
except ModuleNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Module tidak ditemukan. {e}")
    sys.exit(1)

def fix_prediction_issues():
    print("=" * 60)
    print("🔧 DIAGNOSA & PERBAIKAN SISTEM PREDIKSI")
    print("=" * 60)
    
    issues = []
    fixes = []
    model = None
    
    try:
        # ==========================================
        # 1. CEK FILE MODEL (.pkl)
        # ==========================================
        print("\n1️⃣  CEK ARTIFACT MODEL")
        if not os.path.exists(Config.MODEL_PATH):
            issues.append(f"❌ File Model (.pkl) tidak ditemukan di: {Config.MODEL_PATH}")
            print("   ⚠️  File .pkl hilang.")
        else:
            print(f"   ✅ File ditemukan: {os.path.basename(Config.MODEL_PATH)}")
            try:
                # Load Model secara langsung
                bundle = joblib.load(Config.MODEL_PATH)
                
                if isinstance(bundle, dict) and 'model' in bundle:
                    model = bundle['model']
                    print("   ✅ Format Model: Dictionary Bundle (New)")
                else:
                    model = bundle
                    print("   ✅ Format Model: Direct Object (Old)")
                
                if not hasattr(model, 'predict'):
                    issues.append("❌ Objek yang di-load tidak memiliki method 'predict'.")
                else:
                    print("   ✅ Struktur Internal Model Valid")
                    
            except Exception as e:
                issues.append(f"❌ File Model Corrupt: {e}")

        # ==========================================
        # 2. CEK METADATA (.json)
        # ==========================================
        if not os.path.exists(Config.META_PATH):
            issues.append("⚠️ Metadata (.json) tidak ditemukan.")
        else:
            print(f"   ✅ Metadata ditemukan: {os.path.basename(Config.META_PATH)}")

        # ==========================================
        # 3. CEK PREDICTION LOG
        # ==========================================
        print("\n2️⃣  CEK LOG PREDIKSI")
        if not os.path.exists(Config.PREDICTION_LOG):
            print("   ℹ️  File log belum ada. Membuat baru...")
            try:
                pp = DiabetesPreprocessor()
                headers = pp.feature_order + ['result', 'confidence', 'timestamp']
                
                df_log = pd.DataFrame(columns=headers)
                df_log.to_csv(Config.PREDICTION_LOG, index=False, encoding='utf-8')
                
                fixes.append("✅ File Log Prediksi baru dibuat.")
                print("   ✅ Log file berhasil dibuat.")
            except Exception as e:
                issues.append(f"❌ Gagal membuat log file: {e}")
        else:
            print(f"   ✅ Log file ada: {os.path.basename(Config.PREDICTION_LOG)}")

        # ==========================================
        # 4. TES FUNGSI PREDIKSI (END-TO-END)
        # ==========================================
        print("\n3️⃣  TES PREDIKSI (SIMULASI API)")
        
        if model is None:
            print("   ⚠️  Skip tes prediksi karena model gagal dimuat.")
        else:
            # Sample data lengkap
            sample_data = {
                'age': 45, 'gender': 'Male', 'pulse_rate': 80,
                'systolic_bp': 120, 'diastolic_bp': 80,
                'glucose': 120, 'height': 170, 'weight': 70,
                'bmi': 0, 'family_diabetes': 'No', 'hypertensive': 'No',
                'family_hypertension': 'No', 'cardiovascular_disease': 'No', 
                'stroke': 'No'
            }

            try:
                preprocessor = DiabetesPreprocessor()
                df_sample = pd.DataFrame([sample_data])
                df_clean = preprocessor.clean_and_encode(df_sample)
                
                if df_clean.empty:
                    issues.append("❌ Preprocessing gagal: Output DataFrame kosong.")
                else:
                    X = preprocessor.get_features(df_clean)
                    prediction_val = model.predict(X)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probability = float(model.predict_proba(X)[0][1])
                    else:
                        probability = 1.0 if prediction_val == 1 else 0.0
                    
                    label = "Diabetic" if prediction_val == 1 else "Non-Diabetic"
                    print(f"   ✅ Input Sample : Pria, 45th, Gula 120")
                    print(f"   ✅ Hasil Model  : {label}")
                    print(f"   ✅ Confidence   : {probability:.2%}")
                    
                    # --- PERBAIKAN LOGIKA DETEKSI FEATURE IMPORTANCE ---
                    target_est = model
                    if hasattr(model, 'calibrated_classifiers_'):
                        calibrated_clf = model.calibrated_classifiers_[0]
                        # Support Scikit-Learn Lama (base_estimator) & Baru (estimator)
                        if hasattr(calibrated_clf, 'estimator'):
                            target_est = calibrated_clf.estimator
                        elif hasattr(calibrated_clf, 'base_estimator'):
                            target_est = calibrated_clf.base_estimator
                        
                    if hasattr(target_est, 'feature_importances_'):
                        print("   ✅ Feature Importance: TERSEDIA")
                    else:
                        issues.append("⚠️ Feature Importance TIDAK TERSEDIA pada model ini.")

            except Exception as e:
                issues.append(f"❌ Runtime Error saat Tes Prediksi: {e}")
                import traceback
                traceback.print_exc()

        # ==========================================
        # 5. KESIMPULAN & LAPORAN
        # ==========================================
        print("\n" + "=" * 60)
        print("📋 LAPORAN AKHIR")
        print("=" * 60)

        create_fix_report(issues, fixes)

        if issues:
            print(f"⚠️  DITEMUKAN {len(issues)} MASALAH:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            
            print("\n💡 REKOMENDASI PERBAIKAN:")
            if any("Model" in i for i in issues):
                print("   👉 Jalankan: python Scripts/train_model.py")
            if any("Log" in i for i in issues):
                print("   👉 Hapus file log lama di Backend/data/ dan jalankan ulang script ini.")
                
            return False
        else:
            print("✅ SISTEM SEHAT. SIAP DIGUNAKAN.")
            if fixes:
                print("ℹ️  Perbaikan otomatis yang diterapkan:")
                for f in fixes: print(f"   - {f}")
            return True

    except Exception as e:
        print(f"\n❌ Error Tak Terduga: {e}")
        return False

def create_fix_report(issues, fixes):
    """Menulis laporan ke file dengan encoding UTF-8 (Fix Charmap Error)"""
    report_path = os.path.join(Config.DATA_DIR, "prediction_fix_report.txt")
    try:
        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(Config.DATA_DIR)
            
        # PERBAIKAN UTAMA: Tambahkan encoding='utf-8'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LAPORAN DIAGNOSA SISTEM PREDIKSI\n")
            f.write("================================\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if issues:
                f.write("MASALAH DITEMUKAN:\n")
                for i in issues: f.write(f"- {i}\n")
            else:
                f.write("✅ Tidak ada masalah ditemukan.\n")
                
            if fixes:
                f.write("\nPERBAIKAN DITERAPKAN:\n")
                for f in fixes: f.write(f"- {f}\n")
                
        print(f"\n📄 Laporan disimpan: {report_path}")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan laporan: {e}")

if __name__ == "__main__":
    if fix_prediction_issues():
        print("\n✅ Diagnosa Selesai.")
    else:
        sys.exit(1)