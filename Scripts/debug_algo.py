"""
Scripts/debug_algo.py
Script untuk menguji alur Preprocessing dan Algoritma secara manual.
"""

import sys
import os  # <--- INI YANG KURANG TADI
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Setup Path Project
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 2. Import Module
try:
    from Backend.config import Config
    from Backend.models.decision_tree_model import DiabetesModel
    from Backend.models.preprocess import DiabetesPreprocessor
except ModuleNotFoundError:
    # Fallback jika path belum terdeteksi
    sys.path.append(os.getcwd())
    from Backend.config import Config
    from Backend.models.decision_tree_model import DiabetesModel
    from Backend.models.preprocess import DiabetesPreprocessor

def debug_flow():
    print("=" * 60)
    print("🔬 DEBUG LOGIKA: PREPROCESS & ALGORITMA")
    print("=" * 60)

    # ==========================================
    # A. INPUT DATA MENTAH (Simulasi dari User)
    # ==========================================
    # Skenario: Laki-laki paruh baya, Gula Darah Tinggi, tapi lupa isi BMI
    raw_input = {
        'age': 55,
        'gender': 'Male',             # String (Perlu di-encode)
        'pulse_rate': 85,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'glucose': 200,               # Gula darah tinggi
        'height': 170,
        'weight': 80,
        'bmi': 0,                     # User lupa isi BMI (Sistem harus hitung)
        'family_diabetes': 'Yes',     # String
        'hypertensive': 1,
        'family_hypertension': 'No',  # String
        'cardiovascular_disease': 0,
        'stroke': 'No'                # String
    }

    print("\n1️⃣  INPUT DATA MENTAH (Dari Form User)")
    for k, v in raw_input.items():
        print(f"   • {k.ljust(25)}: {v} ({type(v).__name__})")

    # ==========================================
    # B. PROSES PREPROCESSING
    # ==========================================
    print("\n" + "-"*60)
    print("2️⃣  PROSES PREPROCESSING (Pembersihan Data)")
    
    pp = DiabetesPreprocessor()
    
    # Konversi dict ke DataFrame (mirip cara kerja API)
    df_raw = pd.DataFrame([raw_input])
    
    # Jalankan Cleaning
    # is_training=False artinya kita sedang mode prediksi (real-time)
    df_clean = pp.clean_and_encode(df_raw, is_training=False)
    
    # Tampilkan Hasil Konversi
    print("   ⬇️  Mengubah Data...")
    
    # Gender
    print(f"   • Gender '{raw_input['gender']}' \t\t👉 Jadi: {df_clean.iloc[0]['gender']} (1=Male, 0=Female)")
    
    # BMI Calculation
    print(f"   • BMI (Awal: {raw_input['bmi']}) \t👉 Hitung Otomatis: {df_clean.iloc[0]['bmi']:.2f}")
    
    # Yes/No Conversion
    print(f"   • Family Diab '{raw_input['family_diabetes']}' \t👉 Jadi: {df_clean.iloc[0]['family_diabetes']}")
    
    # Cek Shape Akhir
    print(f"\n   ✅ Data siap masuk model (Format Numerik):")
    vals = df_clean.values[0]
    # Tampilkan 5 nilai pertama sebagai sampel
    print(f"   Values (Sample): {vals[:5]} ...")

    # ==========================================
    # C. EKSEKUSI ALGORITMA (MODEL)
    # ==========================================
    print("\n" + "-"*60)
    print("3️⃣  EKSEKUSI MODEL (Decision Tree)")
    
    try:
        model = DiabetesModel.get_instance()
        
        # Prediksi
        result = model.predict(raw_input)
        
        if result['success']:
            print(f"   🎯 HASIL PREDIKSI:")
            print(f"      - Label       : {result['label']}")
            print(f"      - Probabilitas: {result['probability_percent']}%")
            print(f"      - Risiko      : {result['risk_level']}")
            print(f"      - Interpretasi: {result['interpretation']}")
            
            # Penjelasan Logic Decision Tree (Sederhana)
            print("\n   🧠 Logika Algoritma (Simulasi):")
            # Logic sederhana untuk debugging visual
            if df_clean.iloc[0]['glucose'] > 140:
                print("      ⚠️ Gula Darah > 140 mg/dL berkontribusi besar pada prediksi Diabetes.")
            if df_clean.iloc[0]['bmi'] > 25:
                 print("      ⚠️ BMI Overweight berkontribusi pada risiko.")
        else:
            print(f"   ❌ Error Prediksi: {result.get('error')}")
            
    except Exception as e:
        print(f"   ❌ Gagal memuat model: {e}")
        print("      (Pastikan Anda sudah menjalankan 'python Scripts/train_model.py')")

    print("=" * 60)

if __name__ == "__main__":
    debug_flow()