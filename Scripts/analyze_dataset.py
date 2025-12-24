"""
Scripts/analyze_dataset.py
Analisis dataset diabetes (Raw & Balanced)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from pathlib import Path

# 1. Setup Path Project
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 2. Import Module (Robust)
try:
    from Backend.config import Config
    from Backend.models.preprocess import DiabetesPreprocessor
except ModuleNotFoundError:
    try:
        from backend.config import Config
        from backend.models.preprocess import DiabetesPreprocessor
    except ModuleNotFoundError:
        print("❌ CRITICAL ERROR: Module 'Backend' tidak ditemukan.")
        sys.exit(1)

def analyze_dataset():
    print("=" * 60)
    print("📊 ANALISIS DATASET DIABETES")
    print("=" * 60)

    try:
        # 1. Cek File RAW
        if not os.path.exists(Config.RAW_DATA):
            print(f"❌ Error: File {Config.RAW_DATA} tidak ditemukan.")
            return False

        print(f"📂 Membaca RAW Data: {Config.RAW_DATA}")
        df_raw = pd.read_csv(Config.RAW_DATA)
        
        # Gunakan Preprocessor untuk membersihkan data
        pp = DiabetesPreprocessor()
        df_clean = pp.clean_and_encode(df_raw, is_training=True)
        
        # 2. Analisis Distribusi Kelas (RAW)
        target_col = 'diabetic'
        if target_col not in df_clean.columns:
            print(f"❌ Kolom target '{target_col}' tidak ditemukan.")
            return False

        counts = df_clean[target_col].value_counts()
        total = len(df_clean)
        neg, pos = counts.get(0, 0), counts.get(1, 0)
        ratio = neg / pos if pos > 0 else 0

        print(f"\n1️⃣  STATISTIK RAW DATA (Setelah Cleaning)")
        print(f"   - Total Sampel : {total}")
        print(f"   - Fitur        : {len(df_clean.columns) - 1}")
        print(f"   - Non-Diabetic : {neg} ({neg/total:.1%})")
        print(f"   - Diabetic     : {pos} ({pos/total:.1%})")
        print(f"   - Imbalance    : {ratio:.2f} : 1")
        
        if ratio > 1.5:
             print("   ⚠️  STATUS: IMBALANCED (Perlu SMOTE)")
        else:
             print("   ✅  STATUS: BALANCED")

        # 3. Analisis Missing Values (Raw)
        print(f"\n2️⃣  KUALITAS DATA MENTAH")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("   ⚠️  Kolom dengan Missing Values:")
            for col, val in missing.items():
                print(f"      - {col.ljust(20)}: {val} kosong")
        else:
            print("   ✅  Tidak ada Missing Values")

        # 4. Cek Dataset Balanced
        print(f"\n3️⃣  STATISTIK BALANCED DATA")
        if os.path.exists(Config.BALANCED_DATA):
            df_bal = pd.read_csv(Config.BALANCED_DATA)
            b_counts = df_bal[target_col].value_counts()
            b_neg, b_pos = b_counts.get(0, 0), b_counts.get(1, 0)
            
            print(f"   - File Ditemukan : ✅")
            print(f"   - Total Sampel   : {len(df_bal)}")
            print(f"   - Non-Diabetic   : {b_neg}")
            print(f"   - Diabetic       : {b_pos}")
            print(f"   - Ratio          : {b_neg/b_pos:.2f} : 1")
        else:
            print(f"   - File Ditemukan : ❌ (Jalankan balance_dataset.py)")

        # 5. Generate Report
        generate_report(df_clean, counts, ratio)
        
        return True

    except Exception as e:
        print(f"\n❌ Error Analisis: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report(df, counts, ratio):
    """Membuat file laporan teks"""
    report_path = Config.DATA_REPORT
    
    with open(report_path, 'w') as f:
        f.write("LAPORAN ANALISIS DATASET\n")
        f.write("========================\n")
        f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("STATISTIK DESKRIPTIF (Cleaned Data):\n")
        f.write(str(df.describe().round(2)))
        f.write("\n\n")
        
        f.write("DISTRIBUSI KELAS:\n")
        f.write(f"Non-Diabetic (0): {counts.get(0, 0)}\n")
        f.write(f"Diabetic (1)    : {counts.get(1, 0)}\n")
        f.write(f"Ratio           : {ratio:.2f}:1\n")
        
        if ratio > 1.5:
            f.write("\nREKOMENDASI: Lakukan Balancing (SMOTE) sebelum training.\n")
        else:
            f.write("\nREKOMENDASI: Data sudah cukup seimbang.\n")

    print(f"\n📄 Laporan lengkap disimpan di: {report_path}")

if __name__ == "__main__":
    if analyze_dataset():
        print("\n✅ Analisis Selesai.")
    else:
        print("\n❌ Analisis Gagal.")