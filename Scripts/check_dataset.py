"""
Scripts/check_dataset.py
Cek integritas dataset diabetes - FINAL STABLE VERSION
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 1. SETUP PATH (Absolute Path Fix)
# Memastikan Python bisa menemukan folder 'Backend'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. IMPORT DENGAN ERROR HANDLING
try:
    from Backend.config import Config
    from Backend.models.preprocess import DiabetesPreprocessor
except ImportError as e:
    print(f"❌ Gagal mengimpor modul: {e}")
    print("Pastikan Anda menjalankan script dari root folder proyek.")
    sys.exit(1)

def check_dataset():
    print("=" * 60)
    print("🔍 CEK INTEGRITAS DATASET DIABETES")
    print("=" * 60)
    
    issues = []
    
    # 3. KONVERSI PATH (Mengatasi AttributeError: 'str' object has no attribute 'exists')
    # Kita bungkus variabel config dengan Path() untuk memastikan fungsinya aktif
    raw_path = Path(Config.RAW_DATA)
    balanced_path = Path(Config.BALANCED_DATA)
    
    # 4. CEK FILE RAW
    if not raw_path.exists():
        print(f"❌ File Raw tidak ditemukan di: {raw_path}")
        print("💡 Pastikan file 'diabetes.csv' ada di folder Backend/data/")
        return
        
    try:
        df = pd.read_csv(raw_path)
        print(f"✅ File Raw Ditemukan")
        print(f"   • Total Sampel: {len(df)} baris")
    except Exception as e:
        print(f"❌ Gagal membaca CSV: {e}")
        return
    
    # 5. CEK STRUKTUR KOLOM
    try:
        pp = DiabetesPreprocessor()
        # Mengambil urutan fitur dari preprocessor
        required_features = getattr(pp, 'feature_order', [])
        required = required_features + ['diabetic']
        
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"❌ Struktur Kolom Tidak Lengkap!")
            print(f"   • Kolom hilang: {missing}")
            issues.append("Struktur kolom tidak valid")
        else:
            print("✅ Struktur Kolom Lengkap")
    except Exception as e:
        print(f"⚠️ Warning Preprocessor: {e}")

    # 6. CEK MISSING VALUE (DATA KOSONG)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"⚠️ Ditemukan {total_nulls} data kosong (Missing Values):")
        # Tampilkan kolom mana saja yang kosong
        for col, count in null_counts[null_counts > 0].items():
            print(f"   • Kolom '{col}': {count} baris")
        issues.append("Missing values found")
    else:
        print("✅ Data Bersih (Tidak ada Missing Values)")

    # 7. CEK DISTRIBUSI KELAS (IMBALANCE)
    if 'diabetic' in df.columns:
        counts = df['diabetic'].value_counts()
        print(f"📊 Distribusi Kelas Raw:")
        for label, count in counts.items():
            name = "Diabetes" if label == 1 else "Sehat"
            print(f"   • {name}: {count} ({ (count/len(df)*100):.1f}%)")
        
        # Jika jomplang banget (dibawah 20% salah satu kelas)
        if (counts.min() / counts.max()) < 0.2:
            print("⚠️ Status: Data Jomplang (Imbalanced). SMOTE sangat disarankan.")

    # 8. CEK BALANCED DATA (HASIL SMOTE)
    print("-" * 60)
    if balanced_path.exists():
        try:
            df_b = pd.read_csv(balanced_path)
            print(f"✅ Dataset Balanced Ditemukan")
            print(f"   • Total Sampel: {len(df_b)} baris")
            
            b_counts = df_b['diabetic'].value_counts()
            if b_counts[0] == b_counts[1]:
                print(f"   • Status: Sempurna (Ratio 1:1)")
            else:
                print(f"   • Status: Belum Sempurna ({b_counts[0]}:{b_counts[1]})")
        except Exception as e:
            print(f"⚠️ Gagal membaca dataset balanced: {e}")
    else:
        print("❌ Dataset Balanced Belum Ada")
        print("💡 Jalankan 'python Scripts/balance_dataset.py' segera!")
        issues.append("Balanced data missing")

    # 9. KESIMPULAN AKHIR
    print("=" * 60)
    if not issues:
        print("🚀 KESIMPULAN: DATASET SIAP UNTUK TRAINING!")
    elif "Struktur kolom tidak valid" in issues:
        print("❌ KESIMPULAN: DATASET RUSAK (Harus diperbaiki)")
    else:
        print("⚠️ KESIMPULAN: DATASET SIAP (Tapi jalankan balancing dulu)")
    print("=" * 60)

if __name__ == "__main__":
    check_dataset()