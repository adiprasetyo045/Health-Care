import pandas as pd
import os
import sys
from imblearn.over_sampling import SMOTE
from collections import Counter

# Menambahkan root folder ke path agar bisa import Backend
# Memastikan skrip bisa dijalankan dari folder manapun
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from Backend.config import Config
from Backend.models.preprocess import DiabetesPreprocessor

def balance_data():
    print("="*60)
    print("⚖️  BALANCING DATASET (SMOTE)")
    print("="*60)

    # --- 1. VALIDASI FILE RAW ---
    if not os.path.exists(Config.RAW_DATA):
        print(f"❌ Error: File RAW data tidak ditemukan di: {Config.RAW_DATA}")
        return

    try:
        # --- 2. LOAD DATA ---
        print(f"📂 Membaca data dari: {Config.RAW_DATA}")
        df = pd.read_csv(Config.RAW_DATA)

        # --- 3. PREPROCESSING (ENCODING & CLEANING) ---
        preprocessor = DiabetesPreprocessor()
        
        print("🔄 Membersihkan dan Encoding Data...")
        # PENTING: Gunakan is_training=True agar 'diabetic' ikut diproses & dibersihkan
        # Ini mencegah mismatch jumlah baris antara X dan y
        df_encoded = preprocessor.clean_and_encode(df, is_training=True)
        
        if df_encoded.empty:
            print("❌ Data kosong setelah preprocessing. Cek raw data.")
            return

        # Pisahkan X dan y menggunakan helper method dari class
        X = preprocessor.get_features(df_encoded)
        y = preprocessor.get_target(df_encoded)

        print(f"📊 Distribusi Awal: {Counter(y)}")

        # --- 4. TERAPKAN SMOTE ---
        print("🔄 Menjalankan algoritma SMOTE (Synthetic Minority Over-sampling)...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print(f"✅ Distribusi Setelah SMOTE: {Counter(y_resampled)}")

        # --- 5. GABUNGKAN KEMBALI & SIMPAN ---
        # Gabungkan X dan y hasil resampling menjadi DataFrame utuh
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced['diabetic'] = y_resampled

        # Simpan ke CSV baru
        output_path = Config.BALANCED_DATA
        df_balanced.to_csv(output_path, index=False)
        print(f"💾 Dataset seimbang disimpan ke: {output_path}")
        print("="*60)

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat balancing: {e}")
        # Opsional: Print traceback untuk debugging detail
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    balance_data()