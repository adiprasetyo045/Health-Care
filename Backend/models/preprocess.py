import pandas as pd
import numpy as np

class DiabetesPreprocessor:
    def __init__(self):
        # 1. Mapping Kategori (Case-insensitive & Komprehensif)
        # Menangani variasi input dari form frontend dan dataset mentah
        self.gender_map = {
            'female': 0, 'woman': 0, 'f': 0, 'perempuan': 0, 'wanita': 0, '0': 0, 0: 0,
            'male': 1, 'man': 1, 'm': 1, 'laki-laki': 1, 'pria': 1, '1': 1, 1: 1
        }
        
        self.stroke_map = {
            '0': 0, 'no': 0, 'tidak': 0, 'n': 0, 0: 0,
            '1': 1, 'yes': 1, 'ya': 1, 'y': 1, 1: 1
        }
        
        self.target_map = {
            'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1,
            'non-diabetic': 0, 'diabetic': 1, 'negative': 0, 'positive': 1
        }
        
        # Mapping untuk kolom Boolean (Yes/No)
        self.bool_replace = {
            'yes': 1, 'ya': 1, 'true': 1, '1': 1, 'y': 1, 'ada': 1,
            'no': 0, 'tidak': 0, 'false': 0, '0': 0, 'n': 0, 'nan': 0, 'none': 0
        }
        
        # 2. Urutan Fitur WAJIB (14 Fitur)
        # Harus SAMA PERSIS dengan urutan saat model dilatih (X_train)
        self.feature_order = [
            'age', 'gender', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 
            'glucose', 'height', 'weight', 'bmi', 'family_diabetes', 
            'hypertensive', 'family_hypertension', 'cardiovascular_disease', 'stroke'
        ]

    def clean_and_encode(self, df, is_training=False):
        """
        Membersihkan data, melakukan encoding, dan menangani konversi satuan otomatis.
        """
        if df is None:
            return pd.DataFrame()

        # Jika input adalah dictionary (dari API), ubah ke DataFrame
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        
        # Jika DataFrame kosong
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()

        # --- A. STANDARISASI KOLOM ---
        # Memastikan semua kolom fitur ada, jika tidak isi dengan NaN sementara
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan

        # --- B. CLEANING NUMERIK ---
        numeric_cols = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']
        for col in numeric_cols:
            if col in df.columns:
                # Paksa ke numerik, ganti error dengan NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- C. SMART UNIT CONVERSION ---
        # 1. Glukosa: mg/dL (Satuan umum alat tes) -> mmol/L (Satuan Dataset DiaBD)
        if 'glucose' in df.columns:
            # Logika: Glukosa > 30 biasanya mg/dL (Normal puasa ~70-100). mmol/L biasanya 4-7.
            df['glucose'] = df['glucose'].apply(lambda x: round(x/18, 2) if (pd.notnull(x) and x > 30) else x)

        # 2. Tinggi: cm -> meter
        if 'height' in df.columns:
            # Logika: Tinggi > 3 biasanya cm (misal 170). Meter biasanya 1.7.
            df['height'] = df['height'].apply(lambda x: round(x/100, 2) if (pd.notnull(x) and x > 3) else x)

        # --- D. AUTO-CALCULATE BMI ---
        # Rumus BMI: Berat (kg) / Tinggi^2 (m)
        def _calc_bmi(row):
            h = row['height']
            w = row['weight']
            # Hitung hanya jika BMI kosong/0 dan komponen tersedia valid
            if (pd.isnull(row['bmi']) or row['bmi'] == 0) and (pd.notnull(h) and pd.notnull(w) and h > 0):
                return round(w / (h ** 2), 2)
            return row['bmi']

        # Terapkan perhitungan BMI baris per baris
        if 'bmi' in df.columns:
            df['bmi'] = df.apply(_calc_bmi, axis=1)

        # --- E. MAPPING KATEGORIKAL ---
        # Normalisasi string (lowercase, strip space) sebelum mapping
        
        # Gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.lower().str.strip().map(self.gender_map)
        
        # Stroke
        if 'stroke' in df.columns:
            df['stroke'] = df['stroke'].astype(str).str.lower().str.strip().map(self.stroke_map)

        # Kolom Biner Lainnya (Yes/No)
        bool_cols = ['family_diabetes', 'hypertensive', 'family_hypertension', 'cardiovascular_disease']
        for col in bool_cols:
            if col in df.columns:
                # Konversi manual dictionary replace lebih aman daripada map untuk parsial match
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].replace(self.bool_replace)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- F. HANDLING TARGET (KHUSUS TRAINING) ---
        if is_training and 'diabetic' in df.columns:
            df['diabetic'] = df['diabetic'].astype(str).str.lower().str.strip().map(self.target_map)
            df = df.dropna(subset=['diabetic'])
            df['diabetic'] = df['diabetic'].astype(int)

        # --- G. FINAL VALIDATION & FILLNA ---
        # Isi sisa NaN dengan 0 (Default aman untuk Decision Tree)
        # Idealnya data medis tidak boleh kosong, tapi sistem harus robust
        df[self.feature_order] = df[self.feature_order].fillna(0)
        
        # Casting akhir ke float32 (Standar Scikit-Learn)
        for col in self.feature_order:
            df[col] = df[col].astype('float32')

        return df

    def get_features(self, df):
        """Mengambil hanya kolom fitur (X) sesuai urutan training."""
        return df[self.feature_order]

    def get_target(self, df):
        """Mengambil kolom target (y) jika ada."""
        return df['diabetic'] if 'diabetic' in df.columns else None