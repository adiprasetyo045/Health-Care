import os
import joblib
import pandas as pd
import numpy as np
import threading  # Tambahan untuk Thread Safety
from datetime import datetime
from typing import Dict, Any

from Backend.config import Config
from Backend.models.preprocess import DiabetesPreprocessor

class DiabetesModel:
    _instance = None
    _lock = threading.Lock()  # Pengunci untuk Singleton

    @staticmethod
    def get_instance():
        """Singleton Pattern dengan Thread Safety"""
        if DiabetesModel._instance is None:
            with DiabetesModel._lock:
                if DiabetesModel._instance is None:
                    DiabetesModel._instance = DiabetesModel()
        return DiabetesModel._instance

    def __init__(self):
        self.model_bundle = None
        self.preprocessor = DiabetesPreprocessor()
        self.load_bundle()

    def load_bundle(self):
        """Load model .pkl dari disk"""
        if not os.path.exists(Config.MODEL_PATH):
            print(f"⚠️ Warning: Model file not found at {Config.MODEL_PATH}")
            self.model_bundle = None
            return

        try:
            # Gunakan mmap_mode='r' untuk efisiensi jika model sangat besar
            bundle_data = joblib.load(Config.MODEL_PATH)
            
            # Normalisasi Format Bundle (Support Dict & Object langsung)
            if isinstance(bundle_data, dict) and 'model' in bundle_data:
                self.model_bundle = bundle_data
            else:
                # Jika format lama (langsung objek model), bungkus jadi dict
                self.model_bundle = {'model': bundle_data}
            
            print(f"✅ Model loaded successfully from {Config.MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_bundle = None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Melakukan prediksi end-to-end dengan validasi input.
        """
        if not self.model_bundle:
            return {
                "success": False,
                "error": "Sistem belum siap. Model gagal dimuat atau belum dilatih."
            }

        try:
            # 1. Konversi ke DataFrame
            df_raw = pd.DataFrame([input_data])

            # 2. Preprocessing
            df_clean = self.preprocessor.clean_and_encode(df_raw, is_training=False)
            
            if df_clean.empty:
                return {
                    "success": False,
                    "error": "Validasi klinis gagal. Pastikan parameter (Glukosa, BMI, dll) dalam rentang medis yang wajar."
                }

            # 3. Ambil Fitur sesuai urutan saat training
            X = self.preprocessor.get_features(df_clean)

            # 4. Prediksi
            model = self.model_bundle['model']
            
            # Mendukung model yang memiliki method predict_proba (Classifier)
            if hasattr(model, 'predict_proba'):
                prob_diabetic = model.predict_proba(X)[0][1]
            else:
                # Fallback jika model hanya predict (Regresi/Model tanpa probabilitas)
                pred = model.predict(X)[0]
                prob_diabetic = 1.0 if pred == 1 else 0.0

            prediction_label = "Diabetic" if prob_diabetic >= 0.5 else "Non-Diabetic"

            # 5. Interpretasi Klinis
            risk_level, interpretation = self._get_clinical_interpretation(prob_diabetic)

            # 6. Hasil Response
            return {
                "success": True,
                "label": prediction_label,
                "probability_percent": round(float(prob_diabetic) * 100, 2),
                "risk_level": risk_level,
                "interpretation": interpretation,
                # Mengembalikan data bersih untuk verifikasi
                "input_data": df_clean.to_dict(orient='records')[0], 
                "model_info": {
                    "algorithm": self.model_bundle.get('algorithm', 'Decision Tree'),
                    "accuracy": f"{self.model_bundle.get('accuracy_cv', 0.0) * 100:.2f}%",
                    "features_used": list(X.columns)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Internal Prediction Error: {str(e)}"
            }

    def _get_clinical_interpretation(self, probability: float):
        """Logika klasifikasi risiko berdasarkan ambang batas probabilitas """
        if probability >= 0.8:
            return "Sangat Tinggi", "Risiko sangat signifikan. Konsultasi dokter segera."
        elif probability >= 0.6:
            return "Tinggi", "Risiko tinggi. Perlu pemeriksaan lanjutan."
        elif probability >= 0.4:
            return "Sedang", "Risiko moderat. Pantau gaya hidup."
        elif probability >= 0.2:
            return "Rendah", "Risiko rendah. Pertahankan pola hidup sehat."
        else:
            return "Sangat Rendah", "Risiko minimal terpantau."