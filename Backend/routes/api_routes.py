from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from Backend.config import Config
from Backend.models.preprocess import DiabetesPreprocessor

api_bp = Blueprint('api', __name__)

# --- 1. GLOBAL MODEL LOADING ---
model = None
model_meta = {}

def load_model_resources():
    """Memuat model pkl dan metadata json ke dalam memori global secara absolut."""
    global model, model_meta
    
    # Gunakan path absolut dari Config agar aman dijalankan dari folder manapun
    model_path = os.path.normpath(os.path.join(Config.MODELS_DIR, 'decision_tree_bundle.pkl'))
    meta_path = os.path.normpath(os.path.join(Config.MODELS_DIR, 'decision_tree_meta.json'))

    try:
        # Load Model
        if os.path.exists(model_path):
            loaded_data = joblib.load(model_path)
            # Handle jika model disimpan dalam dictionary (format baru) atau langsung model (format lama)
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model = loaded_data['model']
            else:
                model = loaded_data
            print(f"✅ Model loaded successfully from {model_path}")
        else:
            print(f"❌ Model file not found at: {model_path}")

        # Load Metadata (Akurasi, F1 Score, dll)
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                model_meta = json.load(f)
            print(f"✅ Metadata loaded successfully.")
            
    except Exception as e:
        print(f"❌ Error loading model resources: {e}")

# Inisialisasi model saat aplikasi pertama kali dijalankan
load_model_resources()

# --- 2. ENDPOINTS ---

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint utama untuk melakukan inferensi sistem pakar risiko diabetes."""
    global model
    
    # Reload model jika belum siap (Fail-safe mechanism)
    if model is None:
        load_model_resources()
        if model is None:
            return jsonify({'success': False, 'error': 'Sistem Inferensi belum siap. Hubungi admin.'}), 503

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'Format data tidak valid.'}), 400

        preprocessor = DiabetesPreprocessor()
        
        # 1. Konversi Data ke DataFrame
        df = pd.DataFrame([data])
        
        # 2. Preprocessing & Unit Conversion
        # Standar DiaBD: Konversi otomatis imperial ke metrik & hitung BMI
        df_clean = preprocessor.clean_and_encode(df)
        
        if df_clean.empty:
            return jsonify({'success': False, 'error': 'Input data berada di luar jangkauan klinis yang valid.'}), 400

        # 3. Prediksi Status (Inferensi Decision Tree CART)
        X = preprocessor.get_features(df_clean)
        prediction_val = model.predict(X)[0]
        prediction = int(prediction_val)
        
        # 4. Hitung Probabilitas (Calibrated Confidence Score)
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0][1])
        else:
            probability = 1.0 if prediction == 1 else 0.0
        
        # 5. Ekstrak Feature Importance (Faktor Dominan)
        # Menghitung kontribusi Information Gain setiap variabel terhadap keputusan
        feature_importance_list = []
        try:
            target_model = model
            
            # --- PERBAIKAN UTAMA: Penanganan Versi Scikit-Learn ---
            # Jika menggunakan CalibratedClassifierCV, kita harus mengambil estimator aslinya
            if hasattr(model, 'calibrated_classifiers_'):
                # Ambil classifier pertama dari list kalibrasi
                calibrated_clf = model.calibrated_classifiers_[0]
                
                # Cek atribut yang tersedia (estimator untuk sklearn baru, base_estimator untuk lama)
                if hasattr(calibrated_clf, 'estimator'):
                    target_model = calibrated_clf.estimator
                elif hasattr(calibrated_clf, 'base_estimator'):
                    target_model = calibrated_clf.base_estimator

            # Cek apakah target_model memiliki feature_importances_
            if hasattr(target_model, 'feature_importances_'):
                importances = target_model.feature_importances_
                feature_names = preprocessor.feature_order
                
                # Urutkan dari yang paling berpengaruh (Descending)
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                # Mapping nama variabel teknis ke bahasa medis yang user-friendly
                desc_map = {
                    'hypertensive': 'Status Hipertensi (Faktor Utama)',
                    'glucose': 'Kadar Glukosa Darah',
                    'height': 'Tinggi Badan Pasien',
                    'weight': 'Berat Badan Pasien',
                    'systolic_bp': 'Tekanan Darah Sistolik',
                    'pulse_rate': 'Detak Jantung (Pulse)',
                    'age': 'Faktor Usia',
                    'bmi': 'Indeks Massa Tubuh (BMI)',
                    'diastolic_bp': 'Tekanan Darah Diastolik',
                    'gender': 'Faktor Jenis Kelamin',
                    'family_diabetes': 'Riwayat Diabetes Keluarga',
                    'cvd': 'Riwayat Kardiovaskular',
                    'stroke': 'Riwayat Stroke',
                    'family_hypertension': 'Riwayat Hipertensi Keluarga'
                }
                
                # Format output dengan presisi 3 desimal untuk transparansi analisis
                feature_importance_list = [
                    {
                        'name': desc_map.get(name, name.replace('_', ' ').title()), 
                        'value': round(float(val) * 100, 3) 
                    } 
                    for name, val in feat_imp if val > 0
                ][:5] # Ambil Top 5 faktor terkuat
        except Exception as e:
            current_app.logger.warning(f"Feature Importance Extraction Warning: {e}")

        # 6. Logging ke CSV (Pencatatan Riwayat Pasien)
        try:
            log_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'result': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'confidence': f"{round(probability * 100, 2)}%",
                **data
            }
            log_path = Config.PREDICTION_LOG
            
            # Buat header jika file baru dibuat
            header_exists = os.path.exists(log_path)
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(log_path, mode='a', header=not header_exists, index=False, encoding='utf-8')
        except Exception as e:
            current_app.logger.error(f"Logging CSV failed: {e}")

        # 7. Final JSON Response
        # Struktur ini disesuaikan agar formHandler.js bisa merender grafik dan PDF
        return jsonify({
            'success': True,
            'label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_percent': round(probability * 100, 2),
            'risk_level': 'Tinggi' if probability >= 0.7 else ('Sedang' if probability >= 0.4 else 'Rendah'),
            'input_data': data,
            'feature_importance': feature_importance_list,
            'model_info': {
                'name': 'Decision Tree (CART)', 
                # Menggunakan fallback 99.26% jika metadata gagal dimuat
                'accuracy': f"{model_meta.get('accuracy_cv', 0.9926) * 100:.2f}%"
            }
        })

    except Exception as e:
        current_app.logger.error(f"Critical Prediction Error: {e}")
        return jsonify({'success': False, 'error': f'Kesalahan internal sistem: {str(e)}'}), 500

@api_bp.route('/logs', methods=['GET'])
def get_logs():
    """Mengambil riwayat log pemeriksaan untuk dashboard."""
    try:
        log_path = Config.PREDICTION_LOG
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            # Mengisi nilai kosong dengan '-' agar tidak error di frontend
            logs = df.tail(100).fillna('-').to_dict(orient='records')
            logs.reverse() # Tampilkan yang terbaru di atas
            return jsonify({"success": True, "logs": logs})
        return jsonify({"success": True, "logs": []})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """API untuk mengambil metadata performa model."""
    return jsonify(model_meta)