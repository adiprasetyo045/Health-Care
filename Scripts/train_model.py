"""
Scripts/train_model.py
Training model Decision Tree dengan Pipeline, Cross Validation, dan Kalibrasi.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from pathlib import Path

# 1. Setup Path Project
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 2. Import Module
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

def train_model():
    print("=" * 60)
    print("🧠 TRAINING MODEL DIABETES")
    print("=" * 60)

    try:
        # 3. Load Dataset Balanced
        if not os.path.exists(Config.BALANCED_DATA):
            print(f"❌ Dataset balanced tidak ditemukan di: {Config.BALANCED_DATA}")
            return False

        print(f"📂 Membaca dataset: {Config.BALANCED_DATA}")
        df = pd.read_csv(Config.BALANCED_DATA)

        # 4. Preprocessing Cerdas
        preprocessor = DiabetesPreprocessor()
        
        # Cek apakah data sudah numeric (Balanced Data)
        if np.issubdtype(df['gender'].dtype, np.number):
            print("ℹ️  Info: Dataset terdeteksi sudah numerik (Balanced). Skip encoding.")
            df_clean = df.copy()
            df_clean = df_clean.dropna()
        else:
            print("ℹ️  Info: Dataset mentah (String). Menjalankan encoding...")
            df_clean = preprocessor.clean_and_encode(df, is_training=True)
        
        if len(df_clean) == 0:
            print("❌ ERROR: Dataset kosong setelah preprocessing!")
            return False

        # Ambil Fitur & Target
        X = df_clean[preprocessor.feature_order]
        y = df_clean['diabetic']
        
        feature_names = list(X.columns)
        print(f"📊 Dataset Shape: {X.shape}")
        print(f"📊 Distribusi Kelas: {Counter(y)}")

        # 5. Membangun Pipeline
        dt_classifier = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=6,
            min_samples_leaf=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('dt', dt_classifier)
        ])

        # 6. Kalibrasi Model
        calibrated_model = CalibratedClassifierCV(
            estimator=pipeline,
            method='sigmoid',
            cv=5
        )

        # 7. Evaluasi Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\n🔄 Menjalankan 5-Fold Cross Validation...")
        scores = cross_val_score(calibrated_model, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        
        print(f"📈 Rata-rata Akurasi Validasi: {mean_acc:.4f} (±{scores.std():.4f})")

        # 8. Final Training
        print("💪 Melatih model final...")
        calibrated_model.fit(X, y)

        y_pred = calibrated_model.predict(X)
        y_proba = calibrated_model.predict_proba(X)[:, 1]

        # Hitung Metrik
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Extract Feature Importance
        base_pipeline = calibrated_model.calibrated_classifiers_[0].estimator
        importance_vals = base_pipeline.named_steps['dt'].feature_importances_
        feature_importance = sorted(zip(feature_names, importance_vals), key=lambda x: x[1], reverse=True)

        # 10. Simpan Model (BAGIAN INI YANG DIPERBAIKI)
        # Menggunakan Config.MODELS_DIR (Jamak/Plural)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        bundle = {
            'model': calibrated_model,
            'features': feature_names,
            'target': ['Non-Diabetic', 'Diabetic'],
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(bundle, Config.MODEL_PATH)
        print(f"\n💾 Model tersimpan: {Config.MODEL_PATH}")

        # 11. Simpan Metadata
        metadata = {
            'algorithm': 'Calibrated Decision Tree (Entropy)',
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'accuracy_cv': round(mean_acc, 4),
            'accuracy_train': round(metrics['accuracy'], 4),
            'metrics': {k: round(v, 4) for k, v in metrics.items()},
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'feature_importance': {k: round(v, 4) for k, v in feature_importance[:5]}
        }

        with open(Config.META_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"📄 Metadata tersimpan: {Config.META_PATH}")
        return True

    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if train_model():
        print("\n✅ PROSES SELESAI. Model siap digunakan.")
    else:
        print("\n❌ PROSES GAGAL.")
        sys.exit(1)