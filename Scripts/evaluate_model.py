"""
Scripts/evaluate_model.py
Evaluasi mendalam model Decision Tree (Accuracy, ROC, Bias, Confusion Matrix)
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

# 1. Setup Path Project
# Memastikan root folder terdeteksi dengan benar agar bisa import Backend
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root)) 

from Backend.config import Config
from Backend.models.preprocess import DiabetesPreprocessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def evaluate():
    print("=" * 70)
    print("📊 MODEL EVALUATION - DIABETES DSS")
    print("=" * 70)

    # --- 1. Validasi File ---
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Model tidak ditemukan di: {Config.MODEL_PATH}")
        print("   👉 Jalankan: python Scripts/train_model.py")
        return False

    if not os.path.exists(Config.BALANCED_DATA):
        print(f"❌ Dataset balanced tidak ditemukan di: {Config.BALANCED_DATA}")
        print("   👉 Jalankan: python Scripts/balance_dataset.py")
        return False

    try:
        # --- 2. Load Model Bundle ---
        print(f"📂 Loading model dari: {Config.MODEL_PATH}")
        bundle = joblib.load(Config.MODEL_PATH)
        
        # Handle format baru (dict) vs format lama (objek langsung)
        if isinstance(bundle, dict) and 'model' in bundle:
            model = bundle['model']
        else:
            model = bundle
        
        # --- 3. Load & Preprocess Data ---
        print(f"📂 Loading dataset dari: {Config.BALANCED_DATA}")
        df = pd.read_csv(Config.BALANCED_DATA)
        
        # Gunakan preprocessor yang SAMA dengan training/API
        pp = DiabetesPreprocessor()
        
        # Preprocessing (is_training=True agar target 'diabetic' diproses)
        df_clean = pp.clean_and_encode(df, is_training=True)
        
        if df_clean.empty:
            print("❌ Dataset kosong setelah cleaning.")
            return False

        X = pp.get_features(df_clean)
        y = pp.get_target(df_clean)

        print(f"   Total Sampel: {len(df_clean)}")
        print(f"   Distribusi  : {Counter(y)}")

        # --- 4. Lakukan Prediksi ---
        print("\n🔮 Melakukan prediksi pada seluruh dataset...")
        y_pred = model.predict(X)
        
        # Cek apakah model support probabilitas (untuk ROC-AUC)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = y_pred # Fallback

        # --- 5. Hitung Metrik ---
        acc = accuracy_score(y, y_pred)
        # Menggunakan 'binary' karena target klasifikasi adalah 0 dan 1
        prec = precision_score(y, y_pred, average='binary', zero_division=0)
        rec = recall_score(y, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y, y_pred, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = 0.0 

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # --- 6. Tampilkan Hasil ---
        print("\n📈 PERFORMA MODEL (Evaluation Phase)")
        print("-" * 30)
        print(f"   Accuracy        : {acc:.4f}")
        print(f"   Precision       : {prec:.4f}")
        print(f"   Recall          : {rec:.4f} (Sensitivitas)")
        print(f"   F1-Score        : {f1:.4f}")
        print(f"   ROC-AUC         : {auc:.4f}")
        
        print("\n📊 CONFUSION MATRIX")
        print(f"   {'':<15} {'Pred 0':<10} {'Pred 1':<10}")
        print(f"   {'Actual 0':<15} {tn:<10} {fp:<10}")
        print(f"   {'Actual 1':<15} {fn:<10} {tp:<10}")

        # --- 7. Analisis Bias ---
        # Menghitung Recall per kelas untuk melihat apakah model bias ke satu sisi
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print("\n⚖️  ANALISIS BIAS")
        print(f"   Akurasi Kelas 0 (Non-Diabetic) : {recall_0:.2%}")
        print(f"   Akurasi Kelas 1 (Diabetic)     : {recall_1:.2%}")
        
        diff = abs(recall_0 - recall_1)
        if diff > 0.15:
            print(f"   ⚠️  PERINGATAN: Terdeteksi Bias sebesar {diff:.2%}")
        else:
            print(f"   ✅  Model Seimbang (Bias < 15%)")

        # --- 8. Simpan Laporan JSON ---
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "auc": round(auc, 4)
            },
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            },
            "bias_analysis": {
                "recall_class_0": round(recall_0, 4),
                "recall_class_1": round(recall_1, 4),
                "is_balanced": bool(diff <= 0.15)
            }
        }
        
        # PERBAIKAN: Menggunakan os.path.join agar kompatibel dengan path string
        report_path = os.path.join(Config.DATA_DIR, "evaluation_results.json")
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        print(f"\n💾 Laporan evaluasi disimpan: {report_path}")
        return True

    except Exception as e:
        print(f"\n❌ Evaluasi Gagal: {e}")
        return False

if __name__ == "__main__":
    if evaluate():
        print("\n✅ Evaluasi Selesai.")
    else:
        sys.exit(1)