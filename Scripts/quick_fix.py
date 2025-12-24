"""
Scripts/quick_fix.py
Perbaikan otomatis struktur project, dependency, dan file konfigurasi.
"""

import sys
import os
import datetime
import shutil
from pathlib import Path

# Setup Project Root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quick_fix():
    """Fungsi utama perbaikan sistem"""
    print("=" * 60)
    print("⚡ PERBAIKAN CEPAT SISTEM DIABETES")
    print("=" * 60)
    
    fixes = []
    
    try:
        # 1. Cek dan Buat Direktori Wajib
        directories = [
            project_root / "Backend" / "data",
            project_root / "Backend" / "models",
            project_root / "Backend" / "routes",
            project_root / "Backend" / "static" / "css",
            project_root / "Backend" / "static" / "js",
            project_root / "Backend" / "templates",
            project_root / "Scripts"
        ]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                fixes.append(f"✅ Directory dibuat: {directory.relative_to(project_root)}")
        
        # 2. FILE KRUSIAL: Backend/__init__.py (Agar Backend terbaca sebagai Package)
        init_file = project_root / "Backend" / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write('"""Backend Package"""\n')
            fixes.append("✅ Backend/__init__.py dibuat (Fix ModuleNotFoundError)")

        # 3. Cek requirements.txt (Versi Stabil)
        requirements_path = project_root / "requirements.txt"
        # Kita overwrite saja untuk memastikan versi aman
        with open(requirements_path, 'w') as f:
            f.write("""# Core Web Framework
Flask==3.0.0
Flask-CORS==4.0.0
Werkzeug==3.0.1

# Data Manipulation
# PENTING: Pin numpy di bawah versi 2.0
numpy<2.0.0
pandas==2.1.4

# Machine Learning
scikit-learn==1.3.2
joblib==1.3.2
imbalanced-learn==0.11.0

# Utilities
python-dotenv==1.0.0
""")
        fixes.append("✅ requirements.txt diperbarui (Versi Stabil)")
        
        # 4. Cek File Utama (run_app, start.sh, gitignore)
        
        # --- A. run_app.py (Port 8000) ---
        run_app_path = project_root / "run_app.py"
        if not run_app_path.exists():
            with open(run_app_path, 'w') as f:
                f.write("""import sys
import os

# Pastikan root folder masuk ke path
sys.path.append(os.getcwd())

from Backend.app import create_app

# Buat aplikasi
app = create_app()

if __name__ == "__main__":
    PORT = 8000
    print(f"🏥 Starting Diabetes DSS Server on Port {PORT}...")
    
    # Jalankan di Port 8000
    app.run(host='0.0.0.0', port=PORT, debug=True)
""")
            fixes.append("✅ run_app.py dibuat")

        # --- B. start.sh ---
        start_sh_path = project_root / "start.sh"
        if not start_sh_path.exists():
            with open(start_sh_path, 'w') as f:
                f.write("""#!/bin/bash
echo "============================================================"
echo "🚀 Starting Diabetes Detector"
echo "============================================================"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

echo "Starting application on Port 8000..."
python3 run_app.py
""")
            # Make executable
            try:
                os.chmod(start_sh_path, 0o755)
            except:
                pass
            fixes.append("✅ start.sh dibuat")

        # --- C. .gitignore ---
        gitignore_path = project_root / ".gitignore"
        # Overwrite dengan versi final
        with open(gitignore_path, 'w') as f:
            f.write("""__pycache__/
*.py[cod]
.venv/
Backend/models/*.pkl
Backend/data/diabetes_balanced.csv
Backend/data/prediction_log.csv
!Backend/data/diabetes.csv
!Backend/models/decision_tree_meta.json
""")
        fixes.append("✅ .gitignore diperbarui")
        
        # 5. Buat Sample Dataset (Jika Tidak Ada)
        # Data disesuaikan dengan format RAW (String Categories)
        dataset_path = project_root / "Backend" / "data" / "diabetes.csv"
        if not dataset_path.exists():
            import pandas as pd
            import numpy as np
            
            print("⏳ Generating sample dataset...")
            np.random.seed(42)
            n_samples = 200
            
            data = {
                'age': np.random.randint(18, 80, n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'pulse_rate': np.random.randint(60, 100, n_samples),
                'systolic_bp': np.random.randint(100, 180, n_samples),
                'diastolic_bp': np.random.randint(60, 120, n_samples),
                'glucose': np.random.uniform(70, 200, n_samples).round(2),
                'height': np.random.uniform(1.50, 1.90, n_samples).round(2),
                'weight': np.random.uniform(50, 100, n_samples).round(1),
                'bmi': np.zeros(n_samples), # Auto calc nanti
                'family_diabetes': np.random.choice([0, 1], n_samples),
                'hypertensive': np.random.choice([0, 1], n_samples),
                'family_hypertension': np.random.choice([0, 1], n_samples),
                'cardiovascular_disease': np.random.choice([0, 1], n_samples),
                'stroke': np.random.choice(['0', '1'], n_samples),
            }
            
            # Hitung BMI kasar
            data['bmi'] = (data['weight'] / (data['height'] ** 2)).round(2)
            
            # Target
            diabetic = []
            for i in range(n_samples):
                risk = 0
                if data['glucose'][i] > 140: risk += 2
                if data['bmi'][i] > 30: risk += 1
                if data['age'][i] > 50: risk += 1
                diabetic.append('Yes' if risk >= 3 else 'No')
            
            data['diabetic'] = diabetic
            
            df = pd.DataFrame(data)
            df.to_csv(dataset_path, index=False)
            fixes.append("✅ Dataset sampel (diabetes.csv) dibuat")
        
        # 6. Summary Report
        create_quick_fix_report(fixes)
        
        print(f"\n📋 TOTAL PERBAIKAN: {len(fixes)}")
        for fix in fixes:
            print(f"  {fix}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error Critical: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_fix_report(fixes):
    """Buat laporan text"""
    report_path = project_root / "quick_fix_report.txt"
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(report_path, 'w') as f:
        f.write("LAPORAN PERBAIKAN CEPAT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tanggal: {timestamp}\n\n")
        f.write("PERBAIKAN YANG DITERAPKAN:\n")
        f.write("-" * 30 + "\n")
        
        if fixes:
            for i, fix in enumerate(fixes, 1):
                f.write(f"{i}. {fix}\n")
        else:
            f.write("Sistem sudah lengkap.\n")
            
        f.write("\nSTATUS SYSTEM:\n✅ SIAP DIJALANKAN (Port 8000)\n")

if __name__ == "__main__":
    if quick_fix():
        print("\n" + "=" * 60)
        print("✅ SISTEM SIAP! Silakan jalankan langkah berikut:")
        print("1. pip install -r requirements.txt")
        print("2. python Scripts/balance_dataset.py")
        print("3. python Scripts/train_model.py")
        print("4. python run_app.py")
        print("=" * 60)
    else:
        print("❌ GAGAL MEMPERBAIKI SISTEM")
        sys.exit(1)