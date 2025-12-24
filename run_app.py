import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# 1. SETUP PATH PROYEK
# Memastikan root directory ditambahkan ke path agar modul 'Backend' dapat di-import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from Backend.config import Config
    from Backend.routes.api_routes import api_bp
    from Backend.routes.web_routes import web_bp
except ImportError as e:
    print(f"❌ Error saat memuat modul: {e}")
    sys.exit(1)

def create_app():
    """Factory function untuk inisialisasi aplikasi Flask."""
    
    # 2. INISIALISASI APLIKASI & PATH ASET
    # Memberitahu Flask lokasi folder templates dan static di dalam folder Backend
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'Backend', 'templates'),
        static_folder=os.path.join(BASE_DIR, 'Backend', 'static')
    )
    
    # Memuat konfigurasi dari objek Config
    app.config.from_object(Config)
    
    # 3. INISIALISASI SISTEM (Folder data/model/logs)
    # Menjalankan fungsi inisialisasi yang ada di config.py
    if hasattr(Config, 'init_app'):
        Config.init_app()
    
    # Mengaktifkan CORS untuk integrasi frontend-backend yang mulus
    CORS(app) 
    
    # 4. REGISTER BLUEPRINTS
    # Web routes (untuk melayani file HTML)
    app.register_blueprint(web_bp)
    
    # API routes (dengan url_prefix /api sesuai standar REST API)
    app.register_blueprint(api_bp, url_prefix='/api')

    # 5. GLOBAL ERROR HANDLERS
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({
                "error": "Endpoint API tidak ditemukan",
                "path": request.path
            }), 404
        return "<h1>404 - Halaman Tidak Ditemukan</h1>", 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({
            "error": "Internal Server Error", 
            "message": str(e)
        }), 500

    return app

if __name__ == "__main__":
    # Inisialisasi app
    app = create_app()
    
    # Menggunakan port dari Config jika tersedia, atau default 8000
    PORT = getattr(Config, 'SERVER_PORT', 8000)
    HOST = getattr(Config, 'SERVER_HOST', '0.0.0.0')
    DEBUG = getattr(Config, 'DEBUG', True)
    
    print("="*60)
    print(f"🚀 DIABETES PREDICTION SYSTEM BERJALAN")
    print("="*60)
    print(f"🔗 Dashboard  : http://localhost:{PORT}/")
    print(f"🔗 API Health : http://localhost:{PORT}/api/health")
    print(f"📂 Folder Proyek: {BASE_DIR}")
    print("="*60)
    
    # Menjalankan server
    app.run(host=HOST, port=PORT, debug=DEBUG)