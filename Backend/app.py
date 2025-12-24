import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# 1. SETUP PATH
# Mengambil path root project agar folder 'Backend' dikenali sebagai package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Backend.config import Config
from Backend.routes.api_routes import api_bp
from Backend.routes.web_routes import web_bp

def create_app():
    """Factory function untuk inisialisasi aplikasi Flask."""
    
    # 2. DEFINISI APP & PATH ASET
    # Memberitahu Flask lokasi folder templates dan static di dalam folder Backend
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'Backend', 'templates'),
        static_folder=os.path.join(BASE_DIR, 'Backend', 'static')
    )
    
    # Konversi Config jika ada di Backend/config.py
    app.config.from_object(Config)
    
    # Mengaktifkan CORS untuk integrasi antar-origin
    CORS(app) 
    
    # 3. REGISTER BLUEPRINTS
    # Web routes (untuk halaman HTML)
    app.register_blueprint(web_bp)
    
    # API routes (dengan prefix /api agar terstandarisasi)
    app.register_blueprint(api_bp, url_prefix='/api')

    # 4. ERROR HANDLERS
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
    app = create_app()
    
    # Konfigurasi Port
    PORT = 8000 
    
    print("="*60)
    print(f"🚀 DIABETES PREDICTION SYSTEM BERJALAN DI PORT {PORT}")
    print("="*60)
    print(f"📄 Dashboard  : http://localhost:{PORT}/")
    print(f"📄 Form Entry : http://localhost:{PORT}/predict")
    print(f"🔌 API Health : http://localhost:{PORT}/api/health")
    print("="*60)
    
    # Host 0.0.0.0 agar bisa diakses di jaringan lokal (development mode)
    app.run(host='0.0.0.0', port=PORT, debug=True)