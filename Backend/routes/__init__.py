"""
Backend/routes/__init__.py
Mengatur registrasi semua Blueprint (API & Web)
"""

from flask import Flask
from .api_routes import api_bp
from .web_routes import web_bp

def register_routes(app: Flask):
    """
    Mendaftarkan semua blueprint ke aplikasi Flask utama.
    """
    
    # 1. Register Web Routes (Dashboard, History)
    # URL: localhost:8000/
    app.register_blueprint(web_bp)
    
    # 2. Register API Routes (Predict, Logs, dll)
    # PENTING: Tambahkan url_prefix='/api' agar alamat menjadi localhost:8000/api/...
    # Ini wajib agar sinkron dengan formHandler.js yang memanggil '/api/predict'
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

# Expose blueprints agar bisa diimport manual jika perlu
__all__ = ['register_routes', 'api_bp', 'web_bp']