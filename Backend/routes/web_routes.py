from flask import Blueprint, render_template, current_app
import pandas as pd
import os
from Backend.config import Config

web_bp = Blueprint('web', __name__)

# --- RUTE 1: LANDING PAGE (HOME) ---
@web_bp.route('/')
def index():
    """
    Halaman Utama Sistem.
    Menampilkan visualisasi awal dan informasi umum model Decision Tree.
    """
    return render_template('pages/index.html', active_page='home')

# --- RUTE 2: TENTANG (ABOUT) ---
@web_bp.route('/about')
def about():
    """
    Halaman Dokumentasi Metodologi.
    Menjelaskan dataset DiaBD (5.288 pasien) dan performa model (Akurasi 99.26%).
    """
    return render_template('pages/about.html', active_page='about')

# --- RUTE 4: FORM DIAGNOSIS (PREDIKSI) ---
@web_bp.route('/predict')
def predict():
    """Halaman Formulir Input Klinis."""
    return render_template('pages/form.html', active_page='predict')

# --- RUTE 5: RIWAYAT LENGKAP ---
@web_bp.route('/history')
def history():
    """Halaman Log Riwayat Lengkap Aktivitas Prediksi."""
    return render_template('pages/logs.html', active_page='history')