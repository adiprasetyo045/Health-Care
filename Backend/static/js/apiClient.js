/**
 * Diabetes API Client - Frontend JavaScript
 * REVISI FINAL: Sinkron dengan Backend Port 8000
 * Berfungsi sebagai jembatan komunikasi antara Browser dan Flask
 */

class DiabetesApiClient {
    constructor(baseUrl = '') {
        // Otomatis deteksi URL. Jika kosong, gunakan origin saat ini (http://localhost:8000)
        this.baseUrl = baseUrl || window.location.origin;
        this.isConnected = false;
        
        console.log('🌐 Diabetes API Client initialized');
        console.log('   Base URL:', this.baseUrl);
    }

    /**
     * Core Request Handler (Wrapper untuk fetch)
     */
    async request(endpoint, options = {}) {
        // Pastikan endpoint diawali dengan '/'
        const cleanEndpoint = endpoint.startsWith('/') ? endpoint : '/' + endpoint;
        const url = this.baseUrl + cleanEndpoint;
        
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };

        const finalOptions = { ...defaultOptions, ...options };

        if (finalOptions.body && typeof finalOptions.body === 'object') {
            finalOptions.body = JSON.stringify(finalOptions.body);
        }

        try {
            const response = await fetch(url, finalOptions);
            const contentType = response.headers.get("content-type");
            
            // Handle jika server error dan tidak mengembalikan JSON (misal 500 HTML error)
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error(`Server Error: Received non-JSON response (${response.status})`);
            }

            const data = await response.json();

            if (!response.ok) {
                const errorMessage = data.error || data.message || `HTTP Error ${response.status}`;
                throw new Error(errorMessage);
            }

            this.isConnected = true;
            return data;

        } catch (error) {
            console.error(`❌ API Error [${finalOptions.method}] ${endpoint}:`, error.message);
            this.isConnected = false;
            throw error;
        }
    }

    /**
     * CEK KONEKSI (Health Check)
     * Menggunakan endpoint model-info karena api_routes.py tidak memiliki /health khusus.
     */
    async checkConnection() {
        return this.request('/api/model-info');
    }

    /**
     * PREDIKSI (Endpoint Utama)
     * Mengirim data payload mentah ke Backend.
     * Biarkan Backend/models/preprocess.py yang melakukan cleaning agar konsisten.
     */
    async predict(payload) {
        if (!payload || typeof payload !== 'object') {
            throw new Error("Data input tidak valid.");
        }
        
        return this.request('/api/predict', {
            method: 'POST',
            body: payload
        });
    }

    /**
     * GET LOGS/HISTORY
     * Mengambil riwayat prediksi untuk Dashboard
     */
    async getLogs() {
        return this.request('/api/logs');
    }

    /**
     * GET MODEL INFO
     * Mengambil metadata akurasi model
     */
    async getModelInfo() {
        return this.request('/api/model-info');
    }
}

// Global Export agar bisa dipakai di console browser untuk debugging
const api = new DiabetesApiClient();
window.apiClient = api;

// Auto-check connection saat file dimuat
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await api.checkConnection();
        console.log('✅ API Connected & Healthy');
        // Dispatch event custom jika ada bagian UI yang butuh tahu status koneksi
        document.dispatchEvent(new CustomEvent('api-connected'));
    } catch (e) {
        console.warn('⚠️ API Offline atau tidak merespons: Pastikan server Python berjalan di Port 8000');
    }
});