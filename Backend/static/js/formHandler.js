/**
 * formHandler.js - Final Fixed Version
 * Memaksa teks menjadi HITAM agar terlihat di kertas PDF PUTIH.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. INISIALISASI ---
    const hInput = document.getElementById('height');
    const wInput = document.getElementById('weight');
    const bInput = document.getElementById('bmi');
    const btnPredict = document.getElementById('btnPredict');
    const form = document.getElementById('predictionForm');
    const resContainer = document.getElementById('resultContainer');

    // --- 2. LOGIKA AUTO BMI ---
    const updateBMI = () => {
        const h = parseFloat(hInput.value);
        const w = parseFloat(wInput.value);
        if (h > 0 && w > 0) {
            const bmi = (w / (h * h)).toFixed(2);
            bInput.value = bmi;
        } else {
            bInput.value = '';
        }
    };

    if (hInput && wInput) {
        hInput.addEventListener('input', updateBMI);
        wInput.addEventListener('input', updateBMI);
    }

    // --- 3. LOGIKA PREDIKSI ---
    if (btnPredict && form) {
        btnPredict.addEventListener('click', async (e) => {
            e.preventDefault(); 

            if (!form.checkValidity()) {
                form.reportValidity();
                return;
            }

            const originalText = btnPredict.innerHTML;
            btnPredict.innerHTML = '⏳ Menjalankan Inferensi...';
            btnPredict.disabled = true;
            resContainer.style.display = 'none';

            try {
                const formData = new FormData(form);
                const rawData = Object.fromEntries(formData.entries());
                
                const intFields = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp'];
                const floatFields = ['glucose', 'height', 'weight', 'bmi'];
                
                const payload = {};
                for (const key in rawData) {
                    if (intFields.includes(key)) {
                        payload[key] = parseInt(rawData[key]) || 0;
                    } else if (floatFields.includes(key)) {
                        payload[key] = parseFloat(rawData[key]) || 0;
                    } else {
                        payload[key] = rawData[key];
                    }
                }

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

                const res = await response.json();

                if (res.success) {
                    resContainer.style.display = 'block';

                    const isDiabetic = res.label.toLowerCase() === 'diabetic';
                    const color = isDiabetic ? '#ef4444' : '#22c55e';
                    
                    const resLabel = document.getElementById('resLabel');
                    resLabel.innerText = isDiabetic ? 'DIAGNOSIS: DIABETIC' : 'DIAGNOSIS: NON-DIABETIC';
                    resLabel.style.color = color;
                    
                    document.getElementById('resRisk').innerText = res.risk_level;
                    const probEl = document.getElementById('resProb');
                    probEl.innerText = res.probability_percent + '%';
                    probEl.style.color = color;

                    document.getElementById('modName').innerText = res.model_info.name || 'Decision Tree (CART)';
                    document.getElementById('modMethod').innerText = '5-Fold CV (Calibrated)';
                    document.getElementById('modAcc').innerText = res.model_info.accuracy || '99.26%';

                    const iList = document.getElementById('inputList');
                    iList.innerHTML = '';
                    const labelMap = { 
                        age: 'Usia Pasien', gender: 'Jenis Kelamin', glucose: 'Kadar Glukosa', 
                        bmi: 'BMI (kg/m²)', systolic_bp: 'Tensi Sistolik', diastolic_bp: 'Tensi Diastolik',
                        height: 'Tinggi (m)', weight: 'Berat (kg)', family_diabetes: 'Riwayat Diabetes',
                        hypertensive: 'Status Hipertensi'
                    };
                    
                    ['age', 'gender', 'glucose', 'bmi', 'systolic_bp', 'diastolic_bp', 'hypertensive'].forEach(k => {
                        if (res.input_data[k] !== undefined) {
                            iList.innerHTML += `<li><span>${labelMap[k] || k}</span> <strong>${res.input_data[k]}</strong></li>`;
                        }
                    });

                    // RENDER GRAFIK FEATURE IMPORTANCE (Dengan Class CSS yang benar)
                    const fList = document.getElementById('featList');
                    fList.innerHTML = '';
                    
                    if (res.feature_importance && res.feature_importance.length > 0) {
                        res.feature_importance.forEach(f => {
                            fList.innerHTML += `
                                <li style="margin-bottom: 12px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span class="feat-name" style="font-size:0.85rem; opacity: 0.8;">• ${f.name}</span>
                                        <strong class="feat-val" style="font-size:0.85rem; color: ${color};">${f.value.toFixed(3)}%</strong>
                                    </div>
                                    <div class="bar-track" style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden;">
                                        <div class="bar-fill" style="width:${f.value}%; height:100%; background:${color}; opacity:0.8;"></div>
                                    </div>
                                </li>`;
                        });
                    } else {
                        fList.innerHTML = '<li style="color:#94a3b8; font-style:italic;">Data faktor dominan tidak tersedia.</li>';
                    }

                    resContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

                } else {
                    alert("Analisis Gagal: " + (res.error || "Gagal menghubungi server."));
                }
            } catch (err) {
                console.error("Fetch Error:", err);
                alert("Kesalahan koneksi ke server. Pastikan backend Flask berjalan.");
            } finally {
                btnPredict.innerHTML = originalText;
                btnPredict.disabled = false;
            }
        });
    }

    // --- 4. LOGIKA EXPORT PDF (BRUTE FORCE COLORING) ---
    // Logika ini menjamin PDF tidak kosong.
    const btnPdf = document.getElementById('btnPdf');
    if (btnPdf) {
        btnPdf.addEventListener('click', () => {
            const originalElement = document.getElementById('resultContainer');
            const originalBtnText = btnPdf.innerHTML;

            btnPdf.innerHTML = '⏳ Memproses PDF...';
            btnPdf.disabled = true;

            // 1. BUAT KLONING (Agar layar user tidak berubah)
            const clone = originalElement.cloneNode(true);
            
            // 2. LEMPAR KLONING KELUAR LAYAR (Supaya user tidak lihat)
            clone.style.position = 'absolute';
            clone.style.left = '-9999px'; 
            clone.style.top = '0';
            clone.style.width = '800px'; // Lebar A4 standar
            clone.style.background = '#ffffff'; // Kertas Putih Wajib!
            clone.style.padding = '40px';
            
            // 3. PAKSA SEMUA TEKS JADI HITAM (Looping ke semua elemen)
            // Ini kunci agar PDF tidak kosong!
            const allElements = clone.querySelectorAll('*');
            allElements.forEach(el => {
                // a. Jika elemen adalah grafik batang berwarna, JANGAN dihitamkan
                if (el.classList.contains('bar-fill')) {
                    return; // Biarkan warnanya (Merah/Hijau)
                }

                // b. Jika elemen adalah track (latar belakang grafik), kasih abu-abu
                if (el.classList.contains('bar-track')) {
                    el.style.background = '#e2e8f0';
                    return;
                }

                // c. Jika elemen adalah kartu, kasih border dan background putih
                if (el.classList.contains('result-header-card') || el.classList.contains('info-card')) {
                    el.style.background = '#f8fafc';
                    el.style.border = '1px solid #000000'; // Border hitam tegas
                    el.style.boxShadow = 'none';
                    el.style.color = '#000000';
                    return;
                }

                // d. UNTUK SEMUA TEXT LAINNYA -> HITAMKAN!
                // Cek computed style atau langsung timpa style
                el.style.color = '#000000';
                
                // Hapus background gelap jika ada
                if (el.style.background && !el.style.background.includes('rgb(255')) {
                     el.style.background = 'transparent';
                }
            });

            // Hapus tombol clone di dalam PDF
            const btnInClone = clone.querySelector('#btnPdf');
            if(btnInClone) btnInClone.remove();

            // Masukkan ke body (tak terlihat karena posisi -9999px)
            document.body.appendChild(clone);

            // 4. KONFIGURASI PDF
            const opt = {
                margin:       [0.5, 0.5],
                filename:     `Laporan_Diagnosa_${new Date().getTime()}.pdf`,
                image:        { type: 'jpeg', quality: 1.0 },
                html2canvas:  { 
                    scale: 2, 
                    useCORS: true,
                    backgroundColor: '#ffffff' 
                },
                jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' }
            };

            // 5. CETAK
            setTimeout(() => {
                html2pdf().set(opt).from(clone).save().then(() => {
                    // Bersihkan memori
                    document.body.removeChild(clone);
                    btnPdf.innerHTML = originalBtnText;
                    btnPdf.disabled = false;
                });
            }, 1000);
        });
    }

    // --- 5. LOGIKA RESET ---
    const btnReset = document.getElementById('resetBtn');
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            if (confirm("Reset data?")) {
                form.reset();
                if (bInput) bInput.value = '';
                resContainer.style.display = 'none';
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }
});