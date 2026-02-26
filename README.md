# 🛡️ DrowsyGuard AI Pro MAX
### **Advanced Driver Drowsiness Detection System with Deep Learning**

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red.svg)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/ericsonchandra99/DrowsyGuard-AI-Pro-lokal.git)

**DrowsyGuard AI Pro MAX** adalah aplikasi monitoring keselamatan berkendara berbasis kecerdasan buatan (AI). Proyek ini menggunakan arsitektur **MobileNetV2** yang dioptimalkan untuk mendeteksi kelelahan pengemudi melalui analisis visual secara *real-time*.

---

## 📺 Demo & Riset
* **Video Demo**: [Tonton di YouTube](https://youtu.be/9yV1miCpnIg)
* **Dataset**: YawDD (Yawning Detection Dataset)
* **Metode**: Analisis Temporal EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio)

---

## 🚀 Fitur Unggulan

Sistem ini dirancang dengan antarmuka **Streamlit** yang interaktif dan memiliki 3 modul utama:

### 1. 🎥 Live Monitor
* **Real-time Prediction**: Deteksi wajah langsung melalui webcam.
* **Buffer Smoothing**: Algoritma untuk menstabilkan prediksi agar tidak terjadi *flickering*.
* **Audio Alarm**: Peringatan suara otomatis (Windows-compatible) jika skor kantuk melebihi ambang batas.
* **Live Metrics**: Grafik skor kantuk yang terupdate setiap detik.

### 2. 🔍 Batch Analysis (Upload Predict)
* **Image Predictor**: Analisis instan untuk file gambar tunggal.
* **Video Processor**: Pengolahan file video untuk mendeteksi durasi kantuk di setiap frame secara otomatis.

### 3. 📊 History & Evidence Gallery
* **Auto-Save Evidence**: Mengambil tangkapan layar otomatis saat terdeteksi "Drowsy".
* **Interactive Gallery**: Melihat kembali bukti-bukti kejadian kantuk sebelumnya.
* **CSV Reporting**: Ekspor data laporan kejadian ke format `.csv` untuk analisis statistik.

---

## 🧠 Arsitektur Model & Metodologi

Aplikasi ini menggunakan pendekatan **Transfer Learning**:
* **Base Model**: MobileNetV2 (Pre-trained on ImageNet).
* **Optimization**: Adam Optimizer (Learning Rate: 0.0001).
* **Training Strategy**: 
    * *Offline Augmentation* untuk kondisi cahaya malam (gelap) dan siang (terang).
    * *Stratified Undersampling* untuk menangani ketidakseimbangan kelas data.
    * *Temporal Validation* (EAR/MAR) untuk akurasi pendeteksian microsleep.

---

## 🛠️ Instalasi

Pastikan Anda memiliki Python 3.9+ terinstal, lalu jalankan perintah berikut:

1. **Clone Repository**
   ```bash
   git clone [https://github.com/ericsonchandra99/DrowsyGuard-AI-Pro-lokal.git](https://github.com/ericsonchandra99/DrowsyGuard-AI-Pro-lokal.git)
   cd DrowsyGuard-AI-Pro-lokal

```

2. **Install Library**
```bash
pip install streamlit tensorflow opencv-python pandas numpy dlib

```


3. **Jalankan Aplikasi**
```bash
streamlit run code_deploy.py

```



---

## 👨‍💻 System Architect

**Ericson Chandra Sihombing**
*Data Science Student | ITERA '21*

> **"HAMORAON, HAGABEON, HASANGAPON"**
> Systematic Thinker | Leadership 📍 Lampung - Jakarta

* **LinkedIn**: [ericsonchandra99](https://www.google.com/search?q=https://www.linkedin.com/in/ericsonchandra99)
* **Instagram**: [@ericsonchandra99](https://www.google.com/search?q=https://instagram.com/ericsonchandra99)

---

## 📁 Struktur Project

```text
├── code_deploy.py          # Script utama aplikasi Streamlit
├── drowsy_model.keras      # Model Deep Learning (H5/Keras)
├── score.wav               # Sound alarm peringatan
├── evidence/               # Folder penyimpanan bukti (Auto-generated)
├── shape_predictor_68...   # File Dlib Landmark (opsional untuk ekstraksi)
└── README.md

```

---

© 2026 | **DrowsyGuard AI Pro MAX** | 

```

