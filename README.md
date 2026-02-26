# 🛡️ DrowsyGuard AI Pro MAX
### Advanced Driver Drowsiness Detection System with Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red.svg)](https://streamlit.io/)
[![Version](https://img.shields.io/badge/Version-2.5.0--PRO--MAX-success.svg)]()

---

## 🎥 Live Demo

<p align="center">
  <a href="https://youtu.be/9yV1miCpnIg">
    <img src="https://img.youtube.com/vi/9yV1miCpnIg/maxresdefault.jpg" width="80%">
  </a>
</p>

<p align="center">
  ▶ Click thumbnail above to watch full demo on YouTube
</p>

---

## 📌 Overview

**DrowsyGuard AI Pro MAX** adalah sistem monitoring keselamatan berkendara berbasis **Artificial Intelligence** yang mampu mendeteksi kondisi mengantuk secara **real-time** menggunakan Computer Vision dan Deep Learning.

Model menggunakan arsitektur **MobileNetV2 (Transfer Learning)** yang dioptimalkan untuk efisiensi dan akurasi tinggi dalam skenario dunia nyata.

Sistem ini mengintegrasikan:

- 🧠 Deep Learning (TensorFlow / Keras)
- 🎥 Computer Vision (OpenCV)
- 🌐 Interactive Web Dashboard (Streamlit)
- 📊 Real-time Risk Monitoring
- 📸 Automatic Evidence Capture
- 📑 CSV Reporting System

---

## 🚀 Core Features

### 🎥 1. Live Monitor
- Real-time webcam prediction
- Adjustable danger threshold
- Buffer smoothing (anti flicker)
- Live risk score chart
- Automatic alarm trigger (Windows-supported)
- Evidence auto-capture saat terdeteksi "Drowsy"

---

### 🔍 2. Batch Analysis (Upload Predict)
- 📷 Image prediction (single frame analysis)
- 🎞️ Video frame-by-frame processing
- Progress tracking
- Risk visualization overlay

---

### 📊 3. History & Evidence Gallery
- Auto-saved drowsiness evidence
- Interactive gallery view
- Download individual images
- Full report export (.CSV)

---

## 🧠 Model Architecture & Methodology

### 🔹 Base Model
- **MobileNetV2** (Pre-trained on ImageNet)

### 🔹 Optimization
- Adam Optimizer  
- Learning Rate: 0.0001  
- Input Size: 224x224  

### 🔹 Training Strategy
- Offline Augmentation (Low-light & daylight simulation)
- Stratified Undersampling (Class imbalance handling)
- Temporal Validation:
  - EAR (Eye Aspect Ratio)
  - MAR (Mouth Aspect Ratio)

---

## 🏗️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| TensorFlow / Keras | Deep Learning Model |
| OpenCV | Frame Processing |
| Streamlit | Web Application Interface |
| NumPy | Numerical Computation |
| Pandas | Data Logging & Reporting |
| Dlib (Optional) | Facial Landmark Extraction |

---

## 📂 Project Structure

```
DrowsyGuard-AI-Pro-MAX/
│
├── code_deploy.py          # Main Streamlit Application
├── drowsy_model.keras      # Deep Learning Model
├── score.wav               # Alarm Sound
├── evidence/               # Auto-generated evidence folder
├── shape_predictor_68.dat  # Dlib Landmark (Optional)
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ericsonchandra99/DrowsyGuard-AI-Pro-lokal.git
cd DrowsyGuard-AI-Pro-lokal
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install streamlit tensorflow opencv-python pandas numpy pillow dlib
```

---

### 4️⃣ Run Application

```bash
streamlit run code_deploy.py
```

Access via browser:

```
http://localhost:8501
```

---

## 🎯 Use Case Applications

- 🚗 Driver Monitoring System
- 🏭 Industrial Safety Supervision
- 🖥️ Focus Monitoring System
- 📚 Academic Research (Computer Vision)
- 🤖 AI-based Safety Implementation

---

## 📈 System Workflow

1. Capture frame (webcam / uploaded media)
2. Preprocessing (Resize + Normalization)
3. Deep Learning inference
4. Score smoothing using buffer
5. Threshold evaluation
6. If exceeded:
   - Trigger alarm
   - Save evidence image
   - Log report entry

---

## 👨‍💻 System Architect

**Ericson Chandra Sihombing**  
Data Science Student — Institut Teknologi Sumatera (ITERA)

> *"HAMORAON, HAGABEON, HASANGAPON"*  
> Systematic Thinker | Leadership  
> Lampung – Jakarta

---

## 🔮 Future Development

- Hybrid EAR + CNN Fusion Model
- Cloud Deployment
- Mobile Integration
- GPU Acceleration
- Real-world in-vehicle hardware integration

---

## 📜 License

This project is developed for academic research and portfolio purposes.

---

## ⭐ Support & Contribution

If you find this project useful:

- ⭐ Star this repository
- 🍴 Fork for development
- 💬 Provide feedback or suggestions

---

# 🛡️ AI for Safety. AI for Humanity.
