import streamlit as st
import os
import platform
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import tempfile
from collections import deque
from datetime import datetime
import winsound
import zipfile
import io
from PIL import Image

# ===============================
# CONFIG & INITIALIZATION
# ===============================
st.set_page_config(
    page_title="DrowsyGuard AI Pro MAX",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

MODEL_NAME = "drowsy_model.keras"
SOUND_NAME = "score.wav"
SOUND_PATH = os.path.join(BASE_DIR, SOUND_NAME)

# List kemungkinan nama file foto profil Anda
PROFILE_NAMES = ["fotosaya.jpeg", "fotosaya.jpg", "fotosaya.JPG", "FOTOSAYA.JPG"]

# ===============================
# SESSION STATE
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False
if "score_history" not in st.session_state:
    st.session_state.score_history = []

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_ai_model():
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        # Menggunakan compile=False agar lebih cepat saat loading
        return tf.keras.models.load_model(model_path, compile=False)
    return None

model = load_ai_model()

# ===============================
# STYLING
# ===============================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title-text {
        text-align:center; font-size:45px; font-weight:800;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .profile-card {
        padding: 20px; border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.3);
        margin-bottom: 20px;
    }
    .metric-box {
        padding: 15px; border-radius: 12px; text-align: center;
        font-size: 20px; font-weight: bold; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# UTILS
# ===============================
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def save_evidence(frame):
    filename = f"drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(path, frame)
    return filename

def start_alarm():
    if not st.session_state.alarm_active and platform.system() == "Windows":
        if os.path.exists(SOUND_PATH):
            st.session_state.alarm_active = True
            winsound.PlaySound(SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

def stop_alarm():
    if st.session_state.alarm_active and platform.system() == "Windows":
        winsound.PlaySound(None, winsound.SND_PURGE)
        st.session_state.alarm_active = False

# ===============================
# SIDEBAR / CONTROL PANEL
# ===============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.header("⚙️ Control Panel")
    
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65, help="Batas persentase untuk memicu alarm")
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5, help="Jumlah frame untuk rata-rata (menghindari flickering)")
    alarm_enabled = st.toggle("🔔 Enable Alarm Sound", True)

    st.divider()
    st.subheader("🗂 Report Management")
    if st.button("🗑️ Clear All Reports", use_container_width=True):
        st.session_state.report_data = []
        for f in os.listdir(EVIDENCE_DIR):
            os.remove(os.path.join(EVIDENCE_DIR, f))
        st.rerun()

# ===============================
# HEADER & PROFILE SECTION
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

col_p1, col_p2 = st.columns([1, 3])
with col_p1:
    found_profile = next((os.path.join(BASE_DIR, n) for n in PROFILE_NAMES if os.path.exists(os.path.join(BASE_DIR, n))), None)
    if found_profile:
        st.image(found_profile, width=200, caption="System Architect")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200, caption="Developer Photo Not Found")

with col_p2:
    st.markdown(f"""
    <div class="profile-card">
        <h2 style="margin:0; color:#00f2fe;">Ericson Chandra Sihombing</h2>
        <p style="margin:5px 0;">🎓 <b>Data Science Student</b> — Institut Teknologi Sumatera (ITERA)</p>
        <p style="margin:5px 0;">🤖 <b>Project:</b> Deep Learning-based Drowsiness Detection System</p>
        <p style="font-size: 0.9em; opacity: 0.8;">Status: <span style="color:#00ff00;">● Online</span> | Ver: 2.5.0-PRO-MAX</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# MAIN TABS
# ===============================
# Perbaikan Error: Nama variabel tab disamakan
tab1, tab2, tab3 = st.tabs(["🎥 Live Monitor", "📤 Upload Predict", "📊 History Reports"])

# -------------------------------
# TAB 1: LIVE MONITOR
# -------------------------------
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        btn_start = st.button("▶ Start Camera", use_container_width=True)
        btn_stop = st.button("⏹ Stop Camera", use_container_width=True)
        frame_placeholder = st.empty()
        
        if btn_start: st.session_state.camera_running = True
        if btn_stop: 
            st.session_state.camera_running = False
            stop_alarm()

    with c2:
        st.subheader("Real-time Metrics")
        status_placeholder = st.empty()
        chart_placeholder = st.empty()

    if st.session_state.camera_running:
        if model is None:
            st.error("Model file not found! Please check 'drowsy_model.keras'.")
        else:
            cap = cv2.VideoCapture(0)
            buffer = deque(maxlen=smoothing)
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret: break
                
                # Logic AI
                preds = model.predict(preprocess_frame(frame), verbose=0)[0]
                buffer.append(preds[0])
                score = np.mean(buffer) * 100
                st.session_state.score_history.append(score)
                if len(st.session_state.score_history) > 50: st.session_state.score_history.pop(0)
                
                # Visuals
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                is_danger = score >= threshold
                bg_color = "#ff4b4b" if is_danger else "#00cc88"
                lbl_text = "⚠️ DROWSY" if is_danger else "✅ ALERT"
                
                status_placeholder.markdown(f"""
                    <div class='metric-box' style='background:{bg_color};'>
                        {lbl_text}<br><span style='font-size:30px;'>{score:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                
                chart_placeholder.line_chart(st.session_state.score_history)
                
                # Action
                if is_danger:
                    if alarm_enabled: start_alarm()
                    # Log only every second change
                    now = datetime.now().strftime("%H:%M:%S")
                    if not st.session_state.report_data or st.session_state.report_data[-1]["Timestamp"] != now:
                        fname = save_evidence(frame)
                        # "Evidence" digunakan agar sesuai dengan bagian Galeri di Tab 3
                        st.session_state.report_data.append({"Timestamp": now, "Risk": f"{score:.1f}%", "Evidence": fname})
                else:
                    stop_alarm()
            
            cap.release()
            frame_placeholder.info("Camera Stopped.")

# -------------------------------
# TAB 2: UPLOAD PREDICT
# -------------------------------
with tab2:
    st.subheader("🔍 Batch Analysis (Image/Video)")
    up_col1, up_col2 = st.columns(2)
    
    with up_col1:
        img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="img_up")
        if img_file and model:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Prediksi Gambar
            p = model.predict(preprocess_frame(img), verbose=0)[0][0] * 100
            
            # Tampilkan Hasil
            st.image(img, channels="BGR", caption=f"Hasil Analisis: {p:.1f}%")
            if p >= threshold:
                st.error(f"⚠️ Terdeteksi Mengantuk! ({p:.1f}%)")
            else:
                st.success(f"✅ User Terlihat Segar/Alert ({p:.1f}%)")

    with up_col2:
        vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="vid_up")
        if vid_file and model:
            # Simpan video ke file sementara karena OpenCV tidak bisa baca bytes langsung
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_file.read())
            
            v_cap = cv2.VideoCapture(tfile.name)
            
            # Placeholder untuk tampilan streaming video dan teks status
            video_display = st.empty()
            status_display = st.empty()
            progress_bar = st.progress(0)
            
            total_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            st.info("Sedang memproses video... Mohon tunggu.")
            
            while v_cap.isOpened():
                ret, frame = v_cap.read()
                if not ret:
                    break
                
                # Preprocessing dan Prediksi per frame
                input_data = preprocess_frame(frame)
                preds = model.predict(input_data, verbose=0)[0][0] * 100
                
                # Visualisasi pada frame
                color = (0, 0, 255) if preds >= threshold else (0, 255, 0)
                label = f"DROWSY: {preds:.1f}%" if preds >= threshold else f"SAFE: {preds:.1f}%"
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                
                # Tampilkan frame ke Streamlit
                video_display.image(frame, channels="BGR", use_container_width=True)
                
                # Update status
                if preds >= threshold:
                    status_display.markdown(f"<h3 style='color:red; text-align:center;'>STATUS: MENGANTUK ({preds:.1f}%)</h3>", unsafe_allow_html=True)
                else:
                    status_display.markdown(f"<h3 style='color:green; text-align:center;'>STATUS: AMAN ({preds:.1f}%)</h3>", unsafe_allow_html=True)
                
                # Update Progress Bar
                current_frame += 1
                if total_frames > 0:
                    progress_bar.progress(current_frame / total_frames)

            v_cap.release()
            tfile.close()
            if os.path.exists(tfile.name):
                os.remove(tfile.name) # Hapus file sementara setelah selesai
            
            st.success("Analisis Video Selesai!")

# -------------------------------
# TAB 3: HISTORY & GALLERY
# -------------------------------
# Perbaikan: Menggunakan variabel tab3 yang sudah didefinisikan
with tab3:
    if st.session_state.report_data:
        st.subheader("📸 Galeri Bukti Mengantuk")
        
        # Gallery Grid
        display_items = st.session_state.report_data[-8:] # 8 terakhir
        rows = st.columns(4)
        for i, item in enumerate(reversed(display_items)):
            with rows[i % 4]:
                # Menggunakan key "Evidence" sesuai data yang disimpan di Tab 1
                img_path = os.path.join(EVIDENCE_DIR, item["Evidence"])
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Pukul: {item['Timestamp']} ({item['Risk']})")
                    # Tombol Download Gambar Individu
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Foto",
                            data=f,
                            file_name=f"Drowsy_{item['Timestamp']}.jpg",
                            mime="image/jpeg",
                            key=f"dl_{i}_{item['Evidence']}"
                        )
        
        st.divider()
        st.subheader("📑 Laporan Data (CSV)")
        df = pd.DataFrame(st.session_state.report_data)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Export Full Report CSV", df.to_csv(index=False).encode('utf-8'), "laporan_mengantuk.csv", "text/csv")
    else:
        st.info("Belum ada aktivitas terdeteksi.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(f"<center>© {datetime.now().year} | <b>Ericson Chandra Sihombing</b> | ITERA Data Science Project</center>", unsafe_allow_html=True)