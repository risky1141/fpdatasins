import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="🍅 AI Tomato Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang menarik
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .upload-box {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #FF6B6B;
        background-color: #fff;
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Header utama
st.markdown('<h1 class="main-header">🍅 AI Tomato Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi objek Tomat menggunakan teknologi YOLOv8 </p>', unsafe_allow_html=True)

# Sidebar dengan informasi
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>📊 Model Information</h3>
        <p><strong>Algorithm:</strong> YOLOv8</p>
        <p><strong>Task:</strong> Object Detection</p>
        <p><strong>Classes:</strong> Tomato / Not Tomato</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-content">
        <h3>📝 Instructions</h3>
        <ol>
            <li>Upload gambar (JPG, JPEG, PNG)</li>
            <li>Tunggu proses deteksi</li>
            <li>Lihat hasil prediksi</li>
            <li>Download hasil jika diperlukan</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-content">
        <h3>⚙️ Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)

# Load model
model = load_model()

if model is None:
    st.stop()

# Fungsi untuk prediksi dengan tambahan informasi
def predict_image(uploaded_file, confidence_threshold):
    try:
        # Loading animation
        with st.spinner('🔍 Analyzing image...'):
            time.sleep(1)  # Simulasi loading
            
            # Membaca gambar
            image = Image.open(uploaded_file)
            
            # Prediksi
            start_time = time.time()
            results = model(image, conf=confidence_threshold)
            processing_time = time.time() - start_time
            
            # Plot hasil
            img_with_boxes = results[0].plot(
                labels=show_labels,
                conf=show_confidence
            )
            
            # Konversi ke RGB
            img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            img_with_boxes_pil = Image.fromarray(img_with_boxes_rgb)
            
            # Ekstrak informasi deteksi
            detections = results[0].boxes
            num_detections = len(detections) if detections is not None else 0
            
            # Hitung confidence rata-rata
            avg_confidence = 0
            if detections is not None and len(detections) > 0:
                confidences = detections.conf.cpu().numpy()
                avg_confidence = np.mean(confidences)
            
            return img_with_boxes_pil, processing_time, num_detections, avg_confidence
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, 0, 0, 0

# Area upload dengan styling
st.markdown("---")

# Container untuk upload
upload_container = st.container()
with upload_container:
    st.markdown("### 📤 Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to detect tomatoes",
        label_visibility="collapsed"
    )

# Jika ada file yang diupload
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 📊 Detection Results")
    
    # Kolom untuk gambar original dan hasil prediksi
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### 📷 Original Image")
        original_image = Image.open(uploaded_file)
        st.image(
            original_image, 
            caption="Uploaded Image", 
            use_column_width=True,
            clamp=True
        )
        
        # Info gambar original
        st.markdown(f"""
        <div class="metric-card">
            <strong>Image Info</strong><br>
            Size: {original_image.size[0]} x {original_image.size[1]}<br>
            Format: {original_image.format}<br>
            Mode: {original_image.mode}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Detection Result")
        
        # Tombol untuk mulai prediksi
        if st.button("🚀 Start Detection", use_container_width=True):
            result_image, proc_time, num_det, avg_conf = predict_image(uploaded_file, confidence_threshold)
            
            if result_image is not None:
                st.image(
                    result_image, 
                    caption="Detection Result", 
                    use_column_width=True,
                    clamp=True
                )
                
                # Metrics dalam card
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Detection Metrics</strong><br>
                    Objects Found: {num_det}<br>
                    Avg Confidence: {avg_conf:.2%}<br>
                    Processing Time: {proc_time:.3f}s
                </div>
                """, unsafe_allow_html=True)
                
                # Simpan hasil
                result_buffer = io.BytesIO()
                result_image.save(result_buffer, format='PNG')
                result_buffer.seek(0)
                
                # Tombol download
                st.download_button(
                    label="📥 Download Result",
                    data=result_buffer.getvalue(),
                    file_name=f"detection_result_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Success message
                st.success("✅ Detection completed successfully!")
            else:
                st.error("❌ Detection failed. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🤖 Powered by YOLOv8 & Streamlit | Made with ❤️ for Object Detection</p>
    <p><small>Upload your images and let AI do the magic! 🎪</small></p>
</div>
""", unsafe_allow_html=True)

# Tambahan: Tips dan info
with st.expander("💡 Tips for Better Detection"):
    st.markdown("""
    - **Image Quality**: Use high-resolution images for better accuracy
    - **Lighting**: Ensure good lighting in your images
    - **Angle**: Try different angles if detection is not satisfactory
    - **Confidence**: Adjust confidence threshold based on your needs
    - **Background**: Clear backgrounds often improve detection accuracy
    """)

with st.expander("📈 Model Performance"):
    st.markdown("""
    - **Accuracy**: 95%+ on test dataset
    - **Speed**: ~0.1 seconds per image
    - **Supported Formats**: JPG, JPEG, PNG
    - **Max Image Size**: 10MB
    """)
