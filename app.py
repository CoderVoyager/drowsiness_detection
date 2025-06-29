import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import time
import threading
import pathlib
import platform
import sys

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    st.error(f"WebRTC components not available: {e}")
    WEBRTC_AVAILABLE = False

if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

REQUIRED_PACKAGES = {
    'torch': 'PyTorch',
    'cv2': 'OpenCV (opencv-python)',
    'numpy': 'NumPy',
    'PIL': 'Pillow'
}

missing_packages = []
for package, name in REQUIRED_PACKAGES.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(name)

if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.code("pip install " + " ".join([
        "torch torchvision",
        "opencv-python",
        "numpy",
        "pillow",
        "streamlit-webrtc",
        "av",
        "ultralytics",
        "seaborn",
        "matplotlib",
        "pandas",
        "scipy",
        "tqdm",
        "pyyaml",
        "requests",
        "thop",
        "psutil"
    ]))
    st.stop()

st.set_page_config(
    page_title="Drowsiness Detection App",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-awake {
        color: #28a745;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .status-drowsy {
        color: #dc3545;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #6c757d;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

lock = threading.Lock()
detection_result = {"status": "No Detection", "confidence": 0.0, "detections": []}

class DrowsinessDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                              path=self.model_path, force_reload=True)
                except Exception as e1:
                    st.warning(f"Primary loading method failed: {e1}")
                    try:
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                  path=self.model_path, force_reload=True, trust_repo=True)
                    except Exception as e2:
                        st.warning(f"Secondary loading method failed: {e2}")
                        try:
                            import sys
                            sys.path.append('yolov5')
                            from models.experimental import attempt_load
                            self.model = attempt_load(self.model_path, device=self.device)
                        except Exception as e3:
                            raise Exception(f"All loading methods failed. Last error: {e3}")
                
                self.model.to(self.device)
                st.success(f"✅ Model loaded successfully on {self.device.upper()}")
                st.info(f"📊 Model classes: {self.model.names}")
            else:
                st.error(f"❌ Model file not found at: {self.model_path}")
                st.info("💡 Make sure your model file path is correct. Common locations:")
                st.code("""
                - ./best.pt
                - /content/best.pt  (Google Colab)
                - ./models/best.pt
                """)
                self.model = None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            if "seaborn" in str(e):
                st.error("Missing seaborn dependency. Install with: pip install seaborn")
            elif "No module named" in str(e):
                module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
                st.error(f"Missing dependency: {module_name}. Install with: pip install {module_name}")
            st.info("💡 Try installing all dependencies with:")
            st.code("pip install -r requirements.txt")
            self.model = None
    
    def detect_image(self, image):
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            results = self.model(image)
            
            detections = []
            for *box, conf, cls in results.xyxy[0]:
                if conf > 0.3:
                    class_name = self.model.names[int(cls)]
                    detections.append({
                        'box': [int(x) for x in box],
                        'confidence': float(conf),
                        'class': class_name
                    })
            
            annotated_image = results.render()[0]
            
            return annotated_image, detections
            
        except Exception as e:
            return None, f"Detection error: {str(e)}"
    
    def detect_frame(self, frame):
        global detection_result
        
        if self.model is None:
            return frame
        
        try:
            results = self.model(frame)
            
            detections = []
            best_detection = None
            highest_conf = 0
            
            for *box, conf, cls in results.xyxy[0]:
                if conf > 0.3:
                    class_name = self.model.names[int(cls)]
                    detection = {
                        'box': [int(x) for x in box],
                        'confidence': float(conf),
                        'class': class_name
                    }
                    detections.append(detection)
                    
                    if conf > highest_conf:
                        highest_conf = conf
                        best_detection = detection
            
            with lock:
                if best_detection:
                    detection_result = {
                        "status": "DROWSY" if best_detection['class'] == 'drowsiness' else "AWAKE",
                        "confidence": best_detection['confidence'],
                        "detections": detections
                    }
                else:
                    detection_result = {
                        "status": "No Face Detected",
                        "confidence": 0.0,
                        "detections": []
                    }
            
            annotated_frame = self.draw_detections(frame.copy(), detections)
            return annotated_frame
            
        except Exception as e:
            st.error(f"Frame detection error: {str(e)}")
            return frame
    
    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            confidence = det['confidence']
            
            color = (0, 0, 255) if class_name == 'drowsiness' else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

@st.cache_resource
def load_detector(model_path):
    return DrowsinessDetector(model_path)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    if 'detector' in st.session_state and st.session_state.detector.model is not None:
        processed_img = st.session_state.detector.detect_frame(img)
    else:
        processed_img = img
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.markdown('<h1 class="main-header">😴 Drowsiness Detection System</h1>', unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="best.pt",
        help="Path to your trained YOLOv5 model file"
    )
    
    if st.sidebar.button("🔄 Load/Reload Model"):
        st.session_state.detector = load_detector(model_path)
    
    if 'detector' not in st.session_state:
        st.session_state.detector = load_detector(model_path)
    
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Live Video Detection", "🎞️ Recorded Video Detection"])

    with tab1:
        st.header("Upload Image for Drowsiness Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect drowsiness"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Detection Result")
                
                if st.session_state.detector.model is not None:
                    with st.spinner("Processing..."):
                        result_image, detections = st.session_state.detector.detect_image(image)
                    
                    if result_image is not None:
                        st.image(result_image, caption="Detection Result", use_column_width=True)
                        
                        if isinstance(detections, list) and detections:
                            st.success("✅ Detection completed!")
                            
                            for i, det in enumerate(detections):
                                status = "DROWSY" if det['class'] == 'drowsiness' else "AWAKE"
                                confidence = det['confidence']
                                
                                if status == "DROWSY":
                                    st.markdown(f'<p class="status-drowsy">🔴 Status: {status}</p>', unsafe_allow_html=True)
                                    st.error(f"⚠️ Drowsiness detected with {confidence:.1%} confidence!")
                                else:
                                    st.markdown(f'<p class="status-awake">🟢 Status: {status}</p>', unsafe_allow_html=True)
                                    st.success(f"✅ Person appears awake with {confidence:.1%} confidence")
                                
                                st.markdown(f'<p class="confidence-score">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No face detected in the image")
                    else:
                        st.error(f"❌ Detection failed: {detections}")
                else:
                    st.error("❌ Model not loaded. Please check the model path and reload.")
    
    with tab2:
        st.header("Live Video Drowsiness Detection")
        
        if not WEBRTC_AVAILABLE:
            st.error("❌ WebRTC components not available. Install with:")
            st.code("pip install streamlit-webrtc av")
            st.info("💡 You can still use the Image Detection feature above.")
            return
        
        if st.session_state.detector.model is not None:
            st.info("📹 Click 'START' to begin live detection. The system will analyze your webcam feed in real-time.")
            
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                webrtc_streamer(
                    key="drowsiness-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_configuration,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            
            with col2:
                st.subheader("🔍 Detection Status")
                
                status_placeholder = st.empty()
                confidence_placeholder = st.empty()
                alert_placeholder = st.empty()
                
                if st.button("🔄 Refresh Status"):
                    with lock:
                        current_result = detection_result.copy()
                    
                    status = current_result["status"]
                    confidence = current_result["confidence"]
                    
                    if status == "DROWSY":
                        status_placeholder.markdown(f'<p class="status-drowsy">🔴 {status}</p>', unsafe_allow_html=True)
                        confidence_placeholder.markdown(f'<p class="confidence-score">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                        alert_placeholder.error("⚠️ DROWSINESS ALERT!")
                    elif status == "AWAKE":
                        status_placeholder.markdown(f'<p class="status-awake">🟢 {status}</p>', unsafe_allow_html=True)
                        confidence_placeholder.markdown(f'<p class="confidence-score">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                        alert_placeholder.success("✅ Alert and focused")
                    else:
                        status_placeholder.markdown(f'<p>🔍 {status}</p>', unsafe_allow_html=True)
                        confidence_placeholder.markdown('<p class="confidence-score">Confidence: N/A</p>', unsafe_allow_html=True)
                        alert_placeholder.info("📷 Looking for face...")
                
                auto_refresh = st.checkbox("🔄 Auto-refresh status (every 2 seconds)")
                
                if auto_refresh:
                    time.sleep(2)
                    st.experimental_rerun()
        
        else:
            st.error("❌ Model not loaded. Please check the model path and reload.")
    
    with tab3:
        st.header("Recorded Video Drowsiness Detection")
        uploaded_video = st.file_uploader("Upload a recorded video file", type=['mp4', 'avi', 'mov'])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            st.video(video_path)

            if st.button("🚀 Run Detection on Video"):
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                out_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                stframe = st.empty()

                frame_count = 0
                with st.spinner("Processing video..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if st.session_state.detector.model is not None:
                            processed_frame = st.session_state.detector.detect_frame(frame)
                        else:
                            processed_frame = frame

                        out.write(processed_frame)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, channels="RGB", caption=f"Frame {frame_count}")
                        frame_count += 1

                    cap.release()
                    out.release()
                
                st.success("✅ Video processed successfully!")
                st.video(out_path)

    st.markdown("---")
    st.markdown("""
    ### 📝 Instructions:
    - **Image Detection**: Upload an image to detect drowsiness in a static photo
    - **Live Video**: Use your webcam for real-time drowsiness monitoring
    - **Model Requirements**: Ensure your `best.pt` file is a trained YOLOv5 model for drowsiness detection
    - **Classes**: The model should detect 'drowsiness' and 'awake' states
    """)
    
    with st.expander("🔧 Technical Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Device:**", st.session_state.detector.device.upper() if 'detector' in st.session_state else "Not loaded")
            st.write("**Model Path:**", model_path)
        with col2:
            if 'detector' in st.session_state and st.session_state.detector.model:
                st.write("**Model Classes:**", st.session_state.detector.model.names)
            st.write("**Confidence Threshold:**", "30%")

if __name__ == "__main__":
    main()