import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

@st.cache_resource
def load_model():
    """Load the YOLOv5 model"""
    try:
        # Suppress torch hub warnings
        import logging
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path='best.pt',
            force_reload=True,
            verbose=False
        )
        model.conf = 0.25  # Set confidence threshold
        model.iou = 0.45   # Set IoU threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_drowsiness_image(model, image):
    """Detect drowsiness in a single image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Make prediction
        results = model(img_array)
        
        # Render results
        rendered_img = results.render()[0]
        
        # Get detection results - Updated for newer YOLOv5 versions
        detections = []
        if hasattr(results, 'pandas'):
            detections = results.pandas().xyxy[0]
        elif hasattr(results, 'pred') and len(results.pred[0]) > 0:
            # For newer versions, extract predictions manually
            pred = results.pred[0]
            for detection in pred:
                x1, y1, x2, y2, conf, cls = detection.tolist()
                detections.append({
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'confidence': conf, 'class': int(cls), 'name': results.names[int(cls)]
                })
        
        return rendered_img, detections
    except Exception as e:
        st.error(f"Error in image detection: {str(e)}")
        return None, None

def detect_drowsiness_video(model, video_path):
    """Process video file for drowsiness detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return None, None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default fps if detection fails
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output video file with proper codec
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if VideoWriter is opened successfully
        if not out.isOpened():
            st.error("Could not open video writer")
            cap.release()
            return None, None
        
        # Process frames
        frame_count = 0
        drowsy_detections = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Make detection
                results = model(frame)
                rendered_frame = results.render()[0]
                
                # Ensure frame is in correct format
                if rendered_frame.dtype != np.uint8:
                    rendered_frame = rendered_frame.astype(np.uint8)
                
                # Write frame to output video
                out.write(rendered_frame)
                
                # Store detection results
                detections = []
                if hasattr(results, 'pandas'):
                    detections = results.pandas().xyxy[0]
                elif hasattr(results, 'pred') and len(results.pred[0]) > 0:
                    detections = results.pred[0]
                
                if len(detections) > 0:
                    drowsy_detections.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'detections': len(detections)
                    })
                
                frame_count += 1
                if total_frames > 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
            except Exception as e:
                st.warning(f"Error processing frame {frame_count}: {str(e)}")
                continue
        
        cap.release()
        out.release()
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Verify output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, drowsy_detections
        else:
            st.error("Output video file was not created properly")
            return None, None
    
    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        return None, None

def camera_capture_thread(model, frame_queue, stop_event):
    """Thread function for camera capture"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        frame_queue.put(("error", "Could not open camera"))
        return
    
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Make detection
            results = model(frame)
            rendered_frame = results.render()[0]
            
            # Convert BGR to RGB for display
            rendered_frame_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            
            # Get detection results
            detections = []
            if hasattr(results, 'pandas'):
                detections = results.pandas().xyxy[0]
            elif hasattr(results, 'pred') and len(results.pred[0]) > 0:
                detections = results.pred[0]
            
            # Put frame and detection info in queue
            frame_queue.put(("frame", rendered_frame_rgb, detections))
            
            frame_count += 1
            
        except Exception as e:
            frame_queue.put(("error", f"Detection error: {str(e)}"))
            continue
    
    cap.release()
    frame_queue.put(("stop", None))

def main():
    st.title("😴 Drowsiness Detection System")
    st.markdown("---")
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading YOLOv5 model..."):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Failed to load model. Please check if 'best.pt' file exists in the same directory.")
        return
    
    st.success("Model loaded successfully!")
    
    # Sidebar for navigation
    st.sidebar.title("Detection Options")
    option = st.sidebar.selectbox(
        "Choose detection method:",
        ["Image Detection", "Video Detection", "Live Camera Detection"]
    )
    
    if option == "Image Detection":
        st.header("📸 Image Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Detect drowsiness
            with st.spinner("Detecting drowsiness..."):
                result_img, detections = detect_drowsiness_image(st.session_state.model, image)
            
            if result_img is not None:
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_img, use_container_width=True)
                
                # Display detection summary
                if len(detections) > 0:
                    st.subheader("Detection Summary")
                    st.write(f"Total detections: {len(detections)}")
                    
                    if isinstance(detections, list):
                        # Handle list format
                        for idx, detection in enumerate(detections):
                            confidence = detection.get('confidence', 0)
                            class_name = detection.get('name', 'Unknown')
                            st.write(f"- {class_name}: {confidence:.2f}% confidence")
                    else:
                        # Handle DataFrame format
                        for idx, detection in detections.iterrows():
                            confidence = detection.get('confidence', 0)
                            class_name = detection.get('name', 'Unknown')
                            st.write(f"- {class_name}: {confidence:.2f}% confidence")
                else:
                    st.info("No drowsiness detected in the image.")
    
    elif option == "Video Detection":
        st.header("🎥 Video Detection")
        
        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm']
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_video)
            
            if st.button("Process Video", key="process_video"):
                with st.spinner("Processing video... This may take a while."):
                    output_path, detections = detect_drowsiness_video(st.session_state.model, video_path)
                
                if output_path and os.path.exists(output_path):
                    st.subheader("Processed Video")
                    
                    # Read and display processed video
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Provide download link
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                    
                    # Display detection summary
                    if detections:
                        st.subheader("Detection Summary")
                        st.write(f"Total frames with detections: {len(detections)}")
                        
                        # Create a simple chart of detections over time
                        if len(detections) > 0:
                            import pandas as pd
                            chart_data = pd.DataFrame({
                                'Time (seconds)': [d['time'] for d in detections],
                                'Detections': [d['detections'] for d in detections]
                            })
                            st.line_chart(chart_data.set_index('Time (seconds)'))
                    else:
                        st.info("No drowsiness detected in the video.")
                else:
                    st.error("Failed to process video. Please try with a different video format.")
                
                # Clean up temporary files
                try:
                    os.unlink(video_path)
                    if output_path and os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass
    
    elif option == "Live Camera Detection":
        st.header("📹 Live Camera Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use a toggle button instead of separate start/stop buttons
            camera_toggle = st.checkbox("Enable Live Camera Detection", value=st.session_state.camera_active)
            
            if camera_toggle != st.session_state.camera_active:
                st.session_state.camera_active = camera_toggle
                if not camera_toggle:
                    # Force page rerun to stop camera
                    st.rerun()
        
        with col2:
            st.info("Note: Live camera detection requires proper camera permissions. Make sure your browser allows camera access.")
        
        if st.session_state.camera_active:
            st.subheader("Live Detection")
            
            # Add a stop button that's always visible during detection
            if st.button("🛑 Stop Detection", key="emergency_stop"):
                st.session_state.camera_active = False
                st.rerun()
            
            # Create placeholders for video and detection info
            video_placeholder = st.empty()
            detection_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize camera and start detection
            try:
                cap = cv2.VideoCapture(0)
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    st.error("Could not open camera. Please check camera permissions.")
                    st.session_state.camera_active = False
                else:
                    status_placeholder.success("📹 Camera is active")
                    frame_count = 0
                    detection_history = []
                    
                    # Create a container for the camera feed
                    camera_container = st.container()
                    
                    while st.session_state.camera_active:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Failed to read from camera")
                            break
                        
                        try:
                            # Make detection
                            results = st.session_state.model(frame)
                            rendered_frame = results.render()[0]
                            
                            # Convert BGR to RGB for display
                            rendered_frame_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frame
                            with camera_container:
                                video_placeholder.image(rendered_frame_rgb, channels="RGB", use_container_width=True)
                            
                            # Get detection results
                            detections = []
                            if hasattr(results, 'pandas'):
                                try:
                                    detections = results.pandas().xyxy[0]
                                except:
                                    pass
                            elif hasattr(results, 'pred') and len(results.pred[0]) > 0:
                                detections = results.pred[0]
                            
                            # Update detection info
                            if len(detections) > 0:
                                max_confidence = 0
                                try:
                                    if hasattr(detections, 'confidence'):
                                        max_confidence = detections['confidence'].max()
                                    elif hasattr(detections, 'shape') and len(detections.shape) > 0:
                                        max_confidence = detections[:, 4].max()
                                    elif isinstance(detections, (list, tuple)) and len(detections) > 0:
                                        max_confidence = max([d.get('confidence', 0) for d in detections])
                                except:
                                    max_confidence = 0.5  # Default value if extraction fails
                                
                                detection_history.append({
                                    'frame': frame_count,
                                    'detections': len(detections),
                                    'confidence': max_confidence
                                })
                                
                                detection_placeholder.warning(
                                    f"⚠️ Drowsiness detected! "
                                    f"Confidence: {max_confidence:.2f}% | Frame: {frame_count}"
                                )
                            else:
                                detection_placeholder.success("✅ Alert and focused")
                            
                            frame_count += 1
                            
                            # Small delay to prevent overwhelming the system
                            time.sleep(0.03)  # ~30 FPS
                            
                        except Exception as e:
                            st.warning(f"Detection error on frame {frame_count}: {str(e)}")
                            continue
                
                cap.release()
                status_placeholder.info("📹 Camera stopped")
                
            except Exception as e:
                st.error(f"Error with camera: {str(e)}")
                st.session_state.camera_active = False
        
        else:
            st.info("Click the checkbox above to start live camera detection")
    
    # Footer
    st.markdown("---")
    st.markdown("**Drowsiness Detection System** - Stay alert, stay safe! 🚗")

if __name__ == "__main__":
    main()