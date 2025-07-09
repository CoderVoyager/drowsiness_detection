# 😴 Drowsiness Detection System

A real-time drowsiness detection system built with YOLOv5 and Streamlit that can analyze images, live video feeds, and recorded videos to detect drowsiness in drivers or individuals.

## 🚀 Features

- **📷 Image Detection**: Upload images to detect drowsiness in static photos
- **🎥 Live Video Detection**: Real-time drowsiness monitoring using webcam
- **🎞️ Recorded Video Detection**: Process pre-recorded videos for drowsiness analysis
- **🔄 Real-time Processing**: Instant detection with confidence scores
- **📊 Visual Feedback**: Color-coded status indicators and bounding boxes
- **⚙️ Configurable**: Easy model path configuration and reloading

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Webcam (for live detection)
- GPU (optional, for faster processing)

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
streamlit>=1.28.0
streamlit-webrtc>=0.47.0
av>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
ultralytics>=8.0.0
yolov5>=7.0.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/CoderVoyager/drowsiness_detection.git
cd drowsiness_detection
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv drowsiness_env

# Activate virtual environment
# On Windows:
drowsiness_env\Scripts\activate
# On macOS/Linux:
source drowsiness_env/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip
pip install -r requirements.txt

```
## 📁 Project Structure

```
hacksagon/
├── app.py                  # Main Streamlit application
├── best.pt                 # YOLOv5 trained model (you need to provide this)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── drowsiness_env/        # Virtual environment (after setup)
```

## 🎯 Model Requirements

You need a trained YOLOv5 model (`best.pt`) that can detect:
- **drowsiness**: When a person appears drowsy/sleepy
- **awake**: When a person appears alert and awake

### Training Your Own Model
1. Collect and label images of drowsy and awake faces
2. Train using YOLOv5 framework
3. Save the trained model as `best.pt`
4. Place it in the project root directory

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

#### 1. **Image Detection Tab**
- Upload an image (JPG, JPEG, PNG)
- View original and processed images side by side
- Get drowsiness detection results with confidence scores

#### 2. **Live Video Detection Tab**
- Click "START" to begin webcam detection
- Real-time status updates with color-coded indicators
- Auto-refresh option for continuous monitoring

#### 3. **Recorded Video Detection Tab**
- Upload a video file (MP4, AVI, MOV)
- Process entire video with frame-by-frame analysis
- Download processed video with detection annotations

### Configuration
- **Model Path**: Configure the path to your trained model
- **Confidence Threshold**: Currently set to 30% (can be modified in code)
- **Device**: Automatically detects GPU/CPU usage

## 📊 Detection Results

### Status Indicators
- 🟢 **AWAKE**: Person appears alert and focused
- 🔴 **DROWSY**: Drowsiness detected - alert triggered

### Confidence Scores
- Percentage confidence for each detection
- Visual bounding boxes around detected faces
- Color-coded annotations (Red for drowsy, Green for awake)

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Loading**
   ```
   Error: Model file not found at: best.pt
   ```
   **Solution**: Ensure your trained model file is in the correct path

2. **WebRTC Not Available**
   ```
   Error: WebRTC components not available
   ```
   **Solution**: Install missing dependencies
   ```bash
   pip install streamlit-webrtc av
   ```

3. **CUDA/GPU Issues**
   ```
   Error: CUDA not available
   ```
   **Solution**: Install CPU-only PyTorch
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Permission Errors**
   - Ensure webcam permissions are granted
   - Check antivirus software blocking camera access

### Performance Tips
- Use GPU for faster processing
- Reduce image resolution for better performance
- Close other applications using the camera

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


