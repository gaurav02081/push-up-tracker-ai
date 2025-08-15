🎥 **Demo Video** 
Watch the demo =  https://github.com/gaurav02081/push-up-tracker-ai/blob/main/assets/demo.mp4

# Push-Up Tracker

A computer vision-based push-up counter that uses MediaPipe for real-time pose detection and tracking. The application automatically counts your push-ups by monitoring your elbow angle and provides visual feedback.

## 🎯 Features

- **Real-time pose tracking** using MediaPipe
- **Automatic rep counting** based on elbow angle
- **Visual feedback** with progress bars and angle display
- **Session logging** to CSV file with timestamps and statistics
- **Optional face recognition** using InsightFace
- **Session summary popup** at the end of each workout
- **User-friendly interface** with clear visual indicators

## 📋 Prerequisites

Before you begin, ensure you have:

- **Windows 10/11** (tested on Windows)
- **Python 3.10 or higher** installed
- **Webcam** connected and working
- **Good lighting** for accurate pose detection
- **6-8 feet of space** from your camera

## 🚀 Installation Guide

### Step 1: Check Python Installation

First, verify that Python is installed on your system:

```bash
py --version
```

If you see a version number (e.g., "Python 3.10.0"), you're good to go. If not, download and install Python from [python.org](https://www.python.org/downloads/).

### Step 2: Download the Project

1. Download or clone this project to your computer
2. Open Command Prompt or PowerShell
3. Navigate to the project folder:
   ```bash
   cd path\to\your\project\folder
   ```

### Step 3: Install Dependencies

Install the required packages using these commands:

```bash
# Install core dependencies
py -m pip install opencv-python==4.8.1.78 mediapipe numpy==1.26.4

# Optional: Install face recognition (recommended)
py -m pip install onnxruntime insightface
```

**Note:** The specific versions ensure compatibility. If you encounter any errors, try installing without version numbers:
```bash
py -m pip install opencv-python mediapipe numpy
```

### Step 4: Test the Installation

Run a quick test to ensure everything is working:

```bash
py -c "import cv2, mediapipe, numpy; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!", you're ready to go!

## 🎮 How to Use

### Basic Usage

1. **Start the application:**
   ```bash
   py cool.py
   ```

2. **Position yourself:**
   - Stand about 6-8 feet from your camera
   - Make sure your full upper body is visible
   - Ensure good lighting (avoid backlighting)
   - Wear clothing that doesn't obscure your arms

3. **Start your workout:**
   - The app will automatically detect your pose
   - Begin doing push-ups
   - Watch the real-time feedback on screen

4. **End your session:**
   - Press 'q' to quit
   - Review your session summary
   - Your stats are automatically saved

### Understanding the Interface

- **Elbow Angle Display:** Shows your current elbow angle in degrees
- **Push-up Counter:** Displays your total reps
- **Progress Bars:** Visual indicators on the sides showing your position
- **Pose Overlay:** Skeleton lines showing detected body parts
- **User Recognition:** Shows recognized user name (if face recognition is set up)

### Controls

- **'q'** - Quit the application and show summary
- **Any other key** - Continue the session

## 👤 Face Recognition Setup (Optional)

To enable automatic user recognition:

1. **Create the faces folder structure:**
   ```
   faces/
   ├── YourName/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── photo3.jpg
   └── AnotherUser/
       ├── photo1.jpg
       └── photo2.jpg
   ```

2. **Add photos:**
   - Create a folder with your name inside the `faces/` directory
   - Add 2-3 clear photos of your face (different angles work best)
   - Supported formats: JPG, JPEG, PNG, BMP

3. **Test recognition:**
   - Run the application
   - Look at the camera for the first few seconds
   - Your name should appear in the top-right corner

## 📊 Understanding the Data

### Rep Counting Logic

- **Down position:** Elbow angle < 90°
- **Up position:** Elbow angle > 160°
- **One rep:** Complete cycle from up → down → up

### Session Data

The app automatically saves your workout data to `workout_log.csv`:

| Column | Description |
|--------|-------------|
| datetime | Date and time of the session |
| name | Recognized user name or "Unknown" |
| reps | Total push-ups completed |
| duration_sec | Session duration in seconds |
| pace_rpm | Reps per minute (pace) |

### Example CSV Output
```csv
datetime,name,reps,duration_sec,pace_rpm
2024-01-15 14:30:25,John,15,120.5,7.47
2024-01-15 16:45:12,John,20,180.2,6.66
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**Camera not working:**
- Make sure your webcam is connected and not being used by another application
- Try closing other video applications (Zoom, Teams, etc.)
- Check Windows camera permissions

**Poor pose detection:**
- Improve lighting (avoid shadows and backlighting)
- Ensure your full upper body is visible
- Wear clothing that doesn't obscure your arms
- Stand 6-8 feet from the camera

**Package installation errors:**
```bash
# If you get NumPy conflicts:
py -m pip uninstall numpy
py -m pip install numpy==1.26.4

# If OpenCV fails:
py -m pip install opencv-contrib-python
```

**Face recognition not working:**
- Ensure InsightFace is installed: `py -m pip install insightface`
- Check that your photos are clear and well-lit
- Make sure the folder structure is correct
- Try adding more photos of different angles

**Application crashes:**
- Update your graphics drivers
- Try running with a different camera
- Check Windows compatibility mode

### Performance Tips

- **Close unnecessary applications** to free up system resources
- **Use good lighting** for better detection accuracy
- **Position yourself correctly** - full upper body visible
- **Wear contrasting clothing** to your background

## 📁 Project Structure

```
push-up-tracker/
├── cool.py              # Main application
├── README.md            # This file
├── requirements.txt     # Dependencies list
├── workout_log.csv      # Session history (auto-generated)
└── faces/               # Face recognition photos
    └── YourName/
        ├── photo1.jpg
        ├── photo2.jpg
        └── photo3.jpg
```

## 🛠️ Technical Details

### Dependencies

- **OpenCV 4.8.1.78** - Computer vision and camera handling
- **MediaPipe 0.10.14** - Real-time pose estimation
- **NumPy 1.26.4** - Numerical computations
- **InsightFace 0.7.3** - Face recognition (optional)
- **ONNX Runtime** - Machine learning inference

### How It Works

1. **Pose Detection:** MediaPipe analyzes each frame to detect 33 body landmarks
2. **Angle Calculation:** Computes elbow angle from shoulder, elbow, and wrist positions
3. **Rep Counting:** State machine tracks up/down positions to count complete reps
4. **Face Recognition:** InsightFace creates embeddings from photos and matches them in real-time
5. **Data Logging:** Session statistics are saved to CSV for tracking progress

### System Requirements

- **CPU:** Any modern multi-core processor
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 100MB free space
- **Camera:** Any USB webcam or built-in camera
- **OS:** Windows 10/11 (tested), should work on macOS/Linux

## 🤝 Contributing

Feel free to improve this project by:
- Adding support for other exercises
- Improving the UI/UX
- Adding more detailed analytics
- Supporting different camera setups

## 📄 License

This project is open source. Feel free to use and modify for personal or educational purposes.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all prerequisites are met
3. Try reinstalling dependencies
4. Check that your camera and lighting are adequate

---

**Happy working out! 💪**
