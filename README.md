# Hand Gesture Control System

A real-time computer vision system that enables hands-free control of your computer using hand gestures. Control media playback, navigate windows, move the mouse, and more—all through intuitive hand movements captured by your webcam.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

## ✨ Features

### 🖱️ Mouse Control
- **Point & Move**: Point with your index finger to control the cursor
- **Left Click**: Quick pinch gesture (thumb + index finger)
- **Right Click**: Pinch with middle finger extended
- **Drag & Drop**: Hold pinch for 0.25 seconds to start dragging
- **Scroll**: Two fingers up (index + middle) and move vertically

### 🎵 Media Control
- **Play/Pause**: Close fist (all fingers down)
- **Next Track**: Swipe right with pointing finger
- **Previous Track**: Swipe left with pointing finger
- **Volume Up/Down**: Rock sign (🤘) and move hand vertically
- **Mute**: OK sign (👌)
- **Seek Forward/Back**: Two-hand pinch and move horizontally
- **Fullscreen Toggle**: ILY sign (🤟)

### 🪟 System Navigation
- **Window Switch (Next)**: Three-finger swipe right
- **Window Switch (Previous)**: Three-finger swipe left
- **Open Spotify**: Call-me sign (🤙)

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Operating System: Linux, macOS, or Windows

## 🚀 Installation

### 1. Clone or Download

Download the appropriate script for your operating system:
- `gesture_control_linux.py` - For Linux
- `gesture_control_macos.py` - For macOS
- `gesture_control_windows.py` - For Windows

### 2. Install Dependencies

#### Common Dependencies (All Platforms)
```bash
pip install opencv-python mediapipe numpy pyautogui
```

#### Platform-Specific Dependencies

**Linux:**
```bash
# Install xdotool for system control
sudo apt install xdotool

# No additional Python packages needed
```

**macOS:**
```bash
# No additional dependencies needed
# AppleScript is built-in to macOS
```

**Windows:**
```bash
# Install keyboard library for media keys
pip install keyboard

# Note: Run the script as administrator for full media key support
```

### 3. Camera Permissions

Ensure your system allows Python/OpenCV to access your webcam:

- **macOS**: Grant camera permissions in System Preferences → Security & Privacy → Camera
- **Windows**: Check camera permissions in Settings → Privacy → Camera
- **Linux**: User should be in the `video` group (`sudo usermod -a -G video $USER`)

## 🎮 Usage

### Starting the Application

**Linux:**
```bash
python gesture_control_linux.py
```

**macOS:**
```bash
python gesture_control_macos.py
```

**Windows (as Administrator):**
```powershell
# Right-click Command Prompt/PowerShell and "Run as administrator"
python gesture_control_windows.py
```

### First Time Setup

1. Position yourself 2-3 feet from the webcam
2. Ensure good lighting (front-lit is best)
3. The white rectangle shows the active detection area—keep your hands inside
4. Green text at the top shows recognized actions
5. Press **'q'** to quit

## 🖐️ Gesture Reference Guide

### Mouse Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝ Point | Move Cursor | Index finger up, others down |
| 🤏 Pinch | Left Click | Touch thumb and index together briefly |
| 🤏 Pinch + Middle | Right Click | Pinch with middle finger extended |
| 🤏⏱ Hold Pinch | Drag | Hold pinch for 0.25+ seconds |
| ✌ Two Fingers | Scroll | Index + middle up, move vertically |

### Media Control Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| ✊ Fist | Play/Pause | All fingers closed |
| 👌 OK Sign | Mute | Thumb + index circle, others up |
| 🤘 Rock Sign | Volume | Index + pinky up, move vertically |
| 🤟 ILY Sign | Fullscreen | Thumb + index + pinky up |
| 👉 Point Swipe | Track Nav | Point and swipe left/right |
| 🤏🤏 Two Pinch | Seek | Both hands pinch, move horizontally |

### System Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| 👆👆👆 Three-Finger Swipe | Window Switch | Index + middle + ring, swipe left/right |
| 🤙 Call Me | Open Spotify | Thumb + pinky up, others down |

## ⚙️ Configuration

You can customize gesture sensitivity and behavior by editing the `Config` class in your platform-specific script:

```python
class Config:
    # Camera settings
    CAM_INDEX = 0                    # Change if you have multiple cameras
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    
    # Detection sensitivity
    DETECTION_MARGIN = 140           # Detection zone margin (pixels)
    MOUSE_SMOOTHING = 0.25          # Mouse smoothing (0-1, lower = smoother)
    
    # Gesture thresholds
    PINCH_THRESHOLD_DOWN = 0.040    # Pinch sensitivity
    STABLE_FRAMES_REQUIRED = 8      # Frames needed for stable detection
    COOLDOWN_SECONDS = 0.6          # Time between repeated actions
    
    # Swipe detection
    SWIPE_MIN_PIXELS = 170          # Minimum swipe distance
    SWIPE_MAX_DURATION = 0.45       # Maximum swipe duration (seconds)
    
    # Volume/Seeking
    VOLUME_STEP_PIXELS = 35         # Pixels per volume tick
    SEEK_MIN_PIXELS = 18            # Seeking sensitivity
```

## 🔧 Troubleshooting

### Camera Issues
**Problem**: Camera not detected or black screen
- Check if another application is using the camera
- Try changing `CAM_INDEX` in Config (0, 1, 2, etc.)
- Verify camera permissions

**Problem**: Low FPS or laggy performance
- Close other camera applications
- Reduce `CAM_WIDTH` and `CAM_HEIGHT` in Config
- Ensure good lighting to help hand detection

### Gesture Recognition Issues
**Problem**: Gestures not recognized
- Ensure hands are inside the white detection rectangle
- Keep hands well-lit and avoid backlight
- Adjust `STABLE_FRAMES_REQUIRED` (lower = more sensitive)
- Check `min_detection_confidence` in the Hands() initialization

**Problem**: Too many false detections
- Increase `STABLE_FRAMES_REQUIRED` for more stability
- Increase `COOLDOWN_SECONDS` to prevent rapid repeated actions
- Improve lighting and reduce background movement

### Platform-Specific Issues

**Linux:**
- If xdotool commands don't work, ensure it's installed: `sudo apt install xdotool`
- If Spotify doesn't open, verify it's installed and in your PATH

**macOS:**
- Grant accessibility permissions in System Preferences → Security & Privacy → Accessibility
- Some media keys may not work in all applications

**Windows:**
- Run as administrator for full media key functionality
- Install keyboard library: `pip install keyboard`
- Some antivirus software may block keyboard automation

## 🎯 Tips for Best Performance

1. **Lighting**: Position yourself facing a light source, not with light behind you
2. **Distance**: Stay 2-3 feet from the camera for optimal detection
3. **Background**: Plain, contrasting backgrounds work best
4. **Hand Position**: Keep gestures within the white detection rectangle
5. **Deliberate Movements**: Make gestures clearly and hold them briefly for recognition
6. **Calibration**: Adjust sensitivity settings in Config to match your environment

## 🛡️ Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Camera feed is only processed in real-time, not recorded
- Press 'q' at any time to immediately stop the application

## 📝 Technical Details

### Architecture
- **Computer Vision**: MediaPipe Hands for real-time hand landmark detection
- **Gesture Recognition**: Custom algorithms analyzing finger positions and movements
- **System Control**: Platform-specific libraries (xdotool, AppleScript, keyboard)
- **Mouse Control**: PyAutoGUI for cross-platform mouse automation

### Hand Landmarks
The system tracks 21 landmarks per hand:
- Wrist (0)
- Thumb (1-4)
- Index finger (5-8)
- Middle finger (9-12)
- Ring finger (13-16)
- Pinky (17-20)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional gesture recognition patterns
- Performance optimizations
- Support for more applications
- Custom gesture configuration UI
- Multi-language support

## License

MIT License - feel free to modify and distribute

## Acknowledgments

- **MediaPipe** by Google for hand tracking
- **OpenCV** for computer vision capabilities
- **PyAutoGUI** for cross-platform mouse control

## 🔄 Version History

### v1.0.0 (2025-01-03)
- Initial release
- Support for Linux, macOS, and Windows
- Mouse control, media playback, and system navigation
- Real-time hand gesture recognition
- Configurable sensitivity settings

---
