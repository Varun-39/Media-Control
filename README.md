# 📺 Smart Media Controller (ML-Powered HCI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-0.10.9-green?logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated, touchless Human-Computer Interaction (HCI) system that uses machine learning to control media playback through hand gestures. Optimized for low-latency performance on edge devices like the NVIDIA Jetson Nano.

---

## 🚀 Key Features

- **Visual Wake-up Pattern**: Intelligent state management. The system stays idle until it detects an "Open Palm" gesture, conserving resources.
- **Dynamic Mute**: Instant audio control using a "Fist" gesture.
- **Pinch-to-Toggle**: Seamlessly Play/Pause media using a thumb-index pinch.
- **Rotary Volume Control**: A virtual "knob" using the angle of your index finger to precisely adjust volume levels.
- **Real-Time HUD**: Embedded visual feedback on the video feed showing system state, detected gestures, and command execution.

---

## 🛠️ Technical Architecture

### Machine Learning Pipeline
- **Hand Tracking**: Leverages Google's **Mediapipe** for high-fidelity hand landmark detection (21 points per hand).
- **Custom Classifiers**: Mathematical models built on top of raw landmarks to classify gestures (Fist, Open Palm, Pinch, Rotary Angle).
- **State Machine Strategy**: Implements an `ACTIVE` vs `IDLE` state machine to prevent accidental triggers and reduce CPU load during inactivity.

### Tech Stack
- **OpenCV**: Image preprocessing and HUD rendering.
- **Mediapipe**: ML-based hand landmark extraction.
- **Pynput**: System-level keyboard simulation for cross-platform control.
- **NumPy & Math**: Geometric calculations for gesture recognition logic.

---

## 💻 Getting Started

### Prerequisites
- Python 3.10+
- A webcam or integrated camera

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/media-control-ml-nano.git
   cd media-control-ml-nano
   ```

2. **Setup Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage Guide

Run the main controller:
```bash
python work2.py
```

### Gesture Commands
| Gesture | Action | Description |
| :--- | :--- | :--- |
| **Open Palm** | `Wake Up` | Activates the system from IDLE state. |
| **Fist** | `Mute (m)` | Toggles mute on currently active media. |
| **Pinch** | `Play/Pause (Space)` | Toggles playback of media. |
| **Rotary (Index)** | `Vol +/-` | Rotate index finger relative to wrist to adjust volume. |

---

## ⚙️ Edge Optimization
*   **Target Device**: Designed with **NVIDIA Jetson Nano** in mind.
*   **Resolution Control**: Fixed at 640x480 to maintain high FPS for fluid interaction.
*   **Confidence Thresholds**: Tuned `min_detection_confidence=0.8` to ensure high precision in varying lighting conditions.

---

## 🗺️ Roadmap
- [ ] Support for multiple hands.
- [ ] Customizable gesture mappings via config file.
- [ ] Integration with specific media players (Spotify, YouTube Music) via API.
- [ ] Deep Learning-based custom gesture trainer.
