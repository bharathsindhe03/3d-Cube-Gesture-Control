# 3d-Cube-Gesture-Control

This project showcases a 3D cube that can be rotated and scaled using hand gestures detected via a webcam. The application uses `pygame` for rendering the 3D cube and `MediaPipe` for detecting hand landmarks.

## Features

- Rotate the cube around the X, Y, and Z axes using hand gestures.
- Increase and decrease the scale of the cube using hand gestures.
- Display the webcam feed with overlaid hand landmarks.

## Requirements

- Python 3.12
- `pygame`
- `numpy`
- `opencv-python`
- `mediapipe`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bharathsindhe03/3d-Cube-Gesture-Control.git
    cd 3d-cube-rotation
    ```

2. Install the required packages:

    ```bash
    pip install pygame numpy opencv-python mediapipe
    ```

## Usage

Run the main script to start the application:

```bash
python main.py
