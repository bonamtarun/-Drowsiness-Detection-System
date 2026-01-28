# Driver Drowsiness Detection System with Hardware Alert

This project sends alerts to a driver if drowsiness or distraction is detected. It uses Computer Vision (MediaPipe & OpenCV) to monitor eye state and head pose, and communicates with an Arduino to trigger physical alerts (LEDs and Motor/Vibration) alongside audio warnings.

## 🚀 Features

*   **Real-time Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect closed eyes (microsleeps).
*   **Distraction/Occlusion Detection**: Detects if eyes are not visible (e.g., looking away or wearing opaque sunglasses) while the face is still present.
*   **Audio Alerts**: Plays custom warning sounds (`wakeup.mp3`, `eyesnot.mp3`) when danger is detected.
*   **Hardware Integration**:
    *   **Visual Alert**: Blinking LED (Pin 13).
    *   **Physical Alert**: Motor/Vibrator activation (Pin 6).
    *   **System Status**: Indicator LED (Pin 11) showing the system is armed via a physical switch.

## 🛠️ Components Required

### Hardware
*   **Webcam** (Laptop or External USB)
*   **Arduino Board** (Uno, Nano, etc.)
*   **Vibration Motor** (or DC Motor logic)
*   **LEDs** (x2)
*   **Push Switch** / Toggle Switch
*   **Jumper Wires & Breadboard**

### Pin Connections (Arduino)

| Component | Arduino Pin | Description |
| :--- | :--- | :--- |
| **Switch** | Pin 9 | System Enable/Disable (Input with Pull-up) |
| **LED 1 (Status)** | Pin 11 | Turns ON when the system is enabled via switch |
| **LED 2 (Alert)** | Pin 13 | Blinks when drowsiness is detected |
| **Motor** | Pin 6 | Activates (LOW signal) for physical alert |

*Note: The motor logic in the provided code assumes Active LOW (ON = LOW, OFF = HIGH).*

## 💻 Software Requirements

*   Python 3.x
*   Arduino IDE

### Python Libraries
Install the necessary libraries using pip:

```bash
pip install opencv-python mediapipe numpy pyserial pygame
```

## ⚙️ Setup & Installation

1.  **Arduino Setup**:
    *   Connect your components according to the pinout above.
    *   Open `alert_system.ino` in the Arduino IDE.
    *   Select your board and COM port.
    *   Upload the code to the Arduino.
    *   **Note the COM Port** (e.g., `COM3`, `COM15`) used by the Arduino.

2.  **Python Setup**:
    *   Open `main (1).py` (or rename it to `main.py`).
    *   **Update the COM Port**:
        In the code, find the line:
        ```python
        arduino = serial.Serial('COM15', 9600, timeout=1)
        ```
        Change `'COM15'` to the port your Arduino is connected to.
    *   **Audio Files**: Ensure `wakeup.mp3` and `eyesnot.mp3` are in the same directory as the script.

## ▶️ Usage

1.  Connect the Arduino to your PC.
2.  Run the Python script:
    ```bash
    python main.py
    ```
3.  **Arm the System**: Toggle the physical switch connected to Pin 9 on the Arduino. The Status LED (Pin 11) should turn ON.
4.  **Testing**:
    *   **Drowsiness**: Close your eyes for ~2 seconds. The system should play an audio alert, blink the LED, and activate the motor.
    *   **Distraction/Glasses**: Cover your eyes (but keep your face visible) or wear sunglasses. After ~5 seconds, the "Glasses detected" alert will trigger.

## 📊 Configuration

You can adjust the sensitivity in `main.py`:

*   `EAR_THRESHOLD = 0.25`: Increase if detection is too difficult, decrease if too sensitive.
*   `CLOSED_EYE_FRAMES = 20`: Number of consecutive frames eyes must be closed to trigger alert.
*   `NO_EYE_FRAMES = 150`: Frames to wait before triggering "Glasses/No Eyes" alert.

## 🤝 Contributing
Feel free to fork this project and submit pull requests for improvements!
