import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import pygame

# Initialize Pygame mixer for looping audio
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)  # Max volume

# Serial setup (match your Arduino COM port)
try:
    arduino = serial.Serial('COM15', 9600, timeout=1)
    print("✅ Connected to Arduino")
except Exception as e:
    print(f"❌ Arduino connection failed: {e}")
    arduino = None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_IDX = 1
MOUTH_IDX = 13

# Thresholds
EAR_THRESHOLD = 0.25  # Adjust based on sensitivity
CLOSED_EYE_FRAMES = 20  # Frames before drowsiness alert (~0.67 sec at 30 FPS)
NO_EYE_FRAMES = 150  # 5 sec at 30 FPS (adjust if your camera FPS differs)

# State variables
counter = 0
no_eye_counter = 0
alert_active = False
current_alert = None  # 'drowsy' or 'glasses'

# FPS estimation (optional)
prev_time = time.time()

def calculate_ear(landmarks, eye_indices):
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in eye_indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    ear = vertical / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)  # Laptop camera

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate FPS (optional, for debugging)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # EAR Calculation
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        # Check if eyes are visible
        left_eye_visible = landmarks[33].visibility > 0.5
        right_eye_visible = landmarks[362].visibility > 0.5
        eyes_visible = left_eye_visible or right_eye_visible

        # Check nose/mouth for face presence
        nose_visible = landmarks[NOSE_IDX].visibility > 0.5
        mouth_visible = landmarks[MOUTH_IDX].visibility > 0.5
        face_visible = nose_visible or mouth_visible

        # Case 1: Eyes not visible but face detected (glasses/sunglasses)
        if not eyes_visible and face_visible:
            no_eye_counter += 1
            counter = 0  # Reset drowsy counter
            if no_eye_counter > NO_EYE_FRAMES:
                if not alert_active or current_alert != 'glasses':
                    print(f"😎 Glasses detected (eyes not visible for 5 sec)")
                    pygame.mixer.music.load("eyesnot.mp3")  # Replace with your file
                    pygame.mixer.music.play(-1)  # -1 = loop forever
                    if arduino:
                        arduino.write(b'1')  # LED blink + motor OFF
                    alert_active = True
                    current_alert = 'glasses'

        # Case 2: Eyes closed (drowsiness)
        elif avg_ear < EAR_THRESHOLD:
            counter += 1
            no_eye_counter = 0  # Reset glasses counter
            if counter > CLOSED_EYE_FRAMES:
                if not alert_active or current_alert != 'drowsy':
                    print("😴 Drowsiness detected")
                    pygame.mixer.music.load("wakeup.mp3")  # Replace with your file
                    pygame.mixer.music.play(-1)  # -1 = loop forever
                    if arduino:
                        arduino.write(b'1')  # LED blink + motor OFF
                    alert_active = True
                    current_alert = 'drowsy'

        # Case 3: Normal state (eyes open)
        else:
            if alert_active:
                print("✅ Back to normal")
                pygame.mixer.music.stop()  # Stop looping audio
                if arduino:
                    arduino.write(b'0')  # LED off + motor ON
                alert_active = False
                current_alert = None
            counter = 0
            no_eye_counter = 0

    # No face detected (optional: add a timeout for full system stop)
    else:
        if not alert_active:
            no_eye_counter += 1
            if no_eye_counter > NO_EYE_FRAMES:
                print("⚠ No face detected for 5 sec")
                pygame.mixer.music.load("eyesnot.mp3")  # Same as glasses alert
                pygame.mixer.music.play(-1)
                if arduino:
                    arduino.write(b'1')  # LED blink + motor OFF
                alert_active = True
                current_alert = 'glasses'

    # Display the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()