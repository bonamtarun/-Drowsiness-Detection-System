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
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices for MAR calculation
MOUTH = [61, 291, 0, 17, 267, 84] # Example: corners and top/bottom lip points

NOSE_IDX = 1
MOUTH_IDX = 13

# Thresholds
EAR_THRESHOLD = 0.25  # Adjust based on sensitivity
MAR_THRESHOLD = 0.7   # Adjust based on sensitivity for yawning
PITCH_THRESHOLD = 15  # Degrees for head nod

CLOSED_EYE_FRAMES = 20  # Frames before drowsiness alert (~0.67 sec at 30 FPS)
NO_EYE_FRAMES = 150  # 5 sec at 30 FPS (adjust if your camera FPS differs)
HEAD_NOD_FRAMES = 15 # Frames for head nod detection
YAWN_FRAMES = 15 # Frames for yawn detection

# State variables
counter = 0
no_eye_counter = 0
head_nod_counter = 0
yawn_counter = 0
alert_active = False
current_alert = None  # 'drowsy', 'glasses', 'head_nod', 'yawn'

# FPS estimation (optional)
prev_time = time.time()

def calculate_ear(landmarks, eye_indices):
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in eye_indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    ear = vertical / (2.0 * horizontal)
    return ear

def calculate_mar(landmarks, mouth_indices):
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in mouth_indices]
    # Vertical distance between inner top and bottom lip landmarks
    vertical = np.linalg.norm(p[1] - p[5])
    # Horizontal distance between mouth corners
    horizontal = np.linalg.norm(p[0] - p[3])
    mar = vertical / horizontal
    return mar

# For head pose estimation
# 3D model points (from MediaPipe documentation or common practice)
# These are approximate and might need fine-tuning based on your specific MediaPipe version/model
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def get_head_pose(landmarks, img_w, img_h):
    # 2D image points from MediaPipe landmarks
    # Using specific landmarks for nose, chin, eye corners, mouth corners
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),       # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h),   # Chin
        (landmarks[33].x * img_w, landmarks[33].y * img_h),     # Left eye corner
        (landmarks[263].x * img_w, landmarks[263].y * img_h),   # Right eye corner
        (landmarks[61].x * img_w, landmarks[61].y * img_h),     # Left mouth corner
        (landmarks[291].x * img_w, landmarks[291].y * img_h)    # Right mouth corner
    ], dtype="double")

    # Camera internals (approximate, usually need calibration)
    # Assuming a standard webcam, these values are often good starting points
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Get rotation matrix
    rmat, jac = cv2.Rodrigues(rotation_vector)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Return pitch, yaw, roll in degrees
    x = angles[0] * 360 # Pitch
    y = angles[1] * 360 # Yaw
    z = angles[2] * 360 # Roll

    return x, y, z

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

        # MAR Calculation
        mar = calculate_mar(landmarks, MOUTH)

        # Head Pose Estimation
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        # Check if eyes are visible
        left_eye_visible = landmarks[33].visibility > 0.5
        right_eye_visible = landmarks[362].visibility > 0.5
        eyes_visible = left_eye_visible or right_eye_visible

        # Check nose/mouth for face presence
        nose_visible = landmarks[NOSE_IDX].visibility > 0.5
        mouth_visible = landmarks[MOUTH_IDX].visibility > 0.5
        face_visible = nose_visible or mouth_visible

        # Reset all counters if not in an alert state and conditions are normal
        if not alert_active and avg_ear >= EAR_THRESHOLD and mar < MAR_THRESHOLD and abs(pitch) < PITCH_THRESHOLD:
            counter = 0
            no_eye_counter = 0
            head_nod_counter = 0
            yawn_counter = 0

        # Case 1: Eyes not visible but face detected (glasses/sunglasses) - use head pose and yawning
        if not eyes_visible and face_visible:
            no_eye_counter += 1
            counter = 0  # Reset drowsy counter

            if no_eye_counter > NO_EYE_FRAMES:
                if not alert_active or current_alert != 'glasses':
                    print(f"😎 Glasses detected (eyes not visible for 5 sec)")
                    # pygame.mixer.music.load("eyesnot.mp3")  # Replace with your file
# pygame.mixer.music.play(-1)  # -1 = loop forever
# if arduino:
#     arduino.write(b'1')  # LED blink + motor OFF
                    alert_active = True
                    current_alert = 'glasses'

            # Within glasses detected state, check for head nod or yawn
            if abs(pitch) > PITCH_THRESHOLD: # Head nod detected
                head_nod_counter += 1
                if head_nod_counter > HEAD_NOD_FRAMES:
                    if not alert_active or current_alert != 'head_nod':
                        print("😴 Head Nod Drowsiness detected")
                        # pygame.mixer.music.load("wakeup.mp3") # Use wakeup for head nod
# pygame.mixer.music.play(-1)
# if arduino:
#     arduino.write(b'1')
                        alert_active = True
                        current_alert = 'head_nod'
            else:
                head_nod_counter = 0

            if mar > MAR_THRESHOLD: # Yawn detected
                yawn_counter += 1
                if yawn_counter > YAWN_FRAMES:
                    if not alert_active or current_alert != 'yawn':
                        print("😮 Yawn Drowsiness detected")
                        # pygame.mixer.music.load("wakeup.mp3") # Use wakeup for yawn
# pygame.mixer.music.play(-1)
# if arduino:
#     arduino.write(b'1')
                        alert_active = True
                        current_alert = 'yawn'
            else:
                yawn_counter = 0

        # Case 2: Eyes closed (drowsiness) - primary EAR detection
        elif avg_ear < EAR_THRESHOLD:
            counter += 1
            no_eye_counter = 0  # Reset glasses counter
            head_nod_counter = 0 # Reset head nod counter
            yawn_counter = 0 # Reset yawn counter

            if counter > CLOSED_EYE_FRAMES:
                if not alert_active or current_alert != 'drowsy':
                    print("😴 Drowsiness detected")
                    pygame.mixer.music.load("wakeup.mp3")  # Replace with your file
                    pygame.mixer.music.play(-1)  # -1 = loop forever
                    if arduino:
                        arduino.write(b'1')  # LED blink + motor OFF
                    alert_active = True
                    current_alert = 'drowsy'

        # Case 3: Normal state (eyes open, no head nod, no yawn)
        else:
            if alert_active:
                print("✅ Back to normal")
                # pygame.mixer.music.stop()  # Stop looping audio
                if arduino:
                    arduino.write(b'0')  # LED off + motor ON
                alert_active = False
                current_alert = None
            counter = 0
            no_eye_counter = 0
            head_nod_counter = 0
            yawn_counter = 0

    # No face detected (optional: add a timeout for full system stop)
    else:
        if not alert_active:
            no_eye_counter += 1 # Re-using no_eye_counter for general no face detection
            if no_eye_counter > NO_EYE_FRAMES:
                print("⚠️ No face detected for 5 sec")
                pygame.mixer.music.load("eyesnot.mp3")  # Same as glasses alert
                pygame.mixer.music.play(-1)
                if arduino:
                    arduino.write(b'1')  # LED blink + motor OFF
                alert_active = True
                current_alert = 'glasses' # Can be a generic 'no_face' alert type

    # Display the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) if 'avg_ear' in locals() else None
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) if 'mar' in locals() else None
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) if 'pitch' in locals() else None
    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
# pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()