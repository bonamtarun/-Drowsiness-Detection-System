import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import pygame

# Initialize Pygame mixer for looping audio
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)

# Serial setup (match your Arduino COM port)
try:
    arduino = serial.Serial('COM15', 9600, timeout=1)
    print("✅ Connected to Arduino")
except Exception as e:
    print(f"❌ Arduino connection failed: {e}")
    arduino = None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAMES = 20
SUNGLASSES_FRAMES = 15  # Reduced for faster detection
SUNGLASSES_SCORE_THRESHOLD = 4  # Minimum score for sunglasses detection

# State variables
counter = 0
sunglasses_counter = 0
alert_active = False
current_alert = None
prev_time = time.time()

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    try:
        p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in eye_indices]
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        ear = vertical / (2.0 * horizontal) if horizontal > 0 else 0
        return ear
    except:
        return 0

def get_eye_brightness_advanced(frame, landmarks, eye_indices):
    """Advanced brightness calculation"""
    h, w = frame.shape[:2]
    
    try:
        # Get eye region coordinates
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points, dtype=np.int32)
        
        # Create expanded region around eye
        x_coords = eye_points[:, 0]
        y_coords = eye_points[:, 1]
        
        x_min, x_max = max(0, np.min(x_coords) - 8), min(w, np.max(x_coords) + 8)
        y_min, y_max = max(0, np.min(y_coords) - 8), min(h, np.max(y_coords) + 8)
        
        # Extract eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size > 0:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            avg_brightness = np.mean(gray_eye)
            min_brightness = np.min(gray_eye)
            std_brightness = np.std(gray_eye)
            
            # Calculate dark pixel ratio
            dark_pixels = np.sum(gray_eye < 50)
            total_pixels = gray_eye.shape[0] * gray_eye.shape[1]
            dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
            
            return {
                'avg_brightness': avg_brightness,
                'min_brightness': min_brightness,
                'std_brightness': std_brightness,
                'dark_ratio': dark_ratio
            }
    except Exception as e:
        pass
    
    return {
        'avg_brightness': 255,
        'min_brightness': 255,
        'std_brightness': 0,
        'dark_ratio': 0
    }

def detect_sunglasses_robust(frame, landmarks):
    """Robust sunglasses detection with scoring system"""
    
    # Get brightness data for both eyes
    left_data = get_eye_brightness_advanced(frame, landmarks, LEFT_EYE)
    right_data = get_eye_brightness_advanced(frame, landmarks, RIGHT_EYE)
    
    # Calculate EAR
    left_ear = calculate_ear(landmarks, LEFT_EYE)
    right_ear = calculate_ear(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2
    
    # Average brightness metrics
    avg_brightness = (left_data['avg_brightness'] + right_data['avg_brightness']) / 2
    min_brightness = min(left_data['min_brightness'], right_data['min_brightness'])
    avg_dark_ratio = (left_data['dark_ratio'] + right_data['dark_ratio']) / 2
    avg_std = (left_data['std_brightness'] + right_data['std_brightness']) / 2
    
    # Scoring system for sunglasses detection
    score = 0
    detection_reasons = []
    
    # Primary indicators (higher weight)
    if avg_brightness < 45:
        score += 3
        detection_reasons.append("very_dark")
    elif avg_brightness < 60:
        score += 1
        detection_reasons.append("dark")
    
    if min_brightness < 20:
        score += 2
        detection_reasons.append("min_very_dark")
    
    if avg_dark_ratio > 0.6:
        score += 2
        detection_reasons.append("high_dark_ratio")
    
    # Secondary indicators
    if avg_std < 18:  # Low variation indicates uniform darkness
        score += 1
        detection_reasons.append("uniform_darkness")
    
    if avg_ear < 0.12:  # Very low EAR might indicate blocked eyes
        score += 1
        detection_reasons.append("very_low_ear")
    elif avg_ear > 0.45:  # Very high EAR might indicate detection issues
        score += 1
        detection_reasons.append("very_high_ear")
    
    # Additional check: Compare with face brightness
    try:
        face_region = frame[int(landmarks[10].y * frame.shape[0] - 20):
                          int(landmarks[152].y * frame.shape[0] + 20),
                          int(landmarks[234].x * frame.shape[1]):
                          int(landmarks[454].x * frame.shape[1])]
        if face_region.size > 0:
            face_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
            brightness_diff = face_brightness - avg_brightness
            if brightness_diff > 30:  # Eyes much darker than face
                score += 2
                detection_reasons.append("eyes_darker_than_face")
    except:
        pass
    
    sunglasses_detected = score >= SUNGLASSES_SCORE_THRESHOLD
    
    return {
        'detected': sunglasses_detected,
        'score': score,
        'reasons': detection_reasons,
        'avg_brightness': avg_brightness,
        'min_brightness': min_brightness,
        'dark_ratio': avg_dark_ratio,
        'ear': avg_ear
    }

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🚗 Enhanced Driver Monitor Started")
print(f"📊 Sunglasses detection threshold: {SUNGLASSES_SCORE_THRESHOLD}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = "Normal"
    status_color = (0, 255, 0)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Run sunglasses detection
        sunglasses_result = detect_sunglasses_robust(frame, landmarks)
        
        # Check for sunglasses first
        if sunglasses_result['detected']:
            sunglasses_counter += 1
            counter = 0  # Reset drowsy counter
            
            if sunglasses_counter > SUNGLASSES_FRAMES:
                if not alert_active or current_alert != 'sunglasses':
                    reasons_str = ", ".join(sunglasses_result['reasons'][:3])  # Show first 3 reasons
                    print(f"🕶️ Sunglasses detected! Score: {sunglasses_result['score']}, Reasons: {reasons_str}")
                    pygame.mixer.music.load("eyesnot.mp3")
                    pygame.mixer.music.play(-1)
                    if arduino:
                        arduino.write(b'1')
                    alert_active = True
                    current_alert = 'sunglasses'
                
                status_text = f"SUNGLASSES! (Score: {sunglasses_result['score']})"
                status_color = (0, 165, 255)
        
        # Regular drowsiness detection (only if no sunglasses)
        else:
            sunglasses_counter = 0
            avg_ear = sunglasses_result['ear']
            
            if avg_ear < EAR_THRESHOLD:
                counter += 1
                
                if counter > CLOSED_EYE_FRAMES:
                    if not alert_active or current_alert != 'drowsy':
                        print("😴 Drowsiness detected")
                        pygame.mixer.music.load("wakeup.mp3")
                        pygame.mixer.music.play(-1)
                        if arduino:
                            arduino.write(b'1')
                        alert_active = True
                        current_alert = 'drowsy'
                    
                    status_text = "DROWSY!"
                    status_color = (0, 0, 255)
            else:
                counter = 0
                # Clear any active alert when returning to normal
                if alert_active:
                    print("✅ Back to normal")
                    pygame.mixer.music.stop()
                    if arduino:
                        arduino.write(b'0')
                    alert_active = False
                    current_alert = None

        # Display debug information
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detection metrics
        cv2.putText(frame, f"Brightness: {sunglasses_result['avg_brightness']:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"EAR: {sunglasses_result['ear']:.3f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"SG Score: {sunglasses_result['score']}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        # No face detected
        status_text = "NO FACE"
        status_color = (0, 255, 255)
        
        if not alert_active:
            sunglasses_counter += 1
            if sunglasses_counter > SUNGLASSES_FRAMES:
                print("⚠️ No face detected")
                pygame.mixer.music.load("eyesnot.mp3")
                pygame.mixer.music.play(-1)
                if arduino:
                    arduino.write(b'1')
                alert_active = True
                current_alert = 'no_face'

    # Display main status
    cv2.putText(frame, status_text, (10, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
    
    cv2.imshow("Enhanced Driver Monitor", frame)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
print("🛑 Shutting down...")
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
print("✅ Cleanup complete")