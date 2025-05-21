import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import cvzone
import math

# Initialize the YOLO model and video capture
model = YOLO('yolov8n-pose.pt') 
cap = cv2.VideoCapture(0)

# Constants for angle thresholds
BODY_ALIGNMENT_THRESH = 160  # Minimum angle for straight body
ELBOW_DOWN_THRESH = 90       # Max bend angle at bottom position
ELBOW_UP_THRESH = 160        # Min straight angle at top position

# State variables
push_up = False
push_up_counter = 0
last_warning_time = 0  # To prevent spam of form warnings

# Smoothing variables
keypoint_history = []
history_length = 5  # Number of frames to keep for smoothing

# Text-to-speech setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('voice', voices[1].id)
speech_queue = queue.Queue()

def speak(text):
    speech_queue.put(text)
    
def worker_speak():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle



def smooth_keypoints(current_keypoints):
    """smooth_keypoints() function that maintains a history of keypoints over multiple frames"""
    """Averages the position of each keypoint across the last 5 frames to reduce jitter"""
    """This makes the skeletal tracking much more stable"""
    global keypoint_history
    
    # Add current keypoints to history
    keypoint_history.append(current_keypoints)
    
    # Keep only the last N frames
    if len(keypoint_history) > history_length:
        keypoint_history.pop(0)
    
    # If we don't have enough history yet, return current keypoints
    if len(keypoint_history) < 3:
        return current_keypoints
    
    # Smooth keypoints by averaging over history
    smoothed_keypoints = []
    for idx in range(len(current_keypoints)):
        x_coords = []
        y_coords = []
        valid_points = 0
        
        # Collect valid coordinates from history
        for past_keypoints in keypoint_history:
            if idx < len(past_keypoints) and past_keypoints[idx] is not None:
                try:
                    x_coords.append(past_keypoints[idx][0])
                    y_coords.append(past_keypoints[idx][1])
                    valid_points += 1
                except (TypeError, IndexError):
                    # Skip if point is not valid
                    pass
        
        # Calculate average position if we have valid points
        if valid_points > 0:
            avg_x = sum(x_coords) / valid_points
            avg_y = sum(y_coords) / valid_points
            smoothed_keypoints.append((int(avg_x), int(avg_y)))
        else:
            # No valid points in history, keep current or set to None
            if idx < len(current_keypoints) and current_keypoints[idx] is not None:
                smoothed_keypoints.append(current_keypoints[idx])
            else:
                smoothed_keypoints.append(None)
    
    return smoothed_keypoints




def filter_confidence(keypoints, confidences, threshold=0.5):
    """filter_confidence() function that removes keypoints with low confidence scores"""
    """Only keypoints above a 0.5 confidence threshold will be used for calculations"""
    """Each keypoint now displays its confidence score"""
    """Filter out low-confidence keypoints"""
    filtered_keypoints = []
    for i, point in enumerate(keypoints):
        if i < len(confidences) and confidences[i] >= threshold:
            filtered_keypoints.append(point)
        else:
            filtered_keypoints.append(None)  # Mark as invalid
    return filtered_keypoints

def draw_angle(frame, point1, point2, point3, angle, color=(255, 255, 0), radius=20):
    # Check if any point is None (invalid)
    if None in (point1, point2, point3):
        return
    
    # Draw the lines between points
    cv2.line(frame, point1, point2, color, 2)
    cv2.line(frame, point2, point3, color, 2)
    
    # Calculate position for angle text (midpoint between lines)
    angle_text_position = point2
    
    # Draw the angle
    cv2.putText(frame, f"{int(angle)}°", angle_text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Draw angle arc
    try:
        # Calculate angle to decide direction of arc
        angle1 = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
        angle2 = math.atan2(point3[1] - point2[1], point3[0] - point2[0])
        
        start_angle = min(angle1, angle2)
        end_angle = max(angle1, angle2)
        
        # Draw arc
        cv2.ellipse(frame, point2, (radius, radius), 
                    0, start_angle * 180 / math.pi, end_angle * 180 / math.pi, color, 2)
    except:
        pass  # Skip arc drawing if there's an error

# Start speech thread
thread_speak = threading.Thread(target=worker_speak, daemon=True)
thread_speak.start()

# Announce that the system is ready
speak("Push-up counter is ready. Begin your workout!")

# Define the essential connections for push-up form
POSE_CONNECTIONS = [
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 6),           # Shoulders
    (5, 11), (6, 12), # Torso
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16)  # Right leg
]

# Main loop for video capture and processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    
    # Make predictions
    results = model(frame, verbose=False)
    
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            # Extract keypoints and confidences
            try:
                kpts_coords = result.keypoints.xy[0].cpu().numpy()
                kpts_conf = result.keypoints.conf[0].cpu().numpy() if hasattr(result.keypoints, 'conf') else None
                
                # Convert keypoints to the correct format for our functions
                keypoint_list = []
                for i in range(len(kpts_coords)):
                    x, y = int(kpts_coords[i][0]), int(kpts_coords[i][1])
                    keypoint_list.append((x, y))
                
                # Filter low confidence keypoints
                if kpts_conf is not None:
                    keypoint_list = filter_confidence(keypoint_list, kpts_conf, threshold=0.5)
                
                # Smooth keypoints to reduce jitter
                smoothed_keypoints = smooth_keypoints(keypoint_list)
                
                # Draw skeleton connections only for valid keypoints
                for connection in POSE_CONNECTIONS:
                    idx1, idx2 = connection
                    if (idx1 < len(smoothed_keypoints) and idx2 < len(smoothed_keypoints) and 
                        smoothed_keypoints[idx1] is not None and smoothed_keypoints[idx2] is not None):
                        cv2.line(frame, smoothed_keypoints[idx1], smoothed_keypoints[idx2], (0, 255, 0), 2)
                
                # Draw keypoints
                for i, point in enumerate(smoothed_keypoints):
                    if point is not None:
                        conf = kpts_conf[i] if kpts_conf is not None and i < len(kpts_conf) else 0
                        # Use color based on confidence
                        color = (0, int(255 * min(conf, 1.0)), int(255 * (1-min(conf, 1.0))))
                        cv2.circle(frame, point, 5, color, -1)
                        cv2.putText(frame, f'{i}:{conf:.2f}', point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Process push-ups if we have all necessary keypoints
                if len(smoothed_keypoints) >= 16:
                    # Check if all required keypoints are valid
                    required_idxs = [5, 6, 7, 8, 9, 10, 11, 15]
                    all_valid = True
                    for idx in required_idxs:
                        if idx >= len(smoothed_keypoints) or smoothed_keypoints[idx] is None:
                            all_valid = False
                            break
                    
                    if all_valid:
                        # Extract keypoints
                        left_shoulder = smoothed_keypoints[5]
                        left_elbow = smoothed_keypoints[7]
                        left_wrist = smoothed_keypoints[9]
                        
                        right_shoulder = smoothed_keypoints[6]
                        right_elbow = smoothed_keypoints[8]
                        right_wrist = smoothed_keypoints[10]
                        
                        left_hip = smoothed_keypoints[11]
                        left_ankle = smoothed_keypoints[15]
                        
                        # Calculate angles
                        left_hand_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        right_hand_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        body_alignment_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                        avg_arm_angle = (left_hand_angle + right_hand_angle) / 2
                        
                        # Draw angles on the frame
                        draw_angle(frame, left_shoulder, left_elbow, left_wrist, left_hand_angle, color=(0, 0, 255))
                        draw_angle(frame, right_shoulder, right_elbow, right_wrist, right_hand_angle, color=(0, 0, 255))
                        draw_angle(frame, left_shoulder, left_hip, left_ankle, body_alignment_angle, color=(255, 0, 0))

                        # Push-up detection logic with improved form checking
                        form_valid = body_alignment_angle >= BODY_ALIGNMENT_THRESH

                        # Down position
                        if not push_up and avg_arm_angle < ELBOW_DOWN_THRESH and form_valid:
                            push_up = True
                            speak("Down")

                        # Up position - count the push-up
                        elif push_up and avg_arm_angle >= ELBOW_UP_THRESH and form_valid:
                            push_up_counter += 1
                            push_up = False
                            speak(f"Up! {push_up_counter}")

                        # Form warnings
                        current_time = cv2.getTickCount() / cv2.getTickFrequency()
                        if not form_valid and (current_time - last_warning_time) > 2:  # Warning every 2 seconds
                            speak("Keep your body straight")
                            last_warning_time = current_time

                        # Visual feedback
                        cvzone.putTextRect(frame, f'Push-ups: {push_up_counter}', (50, 50), 2, 2, colorR=(0,255,0))
                        cvzone.putTextRect(frame, f'Body Alignment: {int(body_alignment_angle)}°', (50, 100), 1, 1)
                        cvzone.putTextRect(frame, f'Arm Angle: {int(avg_arm_angle)}°', (50, 150), 1, 1)
                        
                        if not form_valid:
                            cvzone.putTextRect(frame, "Fix Form!", (50, 200), 2, 2, colorR=(0,0,255))
                    else:
                        cvzone.putTextRect(frame, "Position your body fully in frame", (50, 50), 1, 1, colorR=(0,0,255))
                else:
                    cvzone.putTextRect(frame, "Position your body fully in frame", (50, 50), 1, 1, colorR=(0,0,255))
            except Exception as e:
                # Handle any errors in detection
                cvzone.putTextRect(frame, f"Detection error: {str(e)}", (50, 50), 1, 1, colorR=(0,0,255))

    # Display the frame
    cv2.imshow("Push-up Counter", frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) == 27:
        break

# Cleanup
speech_queue.put(None)  # Signal the speech thread to end
cap.release()
cv2.destroyAllWindows()
thread_speak.join(timeout=1)