import cv2
import numpy as np
import mediapipe as mp
import math



# Helper functions
def draw_text(image, text, x, y, font_scale, font_color, thickness, bg_color):
    """
    Utility function to draw text with a background on the image.
    Args:
    - image: The video frame.
    - text: Text to display.
    - x, y: Coordinates for the text's bottom-left corner.
    - font_scale: Font scale.
    - font_color: Color of the text (B, G, R).
    - thickness: Thickness of the text.
    - bg_color: Background color of the rectangle (B, G, R).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    baseline += 2  # Ensure the background rectangle includes the baseline

    # Draw the background rectangle
    cv2.rectangle(
        image,
        (x, y - text_h - baseline),  # Top-left corner of the rectangle
        (x + text_w, y + baseline),  # Bottom-right corner
        bg_color,
        -1
    )

    # Draw the text on top of the rectangle
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B (vertex)
    c = np.array(c)  # Point C
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle_center(center, point):
    """
    Calculate the angle of a point relative to a center.
    
    Args:
    - center: Tuple of (x, y) coordinates of the center.
    - point: Tuple of (x, y) coordinates of the point.

    Returns:
    - Angle in radians.
    """
    return math.atan2(point[1] - center[1], point[0] - center[0])


# Bicep curl tracking
def track_bicep_curls(image, landmarks, count, curl_state, elbow_index, shoulder_index, wrist_index):
    """Track bicep curls based on arm angle."""
    shoulder = [landmarks[shoulder_index].x, landmarks[shoulder_index].y]
    elbow = [landmarks[elbow_index].x, landmarks[elbow_index].y]
    wrist = [landmarks[wrist_index].x, landmarks[wrist_index].y]
    
    # Calculate the angle
    angle = calculate_angle(shoulder, elbow, wrist)
    
    # Determine curl state and count
    if angle > 160:
        curl_state = "down"
    if angle < 40 and curl_state == "down":
        curl_state = "up"
        count += 1
    
    # Display curl count and angle
    draw_text(image, 'CURLS', 15, 60, 0.5, (255, 255, 255), 1, (0, 0, 0))
    draw_text(image, str(count), 10, 100, 2, (255, 255, 255), 2, (0, 0, 0))
    draw_text(image, f'ANGLE: {int(angle)}', 10, 140, 0.5, (255, 255, 255), 1, (0, 0, 0))
    
    return count, curl_state



def initialize_checkpoints(shoulder_x, shoulder_y, radius, checkpoint_labels):
    """
    Initialize circular checkpoints around a given shoulder point.
    
    Args:
    - shoulder_x: X-coordinate of the shoulder.
    - shoulder_y: Y-coordinate of the shoulder.
    - radius: Radius of the circular path.
    - checkpoint_labels: List of checkpoint labels.
    
    Returns:
    - A dictionary with checkpoint labels as keys and (x, y) positions as values.
    """
    checkpoints = {}
    for i, label in enumerate(checkpoint_labels):
        angle = i * (2 * np.pi / len(checkpoint_labels))  # Divide circle into equal angles
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        checkpoints[label] = (shoulder_x + x_offset, shoulder_y + y_offset)
    return checkpoints

def track_hand_rotation(image, landmarks, rotation_counters, checkpoints):
    """
    Track hand rotations using wrist checkpoints along a circular path.
    Args:
    - image: The video frame.
    - landmarks: Pose landmarks.
    - rotation_counters: Dict to store counters for both hands.
    - checkpoints: Dict to store checkpoint states for both hands.
    """
    mp_pose = mp.solutions.pose
    for side, wrist_idx, shoulder_idx in [
        ("LEFT", mp_pose.PoseLandmark.LEFT_WRIST.value,
         mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        ("RIGHT", mp_pose.PoseLandmark.RIGHT_WRIST.value,
         mp_pose.PoseLandmark.RIGHT_SHOULDER.value)]:
        
        wrist = landmarks[wrist_idx]
        shoulder = landmarks[shoulder_idx]

        # Calculate wrist position and shoulder reference
        wrist_pos = np.array([wrist.x, wrist.y])
        shoulder_x, shoulder_y = shoulder.x, shoulder.y

        # Ensure checkpoints are initialized
        if side not in checkpoints:
            checkpoints[side] = {"current": "bottom"}

        # Checkpoints on the circular path
        checkpoint_labels = [
            "bottom", "bottom-right", "right", "top-right",
            "top", "top-left", "left", "bottom-left"
        ]
        radius = 0.2  # Radius of the circular path
        checkpoints[side].update({
            label: (
                shoulder_x + radius * np.cos(angle),
                shoulder_y + radius * np.sin(angle)
            )
            for label, angle in zip(checkpoint_labels, np.linspace(0, 2 * np.pi, len(checkpoint_labels), endpoint=False))
        })

        # Match wrist position with checkpoints
        current_checkpoint = checkpoints[side]["current"]
        current_idx = checkpoint_labels.index(current_checkpoint)
        next_idx = (current_idx + 1) % len(checkpoint_labels)
        next_checkpoint = checkpoint_labels[next_idx]

        # Check wrist proximity to the next checkpoint
        checkpoint_x, checkpoint_y = checkpoints[side][next_checkpoint]
        distance = np.linalg.norm(wrist_pos - np.array([checkpoint_x, checkpoint_y]))

        if distance < 0.05:  # Threshold for proximity to the checkpoint
            checkpoints[side]["current"] = next_checkpoint

            # Increment count if the wrist completes a full circle
            if next_checkpoint == "bottom":
                rotation_counters[side]["count"] += 1

        # Display rotation count
        y_offset = 60 if side == "LEFT" else 160
        draw_text(image, f'{side} ROTATIONS', 15, y_offset, 0.5, (255, 255, 255), 1, (0, 0, 0))
        draw_text(image, str(rotation_counters[side]["count"]), 15, y_offset + 30, 1, (255, 255, 255), 2, (0, 0, 0))

        # Visualize checkpoints
        for label, point in checkpoints[side].items():
            if label != "current":
                x, y = int(point[0] * image.shape[1]), int(point[1] * image.shape[0])
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)




def track_jumps(image, landmarks, jump_data, jump_threshold=0.5):
    """
    Track jumping exercise by monitoring the Y-coordinate of the left hip.
    
    Args:
    - image: The video frame.
    - landmarks: Pose landmarks.
    - jump_data: Dictionary containing the jump count and stage.
    - jump_threshold: Threshold for Y-coordinate change to detect a jump (default: 0.1).
    
    Returns:
    - Updated jump_data dictionary.
    """
    # Get the Y-coordinate of the left hip
    mp_pose = mp.solutions.pose
    hip_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
    
    # Check for jump start (upward movement above threshold)
    if jump_data["stage"] == "grounded" and hip_y < (1 - jump_threshold):
        jump_data["stage"] = "in_air"
    
    # Check for landing (when Y-coordinate returns below threshold)
    elif jump_data["stage"] == "in_air" and hip_y >= (1 - jump_threshold):
        jump_data["stage"] = "grounded"
        jump_data["count"] += 1  # Increment the jump count

    # Display jump count on the video feed
    draw_text(image, 'JUMPS', 15, 60, 0.5, (0, 0, 0), 1, (255, 255, 255))
    draw_text(image, str(jump_data["count"]), 10, 100, 2, (0, 0, 0), 2, (255, 255, 255))
    
    return jump_data


def track_jumping_jacks(frame,landmarks, rep_data, cooldown_threshold=3, leg_distance_threshold=0.25):
    """
    Tracks the jumping jacks exercise based on arm and leg positions.

    Args:
    - landmarks: Pose landmarks from Mediapipe.
    - rep_data: Dictionary containing rep count and cooldown state.
    - cooldown_threshold: Number of frames for cooldown after a rep (default: 3).
    - leg_distance_threshold: Minimum X-axis distance between ankles to consider legs apart (default: 0.25).

    Returns:
    - Updated rep_data dictionary.
    """
    mp_pose = mp.solutions.pose

    # Extract arm positions (wrist and shoulder Y-coordinates)
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    arms_up = (left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y)

    # Extract leg positions (ankle X-coordinates)
    left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    leg_distance = abs(left_ankle_x - right_ankle_x)
    legs_apart = (leg_distance > leg_distance_threshold)

    # Count rep if both arms are up and legs are apart, with cooldown
    if arms_up and legs_apart and rep_data["cooldown"] == 0:
        rep_data["count"] += 1
        rep_data["cooldown"] = cooldown_threshold  # Set cooldown

    # Decrease cooldown
    rep_data["cooldown"] = max(0, rep_data["cooldown"] - 1)
    # Display the rep count
    cv2.putText(frame, f"Reps: {rep_data['count']}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return rep_data


def track_sit_ups(image, landmarks, sit_up_count, sit_up_state):
    """
    Tracks sit-ups by analyzing the angle between shoulder, hip, and knee.
    Args:
    - image: The video frame.
    - landmarks: Pose landmarks.
    - sit_up_count: The current count of sit-ups.
    - sit_up_state: The current state of sit-up motion ('down' or 'up').
    Returns:
    - sit_up_count: Updated sit-up count.
    - sit_up_state: Updated sit-up state.
    """
    mp_pose = mp.solutions.pose
    # Get landmarks for shoulder, hip, and knee
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

    # Calculate angle at the hip
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])  # First point
        b = np.array([b.x, b.y])  # Middle point (hip)
        c = np.array([c.x, c.y])  # Last point
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    # Sit-up logic
    if hip_angle > 160:  # Fully extended position
        sit_up_state = "down"
    elif hip_angle < 70 and sit_up_state == "down":  # Fully contracted position
        sit_up_state = "up"
        sit_up_count += 1

    # Display sit-up count
    draw_text(image, "SIT-UPS", 15, 260, 0.5, (255, 255, 255), 1, (0, 0, 0))
    draw_text(image, str(sit_up_count), 15, 290, 1, (255, 255, 255), 2, (0, 0, 0))

    # Visualize angle for debugging
    hip_x, hip_y = int(left_hip.x * image.shape[1]), int(left_hip.y * image.shape[0])
    cv2.putText(image, f"{int(hip_angle)}°", (hip_x, hip_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return sit_up_count, sit_up_state


def check_balance(landmarks, frame_height):

    mp_pose = mp.solutions.pose
    # Get keypoints for standing balance (e.g., ankles)
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    # Define a ground threshold (e.g., the ankle position should be close to the bottom of the frame when standing)
    ground_threshold = 0.05  # A small threshold to determine if ankle is near the ground
    ankle_y_threshold = frame_height - 50  # Consider the bottom part of the frame as the ground

    # If the Y position of both ankles is greater than the threshold, the person is standing (feet on the ground)
    if left_ankle.y * frame_height > ankle_y_threshold and right_ankle.y * frame_height > ankle_y_threshold:
        return False  # Both feet on the ground, no need to check balance yet
    
    # If either ankle is above the threshold, we can start checking balance
    balance_threshold = 0.1  # Tolerance level for body balance

    # Get key points for balance check (shoulders, hips, and ankles)
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the distances between body parts
    shoulder_distance = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - np.array([right_shoulder.x, right_shoulder.y]))
    hip_distance = np.linalg.norm(np.array([left_hip.x, left_hip.y]) - np.array([right_hip.x, right_hip.y]))
    ankle_distance = np.linalg.norm(np.array([left_ankle.x, left_ankle.y]) - np.array([right_ankle.x, right_ankle.y]))

    # If the distances between shoulder, hip, and ankle are consistent, the person is balanced
    if abs(shoulder_distance - hip_distance) < balance_threshold and abs(hip_distance - ankle_distance) < balance_threshold:
        return True  # Balanced
    return False  # Not balanced

def calculate_angle_3p(point1, point2, point3):
    """
    Calculate the angle between three points.
    """
    # Convert points to numpy arrays
    p1 = np.array([point1.x, point1.y])
    p2 = np.array([point2.x, point2.y])
    p3 = np.array([point3.x, point3.y])

    # Calculate vectors
    vector1 = p1 - p2
    vector2 = p3 - p2

    # Calculate angle
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def check_plank(landmarks):


    mp_pose = mp.solutions.pose

    # Get keypoints for plank (shoulders, hips, ankles)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Calculate angles
    left_body_angle = calculate_angle_3p(left_shoulder, left_hip, left_ankle)  # Left side of the body
    right_body_angle = calculate_angle_3p(right_shoulder, right_hip, right_ankle)  # Right side of the body

    # Tolerance thresholds for angles
    angle_tolerance = (170, 180)  # Ideal plank: angle between 170 and 180 degrees

    # Check if both sides are within the angle tolerance
    if (angle_tolerance[0] <= left_body_angle <= angle_tolerance[1] and
            angle_tolerance[0] <= right_body_angle <= angle_tolerance[1]):
        return True
    return False



def run_exercise_tracker(exercise_mode="hand_rotation"):
    """Run the exercise tracker for a specified mode."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    # Variables for jump tracking
    jump_data = {"count": 0, "stage": "grounded"}
    # Variables for jumping jacks tracking
    rep_data = {"count": 0, "cooldown": 0}
    # Variables for bicep curl tracking
    bicep_curl_count = 0
    curl_state = "down"

    sit_up_count = 0
    sit_up_state = "down"
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.resize(frame, (640, 480))
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                frame_height, frame_width, _ = image.shape
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    if exercise_mode == "hand_rotation":
                        rotation_counters = {
                            "LEFT": {"count": 0},
                            "RIGHT": {"count": 0}
                        }
                        # Properly initialize the checkpoints dictionary
                        checkpoints = {
                            "LEFT": {"current": "bottom"},
                            "RIGHT": {"current": "bottom"}
                        }
                        track_hand_rotation(image, landmarks, rotation_counters, checkpoints)
                    
                    elif exercise_mode == "bicep_curl":
                        bicep_curl_count, curl_state = track_bicep_curls(
                            image, landmarks, bicep_curl_count, curl_state,
                            mp_pose.PoseLandmark.LEFT_ELBOW.value,
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            mp_pose.PoseLandmark.LEFT_WRIST.value)
                    
                    elif exercise_mode == "jumping":
                        jump_data = track_jumps(image, landmarks, jump_data)
                    
                    elif exercise_mode == "jumping_jacks":
                        rep_data = track_jumping_jacks(image,landmarks, rep_data)

                    elif exercise_mode == "sit_ups":
                        sit_up_count, sit_up_state = track_sit_ups(image, landmarks, sit_up_count, sit_up_state)

                    elif exercise_mode == "standing_balance":
                        if check_balance(landmarks,frame_height ):
                            
                            cv2.putText(image, "Balance: Good", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,cv2.LINE_AA)
                        else:
                            cv2.putText(image, "Balance: Bad", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,cv2.LINE_AA)

                    elif exercise_mode == "plank":
                        if check_plank(landmarks):
                            cv2.putText(image, "Plank: Good", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
                        else:
                            cv2.putText(image, "Plank: Bad", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                cv2.imshow('Mediapipe Feed', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
