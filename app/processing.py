import cv2
import mediapipe as mp
from .exercise_logic import (
    track_bicep_curls,
    track_jumps,
    track_jumping_jacks,
    track_hand_rotation,
    track_sit_ups
)

# Initialize Mediapipe
mp_pose = mp.solutions.pose

def process_workout_video(video_path, exercise_mode):
    """
    Process a workout video to track exercise-specific movements.

    Args:
        video_path (str): Path to the video file.
        exercise_mode (str): The type of exercise to process.

    Returns:
        dict: Results containing rep counts and other feedback.
    """
    # Initialize results
    results = {"reps": 0, "feedback": ""}

    # Initialize tracking variables based on exercise mode
    if exercise_mode == "bicep_curl":
        bicep_curl_count = 0
        curl_state = {"LEFT": "down", "RIGHT": "down"}
    elif exercise_mode == "jumping":
        jump_data = {"count": 0, "stage": "grounded"}
    elif exercise_mode == "jumping_jacks":
        rep_data = {"count": 0, "cooldown": 0}
    elif exercise_mode == "hand_rotation":
        rotation_counters = {
                            "LEFT": {"count": 0},
                            "RIGHT": {"count": 0}
                        }
        checkpoints = {
                            "LEFT": {"current": "bottom"},
                            "RIGHT": {"current": "bottom"}
                        }
    elif exercise_mode == "sit_ups":
        sit_up_count = 0
        sit_up_state = "down"
    else:
        raise ValueError(f"Unsupported exercise mode: {exercise_mode}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Mediapipe Pose instance
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame for pose landmarks
            results_pose = pose.process(frame_rgb)
            if not results_pose.pose_landmarks:
                continue

            landmarks = results_pose.pose_landmarks.landmark

            # Call exercise-specific logic
            if exercise_mode == "bicep_curl":
                bicep_curl_count, curl_state = track_bicep_curls(
                    frame, landmarks, bicep_curl_count, curl_state,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value
                )
                results["reps"] = bicep_curl_count
            elif exercise_mode == "jumping":
                jump_data = track_jumps(frame, landmarks, jump_data)
                results["reps"] = jump_data['count']
            elif exercise_mode == "jumping_jacks":
                rep_data = track_jumping_jacks(frame, landmarks, rep_data)
                results["reps"] = rep_data["count"]
            elif exercise_mode == "hand_rotation":
                track_hand_rotation(frame, landmarks, rotation_counters, checkpoints)
                results["reps"] = rotation_counters
            elif exercise_mode == "sit_ups":
                sit_up_count, sit_up_state = track_sit_ups(frame, landmarks, sit_up_count, sit_up_state)
                results["reps"] = sit_up_count

    # Release the video capture
    cap.release()

    return results
