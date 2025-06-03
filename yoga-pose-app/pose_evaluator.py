import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

COLOR_MAP = {
    "Good": (0, 255, 0),
    "Acceptable": (0, 165, 255),
    "Bad": (0, 0, 255)
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_landmarks_and_angles(results, frame):
    if not results.pose_landmarks:
        return {}, {}

    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks.landmark
    angles, coords = {}, {}

    def p(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return [lm.x, lm.y], (int(lm.x * w), int(lm.y * h))

    points = {
        "left_elbow_angle": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "right_elbow_angle": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        "left_shoulder_angle": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
        "right_shoulder_angle": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
        "left_knee_angle": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right_knee_angle": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
    }

    for key, (a, b, c) in points.items():
        pa, ca = p(a)
        pb, cb = p(b)
        pc, cc = p(c)
        angles[key] = calculate_angle(pa, pb, pc)
        coords[key] = cb

    return angles, coords

def compare_angles(user_angles, ideal_angles):
    evaluations = []
    for joint in ideal_angles:
        if joint not in user_angles:
            continue
        diff = user_angles[joint] - ideal_angles[joint]
        if abs(diff) <= 20:
            eval = "Good"
        elif abs(diff) <= 40:
            eval = "Acceptable"
        else:
            eval = "Bad"
        evaluations.append((joint, eval))
    return evaluations

def evaluate_pose(video_path, pose_name, output_path):
    csv_path = f"Results/Dataset_{pose_name}_Angles.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found for {pose_name}")

    ideal_angles = pd.read_csv(csv_path).iloc[0].to_dict()
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)

        user_angles, coords = get_landmarks_and_angles(results, frame)
        evaluations = compare_angles(user_angles, ideal_angles)

        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for joint, eval in evaluations:
            coord = coords.get(joint)
            if coord:
                color = COLOR_MAP[eval]
                cv2.circle(frame, coord, 10, color, -1)
                cv2.putText(frame, eval, (coord[0]+10, coord[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path
