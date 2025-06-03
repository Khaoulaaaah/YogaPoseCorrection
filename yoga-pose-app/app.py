import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tempfile
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


# Configuration MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# Feedback couleurs
COLOR_MAP = {"Good": (0, 255, 0), "Acceptable": (0, 165, 255), "Bad": (0, 0, 255)}

# Angles à analyser
POSE_POINTS = {
    "left_elbow_angle": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow_angle": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder_angle": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder_angle": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_knee_angle": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee_angle": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
}

# Fonction pour calculer un angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Extraire les angles du corps
def get_landmarks_and_angles(results, frame):
    if not results.pose_landmarks:
        return {}, {}

    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks.landmark
    angles, coords = {}, {}

    def p(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return [lm.x, lm.y], (int(lm.x * w), int(lm.y * h))

    for key, (a, b, c) in POSE_POINTS.items():
        pa, ca = p(a)
        pb, cb = p(b)
        pc, cc = p(c)
        angles[key] = calculate_angle(pa, pb, pc)
        coords[key] = cb

    return angles, coords

# Comparer avec l'angle idéal
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

# === Interface Streamlit ===
st.set_page_config(page_title="Correcteur Yoga", layout="centered")
st.title("🧘 Correcteur de posture yoga")

poses = [
    "ArdhaChandrasana", "Downward Dog", "Triangle", "Veerabhadrasana",
    "Natarajasana", "Vrukshasana", "BaddhaKonasana", "UtkataKonasana"
]

pose_name = st.selectbox("Choisissez une posture :", poses)

# Afficher démonstration
demo_path = f"yoga-pose-app/Demo/{pose_name}.mp4"
image_path = f"yoga-pose-app/Demo/{pose_name}.jpg"

if os.path.exists(demo_path):
    st.video(demo_path)
elif os.path.exists(image_path):
    st.image(image_path, caption="Démonstration", use_column_width=True)
else:
    st.warning("Aucune démonstration disponible pour cette posture.")

uploaded_file = st.file_uploader("📤 Uploadez une vidéo de votre posture", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Charger les angles idéaux
    csv_path = f"Results/Dataset_{pose_name}_Angles.csv"
    if not os.path.exists(csv_path):
        st.error(f"CSV non trouvé pour {pose_name}")
    else:
        ideal_df = pd.read_csv(csv_path)
        ideal_angles = ideal_df.iloc[0].to_dict()

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            user_angles, coords = get_landmarks_and_angles(results, frame)
            evaluations = compare_angles(user_angles, ideal_angles)

            drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for joint, eval in evaluations:
                coord = coords.get(joint)
                if coord:
                    color = COLOR_MAP[eval]
                    cv2.circle(frame, coord, 10, color, -1)
                    (tw, th), _ = cv2.getTextSize(eval, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (coord[0] + 10, coord[1] - 25),
                                        (coord[0] + 10 + tw, coord[1] - 5), (0, 0, 0), -1)
                    cv2.putText(frame, eval, (coord[0] + 10, coord[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        os.unlink(video_path)
