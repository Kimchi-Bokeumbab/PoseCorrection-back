import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 경로 및 라벨 정의
DATASET_DIR = "code/data/dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
csv_file = os.path.join(DATASET_DIR, "posture_chunk_data.csv")

labels = [
    "good_posture", "shoulder_tilt", "forward_head", 
    "head_tilt", "leaning_back"
]

coord_names = [
    "left_shoulder", "right_shoulder",
    "left_ear", "right_ear",
    "left_eye", "right_eye",
    "nose"
]

# 헤더 정의
headers = ["label"]
for t in range(3):  # 3 프레임
    for name in coord_names:
        headers += [f"{name}_dx_t{t}", f"{name}_dy_t{t}", f"{name}_dz_t{t}"]

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def get_pose_landmarks(landmarks):
    return {
        "left_shoulder": (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z),
        "right_shoulder": (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z),
        "left_ear": (landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_EAR].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_EAR].z),
        "right_ear": (landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_EAR].z),
        "left_eye": (landmarks[mp_pose.PoseLandmark.LEFT_EYE].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_EYE].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_EYE].z),
        "right_eye": (landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_EYE].z),
        "nose": (landmarks[mp_pose.PoseLandmark.NOSE].x,
                 landmarks[mp_pose.PoseLandmark.NOSE].y,
                 landmarks[mp_pose.PoseLandmark.NOSE].z),
    }

def compute_difference(initial, current):
    diff = {}
    for key in coord_names:
        dx = current[key][0] - initial[key][0]
        dy = current[key][1] - initial[key][1]
        dz = current[key][2] - initial[key][2]
        diff[key] = (dx, dy, dz)
    return diff

def save_chunk(label, diff_chunk):
    flat = []
    for pose in diff_chunk:
        for key in coord_names:
            flat.extend([pose[key][0], pose[key][1], pose[key][2]])
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + flat)

# 초기화
initial_pose = None
cap = cv2.VideoCapture(0)
collecting = False
frame_buffer = []
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_pose = get_pose_landmarks(result.pose_landmarks.landmark)

        # 초기자세 설정
        if initial_pose is None:
            cv2.putText(frame, "Press 's' to set initial posture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 수집 중이면 버퍼에 저장
        elif collecting:
            diff = compute_difference(initial_pose, current_pose)
            frame_buffer.append(diff)

            elapsed = time.time() - start_time
            cv2.putText(frame, f"Collecting: {elapsed:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

            # 3초 넘으면 저장
            if elapsed >= 3.0:
                collecting = False
                n = len(frame_buffer)
                if n >= 3:
                    idx1 = 0
                    idx2 = n // 2
                    idx3 = n - 1
                    selected = [frame_buffer[idx1], frame_buffer[idx2], frame_buffer[idx3]]
                    save_chunk(current_label, selected)
                    print(f"✅ Saved evenly spaced 3-frame chunk for label: {current_label}")
                else:
                    print("⚠️ Not enough frames collected. Try again.")
                frame_buffer = []

        else:
            cv2.putText(frame, "Press 1~5 to collect 3-sec chunk", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and result.pose_landmarks:
        initial_pose = get_pose_landmarks(result.pose_landmarks.landmark)
        print("✅ Initial posture saved!")
    elif key in [ord(str(i)) for i in range(1, 6)] and initial_pose is not None:
        current_label = labels[int(chr(key)) - 1]
        collecting = True
        start_time = time.time()
        frame_buffer = []
        print(f"⏺️ Started collecting for: {current_label}")

    cv2.imshow("Posture 3-Second Chunk Collector", frame)

cap.release()
cv2.destroyAllWindows()
