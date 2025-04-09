import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 폴더 및 CSV 파일 설정
DATASET_DIR = "posture_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
csv_file = os.path.join(DATASET_DIR, "posture_data.csv")

# CSV 파일 헤더 설정
headers = [
    "label", "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
    "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    "nose_x", "nose_y"
]

# CSV 파일이 없으면 헤더 생성
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# 초기 자세 저장 변수
initial_pose = None

# 라벨 목록
labels = [
    "good_posture", "shoulder_tilt", "forward_head", 
    "head_tilt","leaning_back"
]

def get_pose_landmarks(landmarks):
    """어깨, 귀, 눈, 코 주요 좌표 추출"""
    return {
        "left_shoulder": (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
        "right_shoulder": (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
        "left_ear": (landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y),
        "right_ear": (landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y),
        "left_eye": (landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y),
        "right_eye": (landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y),
        "nose": (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y),
    }

def compute_difference(initial, current):
    """초기 자세와 현재 자세의 차이를 계산"""
    differences = []
    for key in initial.keys():
        diff_x = current[key][0] - initial[key][0]
        diff_y = current[key][1] - initial[key][1]
        differences.extend([diff_x, diff_y])
    return differences

# 데이터 저장 함수 (CSV 파일)
def save_data(label, diff_data):
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + diff_data)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB 변환 후 분석
    frame = cv2.flip(frame, 1)  # 좌우 반전
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 현재 좌표 추출
        current_pose = get_pose_landmarks(result.pose_landmarks.landmark)

        if initial_pose is None:
            # 사용자 입력으로 초기 자세 저장
            cv2.putText(frame, "Press 's' to set initial posture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # 차이값 계산
            difference = compute_difference(initial_pose, current_pose)

            # 자세 라벨링
            cv2.putText(frame, "Press 1-7 to label posture", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        initial_pose = current_pose
        print("Initial posture saved!")
    elif key in [ord(str(i)) for i in range(1, 5)] and initial_pose is not None:
        label = labels[int(chr(key)) - 1]
        save_data(label, difference)
        print(f"Data saved for {label}")

    cv2.imshow("Posture Tracker", frame)

cap.release()
cv2.destroyAllWindows()
