import cv2
import mediapipe as mp
import torch
import numpy as np
from model.posture_classifier import PostureClassifier

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# PyTorch 모델 로드
model = PostureClassifier()
model.load_state_dict(torch.load("posture_model.pth", map_location=torch.device('cpu')))
model.eval()

# 라벨 목록
labels = ["good_posture", "shoulder_tilt", "forward_head", "head_tilt", "leaning_back"]

# 초기 자세 저장용 변수
initial_pose = None

def get_pose_landmarks(landmarks):
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
    differences = []
    for key in initial.keys():
        diff_x = current[key][0] - initial[key][0]
        diff_y = current[key][1] - initial[key][1]
        differences.extend([diff_x, diff_y])
    return differences

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_pose = get_pose_landmarks(result.pose_landmarks.landmark)

        if initial_pose is None:
            cv2.putText(frame, "Press 's' to set initial posture", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            difference = compute_difference(initial_pose, current_pose)
            input_tensor = torch.tensor(difference).float().unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                predicted = torch.argmax(output, dim=1).item()
                posture = labels[predicted]
                cv2.putText(frame, f"Predicted: {posture}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and result.pose_landmarks:
        initial_pose = get_pose_landmarks(result.pose_landmarks.landmark)
        print("Initial posture saved!")

    cv2.imshow("Posture Predictor", frame)

cap.release()
cv2.destroyAllWindows()
