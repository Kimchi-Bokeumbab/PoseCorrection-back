import torch
import cv2
import mediapipe as mp
import numpy as np
from model.rnn_posture_model import RNNPostureModel
from data.posture_data import PostureDataset

# 모델 및 레이블 인코더 준비
model = RNNPostureModel()
model.load_state_dict(torch.load("code/model/rnn_posture_model2.pth"))
model.eval()

# 임시로 레이블 인코더 불러오기 (데이터셋으로부터)
dummy_dataset = PostureDataset("code/data/dataset/posture_chunk_data.csv")
label_encoder = dummy_dataset.get_label_encoder()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 초기 자세 저장
initial_pose = None
chunk = []

def extract_landmarks(landmarks):
    selected = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.NOSE,
    ]
    coords = []
    for lm in selected:
        point = landmarks[lm]
        coords.extend([point.x, point.y, point.z])
    return coords

def compute_diff(current, reference):
    return [c - r for c, r in zip(current, reference)]

cap = cv2.VideoCapture(0)
frame_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = extract_landmarks(result.pose_landmarks.landmark)

        if initial_pose is not None:
            diff = compute_diff(landmarks, initial_pose)
            frame_buffer.append(diff)

            if len(frame_buffer) == 3:
                input_tensor = torch.tensor([frame_buffer], dtype=torch.float32)  # shape: (1, 3, 21)
                output = model(input_tensor)
                pred_idx = output.argmax(dim=1).item()
                label = label_encoder.inverse_transform([pred_idx])[0]

                cv2.putText(frame, f"Posture: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                frame_buffer.pop(0)

        else:
            cv2.putText(frame, "Press 's' to set initial posture", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and result.pose_landmarks:
        initial_pose = extract_landmarks(result.pose_landmarks.landmark)
        print("Initial posture set!")

    cv2.imshow("Posture Prediction", frame)

cap.release()
cv2.destroyAllWindows()
