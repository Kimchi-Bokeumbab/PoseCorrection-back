import cv2
import mediapipe as mp
import torch
from model.posture_classifier import PostureClassifier

# 라벨 맵핑
label_map_reverse = {
    0: "good_posture",
    1: "shoulder_tilt",
    2: "forward_head",
    3: "head_tilt",
    4: "leaning_back"
}

# 모델 로드
model = PostureClassifier()
model.load_state_dict(torch.load('posture_model.pth'))
model.eval()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe 좌표 추출
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 필요한 관절 좌표 추출
        points = [
            landmarks[11].x, landmarks[11].y,  # left_shoulder
            landmarks[12].x, landmarks[12].y,  # right_shoulder
            landmarks[7].x, landmarks[7].y,    # left_ear
            landmarks[8].x, landmarks[8].y,    # right_ear
            landmarks[1].x, landmarks[1].y,    # left_eye
            landmarks[2].x, landmarks[2].y,    # right_eye
            landmarks[0].x, landmarks[0].y     # nose
        ]

        input_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            label = label_map_reverse[predicted]

        # 화면 출력
        cv2.putText(frame, f"Posture: {label}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 랜드마크도 화면에 그림
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Posture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
