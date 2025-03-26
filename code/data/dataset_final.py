import cv2
import mediapipe as mp
import csv
import os

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# CSV 파일 준비
csv_file = "code/data/posture_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
                         "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
                         "left_eye_x", "left_eye_y", "left_eye_z",
                         "right_eye_x", "right_eye_y", "right_eye_z", "label"])

# 웹캠 실행
cap = cv2.VideoCapture(0)

print("Press 'g' to save as Good Posture, 'b' for Bad Posture, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 랜드마크 좌표 추출
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

        # 좌표 (x, y, z)
        left_shoulder_x, left_shoulder_y, left_shoulder_z = left_shoulder.x, left_shoulder.y, left_shoulder.z
        right_shoulder_x, right_shoulder_y, right_shoulder_z = right_shoulder.x, right_shoulder.y, right_shoulder.z
        left_eye_x, left_eye_y, left_eye_z = left_eye.x, left_eye.y, left_eye.z
        right_eye_x, right_eye_y, right_eye_z = right_eye.x, right_eye.y, right_eye.z

        # 어깨 및 눈에 점 표시
        h, w, _ = frame.shape
        cv2.circle(frame, (int(left_shoulder_x * w), int(left_shoulder_y * h)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(right_shoulder_x * w), int(right_shoulder_y * h)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(left_eye_x * w), int(left_eye_y * h)), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_eye_x * w), int(right_eye_y * h)), 5, (0, 255, 0), -1)

        # 화면에 정보 표시
        cv2.putText(frame, "Press 'g' for Good, 'b' for Bad", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        label = None

        if key == ord('g'):
            label = "good"
        elif key == ord('b'):
            label = "bad"
        elif key == ord('q'):
            break

        # 데이터 저장
        if label:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([left_shoulder_x, left_shoulder_y, left_shoulder_z,
                                 right_shoulder_x, right_shoulder_y, right_shoulder_z,
                                 left_eye_x, left_eye_y, left_eye_z,
                                 right_eye_x, right_eye_y, right_eye_z, label])
            print(f"Saved posture data as {label}")

    cv2.imshow("Posture Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
