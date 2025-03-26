import cv2
import mediapipe as mp
import math

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    """세 점으로 각도 계산 함수"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def check_posture(landmarks):
    """어깨 수평 체크와 목 기울기 체크 함수"""
    # 어깨 랜드마크 (팔꿈치와 손목은 제외)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # 얼굴 랜드마크 (눈과 코만 사용)
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # 1️⃣ 어깨 수평 체크
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    if shoulder_diff > 0.05:  # 어깨가 수평이 아니면
        return "bad pose"  # ❌ 불량 자세

    # 2️⃣ 목 기울기 체크 (눈 위치 비교)
    eye_diff = abs(left_eye.y - right_eye.y)
    if eye_diff > 0.05:  # 목이 기울어지면
        return "bad pose"  # ❌ 불량 자세

    return "good pose"  # ✅ 좋은 자세

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우반전
        frame = cv2.flip(frame, 1)

        # BGR -> RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose 모델로 상반신 인식
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 스켈레톤 표시 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 자세 체크
            posture_check = check_posture(landmarks)
            if posture_check != "good pose":
                cv2.putText(frame, posture_check, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 영상 표시
        cv2.imshow("Posture Correction", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
