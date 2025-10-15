# PoseCorrection-back
# 자세교정 AI 프로그램

## 프로젝트 소개
본 프로젝트는 AI를 활용한 자세교정 프로그램을 개발하기 위해 진행하였습니다. 사용자는 프로그램을 활용하여 자신이 습관적으로 취하는 옳지 못한 자세를 교정할 수 있도록 도움을 줍니다.
그외 자세 데이터 분석 및 스트레칭 추천을 통하여 보다 효과적인 자세 교정을 진행할 수 있습니다.

## 팀원 소개
### - 송호진
- **주요 역할**:
- **연락처**:
### - 김진혁
- **주요 역할**:
- **연락처**:
### - 박찬호
- **주요 역할**:
- **연락처**:

## 주요 기능
- **실시간 자세 분석**: 웹캠을 사용하여 사용자의 자세를 실시간으로 분석.
- **자세 교정 피드백**: 잘못된 자세를 감지하여 올바른 자세를 제안.
- **기록 기능**: 자세 개선 이력을 저장 및 추적.
- **사용자 친화적인 인터페이스**: 직관적인 UI 제공.

## 기술 스택
- **프로그래밍 언어**: Python
- **AI 모델**: [사용한 딥러닝 프레임워크 이름 (예: TensorFlow, PyTorch)]
- **이미지 처리**: OpenCV
- **웹캠 사용**: MediaPipe 또는 기타 관련 라이브러리

## 설치 및 실행 방법
1. 프로젝트를 클론합니다.
    ```bash
    git clone https://github.com/사용자명/프로젝트명.git
    cd 프로젝트명
    ```
2. 필요한 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
3. 프로젝트 루트에서 `code` 폴더로 이동한 뒤 아래 명령어로 서버를 실행합니다.
    ```bash
    cd code
    python -m server.predictor
    ```
    위 명령으로 Flask 서버가 구동되면 `/register`·`/login` 엔드포인트와 함께
    예측 및 통계용 API가 활성화되며, 첫 실행 시 자동으로 `../users.db`
    데이터베이스 파일이 생성됩니다.

## 인증 및 자세 기록 기능 상세 설명

백엔드는 사용자 계정뿐 아니라 예측 결과를 시간대별로 저장해 프런트에서
시각화할 수 있도록 아래와 같은 구조를 제공합니다.

### 데이터 저장 방식
- 프로젝트 루트(`PoseCorrection-back/users.db`)에 SQLite 데이터베이스를 생성합니다.
- `users` 테이블에는 `email`, `password_hash`, `created_at` 필드를 저장합니다.
- `posture_logs` 테이블에는 각 예측에 대한 `user_id`, `posture_label`,
  선택적 `score`, `recorded_at`(ISO8601) 값을 저장합니다.
- 이메일은 소문자로 정규화한 후 중복 여부를 검사하고, 비밀번호는 해시(SHA-256 기반 Werkzeug 유틸리티)를 적용해 저장합니다.

### 입력 검증 및 에러 코드
- 이메일/비밀번호가 누락되었거나 문자열이 아닐 경우 `email_and_password_required` 오류를 반환합니다.
- 이메일이 공백일 때는 `email_required`, 비밀번호 길이가 6자 미만이면 `password_too_short` 오류를 반환합니다.
- 이미 가입된 이메일은 `email_already_used`, 존재하지 않는 계정 로그인 시 `user_not_found`, 비밀번호 불일치 시 `invalid_credentials` 오류를 제공합니다.
- `/predict` 요청에서 좌표 길이가 잘못되면 `invalid_frames` 또는 `invalid_frame`, 기준 자세가 없으면 `no_baseline` 오류를 반환합니다.
- 예측 결과를 저장할 때 이메일이 없으면 `email_required`, 점수 형식이 잘못되면 `invalid_score`, 시간 포맷이 맞지 않으면 `invalid_timestamp` 오류가 발생합니다.
- `/posture_stats`에서 기간 파라미터가 숫자가 아니거나 0 이하이면 `invalid_days` 오류가 발생합니다.

### REST API 엔드포인트
- **회원가입**: `POST /register`
  - Body: `{ "email": "user@example.com", "password": "비밀번호" }`
  - 성공 시 `{"ok": true, "message": "user_registered"}` 반환
- **로그인**: `POST /login`
  - Body: `{ "email": "user@example.com", "password": "비밀번호" }`
  - 성공 시 `{"ok": true, "message": "login_success"}` 반환
- **자세 예측 및 기록**: `POST /predict`
  - Body 예시:
    ```json
    {
      "email": "user@example.com",
      "frames": [
        [[x, y, z] * 7],
        [[x, y, z] * 7],
        [[x, y, z] * 7]
      ],
      "recorded_at": "2024-05-01T13:30:00",
      "score": 72.3
    }
    ```
  - `email`은 필수이며 가입된 사용자여야 합니다.
  - `frames`는 최근 3프레임(각 7개 관절)의 좌표를 전달합니다.
  - `recorded_at`은 생략 시 서버 시간이 기록됩니다. `score`는 프런트에서 계산한
    (선택적) 자세 점수입니다.
  - 정상 동작 시 `{ "ok": true, "label": "거북목", "stored": true }`와 같이
    분류 라벨을 반환하고, 같은 정보가 데이터베이스에 저장됩니다.
- **저장 데이터 통계 조회**: `GET /posture_stats?email=user@example.com&days=7`
  - 가입된 사용자의 최근 `days`일(기본 7일) 동안 기록을 집계합니다.
  - 응답 예시:
    ```json
    {
      "ok": true,
      "summary": {
        "range": {"start": "2024-04-25T00:00:00", "end": "2024-05-02T00:00:00"},
        "hourly": [
          {"hour": "08:00", "total": 5, "bad": 2, "avg_score": 74.5},
          {"hour": "09:00", "total": 3, "bad": 1}
        ],
        "weekday": [
          {"weekday": "월", "total": 8, "bad": 3},
          {"weekday": "화", "total": 6, "bad": 1}
        ],
        "labels": [
          {"label": "정상", "count": 12},
          {"label": "거북목", "count": 5}
        ],
        "total_events": 20
      }
    }
    ```
  - `hourly`·`weekday`·`labels` 배열을 이용해 프런트에서 시간대/요일/자세별
    차트를 구성할 수 있습니다.

각 엔드포인트는 JSON 응답을 제공하며, 실패 시 `ok: false`와 함께 위의 에러 코드를 전달합니다. 프런트엔드는 이를 바탕으로 사용자에게 명확한 피드백을 표시할 수 있습니다.
