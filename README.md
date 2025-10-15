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
3. `code` 폴더로 이동 후 아래 명령어로 프로그램을 실행합니다.
    ```bash
    python -m server.predictor
    ```

## 인증 기능 상세 설명

백엔드에 아래와 같은 사용자 인증 기능이 추가되었습니다.

### 데이터 저장 방식
- 프로젝트 루트(`PoseCorrection-back/users.db`)에 SQLite 데이터베이스를 생성합니다.
- `users` 테이블에는 `email`, `password_hash`, `created_at` 필드를 저장합니다.
- 이메일은 소문자로 정규화한 후 중복 여부를 검사하고, 비밀번호는 해시(SHA-256 기반 Werkzeug 유틸리티)를 적용해 저장합니다.

### 입력 검증 및 에러 코드
- 이메일/비밀번호가 누락되었거나 문자열이 아닐 경우 `email_and_password_required` 오류를 반환합니다.
- 이메일이 공백일 때는 `email_required`, 비밀번호 길이가 6자 미만이면 `password_too_short` 오류를 반환합니다.
- 이미 가입된 이메일은 `email_already_used`, 존재하지 않는 계정 로그인 시 `user_not_found`, 비밀번호 불일치 시 `invalid_credentials` 오류를 제공합니다.

### REST API 엔드포인트
- **회원가입**: `POST /register`
  - Body: `{ "email": "user@example.com", "password": "비밀번호" }`
  - 성공 시 `{"ok": true, "message": "user_registered"}` 반환
- **로그인**: `POST /login`
  - Body: `{ "email": "user@example.com", "password": "비밀번호" }`
  - 성공 시 `{"ok": true, "message": "login_success"}` 반환

각 엔드포인트는 JSON 응답을 제공하며, 실패 시 `ok: false`와 함께 위의 에러 코드를 전달합니다. 프런트엔드는 이를 바탕으로 사용자에게 명확한 피드백을 표시할 수 있습니다.
