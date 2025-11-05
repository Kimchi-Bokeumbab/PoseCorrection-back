# PoseCorrection-back
# 자세교정 AI 프로그램

## 프로젝트 소개
본 프로젝트는 AI를 활용한 자세교정 프로그램을 개발하기 위해 진행하였습니다. 사용자는 프로그램을 활용하여 자신이 습관적으로 취하는 옳지 못한 자세를 교정할 수 있도록 도움을 줍니다.
그외 자세 데이터 분석 및 스트레칭 추천을 통하여 보다 효과적인 자세 교정을 진행할 수 있습니다.

## 팀원 소개
### - 송호진
### - 김진혁
### - 박찬호

## 주요 기능
- **실시간 자세 분석**: 웹캠을 사용하여 사용자의 자세를 실시간으로 분석.
- **자세 교정 피드백**: 잘못된 자세를 감지하여 올바른 자세를 제안.
- **기록 기능**: 자세 개선 이력을 저장 및 추적.

## 기술 스택
- **프로그래밍 언어**: Python
- **AI 모델**: [사용한 딥러닝 프레임워크 이름 (예: TensorFlow, PyTorch)]
- **이미지 처리**: OpenCV
- **웹캠 사용**: MediaPipe 또는 기타 관련 라이브러리

## 디렉터리 구조

```
PoseCorrection-back/
├── README.md              # 현재 문서
├── requirements.txt       # 백엔드 실행에 필요한 Python 패키지 목록
├── code/
│   ├── server/            # Flask 서버 및 인증/통계 로직
│   ├── model/             # RNNPostureModel 정의 및 학습된 가중치(.pth)
│   ├── data/              # PostureDataset 정의 및 학습용 CSV 데이터
│   ├── train.py           # RNN 모델 학습 스크립트
│   └── estimate_pose.py   # (보조) 포즈 추정 관련 스크립트
├── img/                   # 문서나 시연용 이미지 자산
└── proto/                 # gRPC/Protobuf 정의 (필요 시 사용)
```

## 개발 환경 준비

1. **필수 조건**
   - Python 3.10 이상 (PyTorch 및 Mediapipe 호환 버전 권장)
   - 가상환경 사용 권장 (`venv`, `conda` 등)
2. **프로젝트 클론 및 패키지 설치**
   ```bash
   git clone https://github.com/<YOUR_ORG>/PoseCorrection-back.git
   cd PoseCorrection-back
   python -m venv .venv           # 선택 사항
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **환경 변수**
   - 기본 설정은 필요하지 않으며, Flask 서버가 실행되면 자동으로 `users.db` 파일을 생성합니다.

## 서버 실행

```bash
cd code
python -m server.predictor
```

- 기본 실행 포트는 `5000`이며 CORS는 로컬 프런트엔드(예: `http://127.0.0.1:5173`)를 허용하도록 설정되어 있습니다.
- 첫 실행 시 프로젝트 루트(`PoseCorrection-back/users.db`)에 SQLite 데이터베이스가 생성됩니다.
- 서버가 정상적으로 구동되면 `/health` 엔드포인트에서 모델 이름과 상태를 확인할 수 있습니다.

## 데이터베이스 구조

| 테이블 | 주요 필드 | 설명 |
| --- | --- | --- |
| `users` | `email`, `password_hash`, `created_at` | 사용자 계정 정보 저장. 이메일은 소문자로 정규화하고, 비밀번호는 Werkzeug SHA-256 해시를 사용합니다. |
| `user_baselines` | `baseline`, `created_at`, `updated_at` | 7개 관절(21개의 float) 기준 자세를 JSON 문자열로 저장하며, 사용자별로 1개만 유지됩니다. |
| `posture_logs` | `posture_label`, `score`, `recorded_at` | 예측 결과를 시간대별로 기록하고, 점수가 제공되면 평균 점수 계산에 활용합니다. |

## REST API

- 모든 엔드포인트는 JSON을 주고받으며, 실패 시 `{"ok": false, "error": <코드>}` 형태로 응답합니다.

### 인증 및 사용자 관리

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `POST` | `/register` | `{ "email", "password" }`를 받아 신규 계정을 생성합니다. 비밀번호는 최소 6자입니다. |
| `POST` | `/login` | 등록된 이메일과 비밀번호로 로그인합니다. 잘못된 자격증명은 `401`, 존재하지 않는 계정은 `404`를 반환합니다. |

### 기준 자세 및 예측

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `POST` | `/set_initial` | 7개의 3차원 좌표 배열(`keypoints`)을 받아 사용자 기준 자세를 저장합니다. 기존 데이터가 있으면 갱신합니다. |
| `POST` | `/predict` | 최근 3프레임의 7개 관절 좌표(`frames` 배열)와 선택적 `score`, `recorded_at`을 받아 자세 라벨을 예측하고 저장합니다. |
| `GET` | `/posture_stats` | `email`과 `days`(기본 7)를 쿼리로 받아 기간별 통계를 반환합니다. |

자세 예측 요청 예시는 다음과 같습니다.

```json
{
  "email": "user@example.com",
  "frames": [
    [[0.1, 0.2, 0.0], ... 7개],
    [[0.0, 0.3, -0.1], ... 7개],
    [[-0.1, 0.1, 0.05], ... 7개]
  ],
  "recorded_at": "2024-05-01T13:30:00",
  "score": 72.3
}
```

`shoulder_tilt` 또는 `head_tilt`로 분류된 경우 추가 규칙(`tilt_refinement.py`)을 적용해 결과를 보정합니다.

## 모델 학습 및 평가

### RNN 모델 학습

```bash
python code/train.py
```

- `code/data/dataset/posture_chunk_data.csv`를 불러와 학습/검증 데이터로 분리합니다.
- 학습 과정에서 `code/model/rnn_posture_model2.pth` 파일을 갱신하고, 에폭별 Loss/Accuracy를 출력 및 그래프로 시각화합니다.

### 규칙 보정 평가

훈련된 모델과 동일한 데이터셋을 사용해 규칙 기반 기울기 보정(`tilt_refinement`)이 성능에 미치는 영향을 평가할 수 있습니다.

```bash
python -m server.evaluate_rule_based_accuracy --batch-size 256
```

출력에는 RNN 단독 정확도와 보정 후 정확도, 보정에 의해 수정된 케이스 수 등이 포함됩니다.
