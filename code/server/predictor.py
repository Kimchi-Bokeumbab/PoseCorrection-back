# code/server/predictor.py
import os
import sys
from functools import wraps

import torch
from flask import Flask, jsonify, g, request
from flask_cors import CORS

# ===== 모델/라벨 로더 =====
from model.rnn_posture_model import RNNPostureModel
from data.posture_data import PostureDataset

app = Flask(__name__)
CORS(app)  # 프론트(127.0.0.1:5173)에서 요청 허용

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
ROOT_DIR = os.path.dirname(BASE_DIR)  # PoseCorrection-back/
MODEL_PATH = os.path.join(BASE_DIR, "model", "rnn_posture_model2.pth")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset", "posture_chunk_data.csv")

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app import auth  # noqa: E402  pylint: disable=wrong-import-position

# 모델
model = RNNPostureModel()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# 라벨 인코더
dummy_dataset = PostureDataset(DATASET_PATH)
label_encoder = dummy_dataset.get_label_encoder()

# 서버에 저장할 기준 좌표 (flatten 21 floats)
baseline_21 = None


def require_jwt(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        scheme, _, token = auth_header.partition(" ")

        if scheme.lower() != "bearer" or not token:
            return (
                jsonify({"ok": False, "error": "unauthorized", "detail": "Missing bearer token"}),
                401,
            )

        payload = auth.verify_token(token)
        if not payload:
            return (
                jsonify({"ok": False, "error": "unauthorized", "detail": "Invalid token"}),
                401,
            )

        g.jwt_payload = payload
        return fn(*args, **kwargs)

    return wrapper


def flatten_kp7(kp7):
    """ kp7: [[x,y,z], ... ×7] -> len=21 리스트로 평탄화 """
    flat = []
    for p in kp7:
        if (not isinstance(p, (list, tuple))) or len(p) != 3:
            return None
        x, y, z = p
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
            return None
        flat.extend([float(x), float(y), float(z)])
    return flat  # len=21

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": os.path.basename(MODEL_PATH)})

# ✅ 기준 좌표 설정: 프론트에서 Mediapipe로 뽑은 7점(x,y,z) 전달
# Body: { "keypoints": [[x,y,z], ..., 7개] }
@app.route("/set_initial", methods=["POST"])
@require_jwt
def set_initial():
    global baseline_21
    data = request.get_json(silent=True) or {}

    kp7 = data.get("keypoints")
    if not isinstance(kp7, list):
        return jsonify({"ok": False, "error": "invalid_payload", "detail": "keypoints missing/invalid"}), 400
    if len(kp7) != 7:
        return jsonify({"ok": False, "error": "invalid_landmarks", "detail": f"expected 7 keypoints, got {len(kp7)}"}), 400

    flat = flatten_kp7(kp7)
    if flat is None or len(flat) != 21:
        return jsonify({"ok": False, "error": "invalid_landmarks", "detail": f"expected 21, got {len(flat) if flat else 0}"}), 400

    baseline_21 = flat
    print("[SET_INITIAL] baseline_21 len:", len(baseline_21))
    print("[SET_INITIAL] sample:", baseline_21[:6], "...")
    return jsonify({"ok": True, "message": "baseline set"})

# ✅ 예측: 프론트에서 최근 3프레임의 7점(x,y,z)을 보냄
# Body: { "frames": [ [[x,y,z]×7], [[x,y,z]×7], [[x,y,z]×7] ] }
@app.route("/predict", methods=["POST"])
@require_jwt
def predict():
    global baseline_21
    if baseline_21 is None:
        return jsonify({"ok": False, "error": "no_baseline", "detail": "call /set_initial first"}), 400

    data = request.get_json(silent=True) or {}
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != 3:
        return jsonify({"ok": False, "error": "invalid_frames", "detail": "frames must be length=3"}), 400

    # 각 프레임 7×3 → 21 플랫 후 baseline과 차분
    diffs = []
    for i, f in enumerate(frames):
        if not isinstance(f, list) or len(f) != 7:
            return jsonify({"ok": False, "error": "invalid_frame", "detail": f"frame[{i}] must have 7 keypoints"}), 400
        flat = flatten_kp7(f)
        if flat is None or len(flat) != 21:
            return jsonify({"ok": False, "error": "invalid_frame", "detail": f"frame[{i}] expected 21, got {len(flat) if flat else 0}"}), 400
        diff21 = [c - r for c, r in zip(flat, baseline_21)]
        diffs.append(diff21)

    # 텐서 shape: (1, 3, 21)
    x = torch.tensor([diffs], dtype=torch.float32)  # 1×3×21
    with torch.no_grad():
        out = model(x)
        pred_idx = out.argmax(dim=1).item()
        label = label_encoder.inverse_transform([pred_idx])[0]

    return jsonify({"ok": True, "label": label})

if __name__ == "__main__":
    # 윈도우 로컬 개발
    app.run(host="127.0.0.1", port=5000, debug=True)
