# code/server/predictor.py
import os
import sys
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===== 모델/라벨 로더 =====
from model.rnn_posture_model import RNNPostureModel
from data.posture_data import PostureDataset

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from auth import (
        authenticate_user,
        fetch_user_baseline,
        get_posture_stats,
        init_db,
        record_posture_event,
        register_user,
        store_user_baseline,
    )
    from tilt_refinement import refine_tilt_prediction
else:
    from .auth import (
        authenticate_user,
        fetch_user_baseline,
        get_posture_stats,
        init_db,
        record_posture_event,
        register_user,
        store_user_baseline,
    )
    from .tilt_refinement import refine_tilt_prediction

app = Flask(__name__)
CORS(app)  # 프론트(127.0.0.1:5173)에서 요청 허용
init_db()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
ROOT_DIR = os.path.dirname(BASE_DIR)  # PoseCorrection-back/
MODEL_PATH = os.path.join(BASE_DIR, "model", "rnn_posture_model2.pth")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset", "posture_chunk_data.csv")

# 모델
model = RNNPostureModel()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# 라벨 인코더
dummy_dataset = PostureDataset(DATASET_PATH)
label_encoder = dummy_dataset.get_label_encoder()

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


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    ok, error = register_user(email, password)
    if not ok:
        return (
            jsonify({"ok": False, "error": error}),
            400,
        )

    return jsonify({"ok": True, "message": "user_registered"})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    ok, error = authenticate_user(email, password)
    if not ok:
        status = 404 if error == "user_not_found" else 401 if error == "invalid_credentials" else 400
        return (
            jsonify({"ok": False, "error": error}),
            status,
        )

    return jsonify({"ok": True, "message": "login_success"})

# ✅ 기준 좌표 설정: 프론트에서 Mediapipe로 뽑은 7점(x,y,z) 전달
# Body: { "keypoints": [[x,y,z], ..., 7개] }
@app.route("/set_initial", methods=["POST"])
def set_initial():
    data = request.get_json(silent=True) or {}

    email = data.get("email")
    kp7 = data.get("keypoints")

    if not isinstance(email, str) or not email.strip():
        return jsonify({"ok": False, "error": "email_required"}), 400

    if not isinstance(kp7, list):
        return jsonify({"ok": False, "error": "invalid_payload", "detail": "keypoints missing/invalid"}), 400
    if len(kp7) != 7:
        return jsonify({"ok": False, "error": "invalid_landmarks", "detail": f"expected 7 keypoints, got {len(kp7)}"}), 400

    flat = flatten_kp7(kp7)
    if flat is None or len(flat) != 21:
        return jsonify({"ok": False, "error": "invalid_landmarks", "detail": f"expected 21, got {len(flat) if flat else 0}"}), 400

    saved, error = store_user_baseline(email, flat)
    if not saved:
        status = 404 if error == "user_not_found" else 400
        return jsonify({"ok": False, "error": error}), status

    print("[SET_INITIAL] stored baseline for", email)
    return jsonify({"ok": True, "message": "baseline_stored"})

# ✅ 예측: 프론트에서 최근 3프레임의 7점(x,y,z)을 보냄
# Body: { "frames": [ [[x,y,z]×7], [[x,y,z]×7], [[x,y,z]×7] ] }
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    frames = data.get("frames")
    email = data.get("email")
    recorded_at = data.get("recorded_at") or data.get("captured_at")
    score = data.get("score")

    if not isinstance(email, str) or not email.strip():
        return jsonify({"ok": False, "error": "email_required"}), 400

    baseline_21, baseline_error = fetch_user_baseline(email)
    if baseline_21 is None:
        status = 404 if baseline_error == "user_not_found" else 400
        return jsonify({"ok": False, "error": baseline_error or "baseline_missing"}), status

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

    if label in {"shoulder_tilt", "head_tilt"}:
        refined_label = refine_tilt_prediction(label, diffs)
        label = refined_label

    stored, error = record_posture_event(email, label, score=score, recorded_at=recorded_at)
    if not stored:
        status = 404 if error == "user_not_found" else 400
        return jsonify({"ok": False, "error": error}), status

    return jsonify({"ok": True, "label": label, "stored": True})


@app.route("/posture_stats", methods=["GET"])
def posture_stats():
    email = request.args.get("email")
    days_param = request.args.get("days", default="7")

    summary, error = get_posture_stats(email or "", days=days_param)
    if summary is None:
        status = 404 if error == "user_not_found" else 400
        return jsonify({"ok": False, "error": error}), status

    return jsonify({"ok": True, "summary": summary})

if __name__ == "__main__":
    # 윈도우 로컬 개발
    app.run(host="127.0.0.1", port=5000, debug=True)
