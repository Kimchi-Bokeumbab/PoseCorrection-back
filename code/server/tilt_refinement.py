"""Utilities for refining shoulder/head tilt predictions with rule-based heuristics."""
from __future__ import annotations

from typing import Sequence

KEYPOINT_ORDER = [
    "left_shoulder",
    "right_shoulder",
    "left_ear",
    "right_ear",
    "left_eye",
    "right_eye",
    "nose",
]

KEYPOINT_TO_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_ORDER)}


def _average_vertical_gap(
    diff_sequence: Sequence[Sequence[float]],
    left_key: str,
    right_key: str,
) -> float | None:
    """Return the mean vertical distance between two keypoints across frames.

    Args:
        diff_sequence: Iterable of per-frame baseline diffs with length 21 (flattened
            XYZ values for the 7 keypoints).
        left_key: Name of the left-side keypoint (must exist in ``KEYPOINT_TO_INDEX``).
        right_key: Name of the right-side keypoint (must exist in ``KEYPOINT_TO_INDEX``).

    Returns:
        Average absolute difference between the Y coordinates of the two keypoints.
        Returns ``None`` if the provided data is malformed.
    """

    left_idx = KEYPOINT_TO_INDEX[left_key] * 3 + 1  # dy 위치
    right_idx = KEYPOINT_TO_INDEX[right_key] * 3 + 1

    values = []
    for frame_diff in diff_sequence:
        try:
            left_dy = frame_diff[left_idx]
            right_dy = frame_diff[right_idx]
        except (TypeError, IndexError):
            return None
        values.append(abs(left_dy - right_dy))

    if not values:
        return 0.0
    return sum(values) / len(values)


def _average_signed_delta(
    diff_sequence: Sequence[Sequence[float]],
    left_key: str,
    right_key: str,
) -> float | None:
    """Return the mean signed height delta between two keypoints."""

    left_idx = KEYPOINT_TO_INDEX[left_key] * 3 + 1
    right_idx = KEYPOINT_TO_INDEX[right_key] * 3 + 1

    values = []
    for frame_diff in diff_sequence:
        try:
            left_dy = frame_diff[left_idx]
            right_dy = frame_diff[right_idx]
        except (TypeError, IndexError):
            return None
        values.append(left_dy - right_dy)

    if not values:
        return 0.0
    return sum(values) / len(values)


def refine_tilt_prediction(
    initial_label: str,
    diff_sequence: Sequence[Sequence[float]],
    *,
    initial_margin: float | None = None,
) -> str:
    """Refine raw RNN tilt predictions using keypoint-derived heuristics.

    Args:
        initial_label: Label predicted by the neural network.
        diff_sequence: Baseline-subtracted keypoint differences for each frame.
        initial_margin: Difference between the top-1 and top-2 softmax
            probabilities for the raw prediction. When the model is confident
            (margin above ~0.12), the heuristic keeps the neural network output
            untouched to avoid harming accuracy.
    """

    if not diff_sequence:
        return initial_label

    if initial_margin is not None and initial_margin >= 0.12:
        # RNN has a confident margin between the top-2 classes; respect it.
        return initial_label

    shoulder_gap = _average_vertical_gap(diff_sequence, "left_shoulder", "right_shoulder")
    ear_gap = _average_vertical_gap(diff_sequence, "left_ear", "right_ear")
    shoulder_signed = _average_signed_delta(diff_sequence, "left_shoulder", "right_shoulder")
    ear_signed = _average_signed_delta(diff_sequence, "left_ear", "right_ear")

    if (
        shoulder_gap is None
        or ear_gap is None
        or shoulder_signed is None
        or ear_signed is None
    ):
        return initial_label

    # 노이즈를 제거하기 위해 양쪽 모두 거의 움직임이 없으면 RNN 결과를 그대로 사용합니다.
    max_gap = max(shoulder_gap, ear_gap)
    if max_gap < 0.03:
        return initial_label

    ratio = ear_gap / (shoulder_gap + 1e-6)
    dominance = shoulder_gap - ear_gap

    # 귀와 어깨가 상반된 방향으로 움직였다면 신뢰하지 않습니다.
    if shoulder_signed * ear_signed < -0.005:
        return initial_label

    head_score = 0.0
    shoulder_score = 0.0

    # 귀/어깨 간 절대 높이 차이를 기반으로 점수를 계산합니다.
    if ear_gap >= 0.04:
        head_score += 1.0
    if ear_gap >= 0.055:
        head_score += 0.5
    if shoulder_gap >= 0.055:
        shoulder_score += 1.0
    if shoulder_gap >= 0.08:
        shoulder_score += 0.5

    # 비율이 특정 방향으로 크게 치우칠수록 해당 점수를 증가시킵니다.
    if ratio >= 1.2:
        head_score += 1.0
    if ratio >= 1.5:
        head_score += 0.5
    if ratio <= 0.8:
        shoulder_score += 0.5
    if ratio <= 0.55:
        shoulder_score += 1.0

    # dominance는 어깨 대비 귀의 우세 여부를 측정합니다.
    if dominance <= -0.025:
        head_score += 1.0
    if dominance <= -0.055:
        head_score += 0.5
    if dominance >= 0.025:
        shoulder_score += 1.0
    if dominance >= 0.055:
        shoulder_score += 0.5

    # signed delta의 상대적 크기가 동일 방향으로 유지되는지를 확인합니다.
    abs_ear = abs(ear_signed)
    abs_shoulder = abs(shoulder_signed)

    if abs_ear >= abs_shoulder * 1.15:
        head_score += 0.75
    if abs_ear >= abs_shoulder * 1.35:
        head_score += 0.25
    if abs_shoulder >= abs_ear * 1.15:
        shoulder_score += 0.75
    if abs_shoulder >= abs_ear * 1.35:
        shoulder_score += 0.25

    if abs_ear >= 0.035:
        head_score += 0.5
    if abs_ear >= 0.05:
        head_score += 0.25
    if abs_shoulder >= 0.045:
        shoulder_score += 0.5
    if abs_shoulder >= 0.065:
        shoulder_score += 0.25

    # 점수 차이가 충분히 크고 일정 기준 이상일 때만 덮어씁니다.
    override_label = initial_label
    if head_score >= 3.0 and head_score >= shoulder_score + 0.75:
        override_label = "head_tilt"
    elif shoulder_score >= 3.0 and shoulder_score >= head_score + 0.75:
        override_label = "shoulder_tilt"

    # 모호한 경우에는 기존 RNN 예측 유지
    return override_label


__all__ = [
    "KEYPOINT_ORDER",
    "KEYPOINT_TO_INDEX",
    "refine_tilt_prediction",
]
