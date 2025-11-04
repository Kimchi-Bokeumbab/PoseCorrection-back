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
    if shoulder_signed * ear_signed < 0:
        return initial_label

    # 귀 높이 차이가 충분히 크고 어깨보다 확실히 우세하면 head_tilt 로 간주합니다.
    if (
        ear_gap >= 0.045
        and ratio >= 1.25
        and dominance <= -0.02
    ):
        return "head_tilt"

    # 어깨 높이 차이가 귀보다 확실히 우세하고 절대값도 충분히 크면 shoulder_tilt 로 간주합니다.
    if (
        shoulder_gap >= 0.065
        and ratio <= 0.6
        and dominance >= 0.035
        and abs(shoulder_signed) >= abs(ear_signed) * 0.9
    ):
        return "shoulder_tilt"

    # 모호한 경우에는 기존 RNN 예측 유지
    return initial_label


__all__ = [
    "KEYPOINT_ORDER",
    "KEYPOINT_TO_INDEX",
    "refine_tilt_prediction",
]
