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


def refine_tilt_prediction(
    initial_label: str,
    diff_sequence: Sequence[Sequence[float]],
) -> str:
    """Refine raw RNN tilt predictions using keypoint-derived heuristics."""

    if not diff_sequence:
        return initial_label

    shoulder_gap = _average_vertical_gap(diff_sequence, "left_shoulder", "right_shoulder")
    ear_gap = _average_vertical_gap(diff_sequence, "left_ear", "right_ear")

    if shoulder_gap is None or ear_gap is None:
        return initial_label

    MIN_EAR_GAP = 0.02
    MIN_SHOULDER_GAP = 0.02
    HEAD_RATIO_THRESHOLD = 0.8
    SHOULDER_DOMINANCE = 1.2

    if shoulder_gap < MIN_SHOULDER_GAP and ear_gap < MIN_EAR_GAP:
        return initial_label

    if ear_gap >= MIN_EAR_GAP and ear_gap >= shoulder_gap * HEAD_RATIO_THRESHOLD:
        return "head_tilt"

    if shoulder_gap >= MIN_SHOULDER_GAP and shoulder_gap >= ear_gap * SHOULDER_DOMINANCE:
        return "shoulder_tilt"

    # 모호한 경우에는 기존 RNN 예측 유지
    return initial_label


__all__ = [
    "KEYPOINT_ORDER",
    "KEYPOINT_TO_INDEX",
    "refine_tilt_prediction",
]
