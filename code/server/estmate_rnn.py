"""Compute F1 scores and confusion matrices for the RNN with and without heuristics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    SERVER_DIR = Path(__file__).resolve().parent
    CODE_DIR = SERVER_DIR.parent
    ROOT_DIR = CODE_DIR.parent
    for path in (str(ROOT_DIR), str(CODE_DIR)):
        if path not in sys.path:
            sys.path.append(path)
    from data.posture_data import PostureDataset
    from model.rnn_posture_model import RNNPostureModel
    from tilt_refinement import refine_tilt_prediction
else:  # pragma: no cover
    from ..data.posture_data import PostureDataset
    from ..model.rnn_posture_model import RNNPostureModel
    from .tilt_refinement import refine_tilt_prediction


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parents[1]
    default_model = code_dir / "model" / "rnn_posture_model2.pth"
    default_dataset = code_dir / "data" / "dataset" / "posture_chunk_data.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model, help="Path to the trained RNN weights (.pth)")
    parser.add_argument("--dataset-path", type=Path, default=default_dataset, help="CSV dataset used for evaluation")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for dataloader")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (e.g. 'cpu', 'cuda:0')")
    parser.add_argument(
        "--average",
        type=str,
        default="macro",
        choices=["micro", "macro", "weighted", "none"],
        help="Averaging strategy for the reported F1 score (set to 'none' for per-class values)",
    )
    return parser.parse_args()


def _collect_predictions(
    dataloader: DataLoader,
    model: RNNPostureModel,
    device: torch.device,
    label_encoder,
) -> tuple[list[int], list[int], list[int]]:
    all_targets: list[int] = []
    all_rnn_preds: list[int] = []
    all_refined_preds: list[int] = []

    tilt_labels = {"shoulder_tilt", "head_tilt"}

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_inputs)
            raw_predictions = logits.argmax(dim=1)
            probabilities = torch.softmax(logits, dim=1)

            if probabilities.size(1) >= 2:
                top2 = torch.topk(probabilities, k=2, dim=1)
                margins = top2.values[:, 0] - top2.values[:, 1]
            else:
                margins = torch.ones(probabilities.size(0), device=probabilities.device)

            all_targets.extend(batch_targets.detach().cpu().tolist())
            all_rnn_preds.extend(raw_predictions.detach().cpu().tolist())

            raw_labels = label_encoder.inverse_transform(raw_predictions.detach().cpu().numpy())
            inputs_cpu = batch_inputs.detach().cpu()
            margins_cpu = margins.detach().cpu()

            refined_indices: list[int] = []
            for idx, raw_label in enumerate(raw_labels):
                refined_label = raw_label
                if raw_label in tilt_labels:
                    diff_sequence: Sequence[Sequence[float]] = inputs_cpu[idx].tolist()
                    margin = margins_cpu[idx].item()
                    refined_label = refine_tilt_prediction(
                        raw_label,
                        diff_sequence,
                        initial_margin=margin,
                    )
                refined_indices.append(label_encoder.transform([refined_label])[0])

            all_refined_preds.extend(refined_indices)

    return all_targets, all_rnn_preds, all_refined_preds


def _print_f1_scores(
    average: str,
    labels: list[int],
    target_names: list[str],
    y_true: np.ndarray,
    y_pred_rnn: np.ndarray,
    y_pred_refined: np.ndarray,
) -> None:
    print("=== F1 Scores ===")
    average_param: str | None = None if average == "none" else average
    if average_param is None:
        rnn_scores = f1_score(y_true, y_pred_rnn, labels=labels, average=None, zero_division=0)
        refined_scores = f1_score(y_true, y_pred_refined, labels=labels, average=None, zero_division=0)

        print("RNN-only per-class F1:")
        for name, score in zip(target_names, rnn_scores):
            print(f"  {name:>20}: {score:.4f}")

        print("Rule-refined per-class F1:")
        for name, score in zip(target_names, refined_scores):
            print(f"  {name:>20}: {score:.4f}")
    else:
        rnn_score = f1_score(y_true, y_pred_rnn, labels=labels, average=average_param, zero_division=0)
        refined_score = f1_score(y_true, y_pred_refined, labels=labels, average=average_param, zero_division=0)

        print(f"RNN-only F1 ({average_param}):      {rnn_score:.4f}")
        print(f"Rule-refined F1 ({average_param}):  {refined_score:.4f}")
        print(f"Delta:                         {refined_score - rnn_score:+.4f}")

    print()


def _print_classification_reports(
    labels: list[int],
    target_names: list[str],
    y_true: np.ndarray,
    y_pred_rnn: np.ndarray,
    y_pred_refined: np.ndarray,
) -> None:
    print("=== Classification Reports ===")
    print("RNN-only:")
    print(classification_report(y_true, y_pred_rnn, labels=labels, target_names=target_names, zero_division=0))
    print("Rule-refined:")
    print(classification_report(y_true, y_pred_refined, labels=labels, target_names=target_names, zero_division=0))


def _print_confusion_matrix(name: str, matrix: np.ndarray, target_names: list[str]) -> None:
    print(f"{name} Confusion Matrix:")
    width = max(len(label) for label in target_names)
    header = " " * (width + 2) + " ".join(f"{label:>8}" for label in target_names)
    print(header)
    for idx, row in enumerate(matrix):
        counts = " ".join(f"{value:>8}" for value in row)
        print(f"{target_names[idx]:>{width}} | {counts}")
    print()


def evaluate(
    model_path: Path,
    dataset_path: Path,
    batch_size: int,
    device: torch.device,
    average: str,
) -> None:
    dataset = PostureDataset(str(dataset_path))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = RNNPostureModel().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    label_encoder = dataset.get_label_encoder()

    targets, rnn_preds, refined_preds = _collect_predictions(dataloader, model, device, label_encoder)

    if not targets:
        print("Dataset is empty; nothing to evaluate.")
        return

    labels = list(range(len(label_encoder.classes_)))
    target_names = list(label_encoder.classes_)

    y_true = np.array(targets)
    y_pred_rnn = np.array(rnn_preds)
    y_pred_refined = np.array(refined_preds)

    print("=== Evaluation Summary ===")
    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")
    print(f"Samples evaluated: {len(targets)}")
    print()

    _print_f1_scores(average, labels, target_names, y_true, y_pred_rnn, y_pred_refined)
    _print_classification_reports(labels, target_names, y_true, y_pred_rnn, y_pred_refined)

    cm_rnn = confusion_matrix(y_true, y_pred_rnn, labels=labels)
    cm_refined = confusion_matrix(y_true, y_pred_refined, labels=labels)

    print("=== Confusion Matrices ===")
    _print_confusion_matrix("RNN-only", cm_rnn, target_names)
    _print_confusion_matrix("Rule-refined", cm_refined, target_names)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    average = args.average
    evaluate(args.model_path, args.dataset_path, args.batch_size, device, average)


if __name__ == "__main__":
    main()
