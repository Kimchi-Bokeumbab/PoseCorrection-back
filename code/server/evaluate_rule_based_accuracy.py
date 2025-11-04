"""Compare RNN-only accuracy against the rule-refined pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
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
else:  # pragma: no cover - module import path when packaged
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
    return parser.parse_args()


def evaluate(model_path: Path, dataset_path: Path, batch_size: int, device: torch.device) -> None:
    dataset = PostureDataset(str(dataset_path))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = RNNPostureModel().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    label_encoder = dataset.get_label_encoder()

    total_samples = 0
    correct_rnn = 0
    correct_refined = 0

    tilt_cases = 0
    tilt_changed = 0
    tilt_fixed = 0
    tilt_regressed = 0

    tilt_labels = {"shoulder_tilt", "head_tilt"}

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_inputs)
            raw_predictions = logits.argmax(dim=1)

            correct_rnn += (raw_predictions == batch_targets).sum().item()

            raw_labels = label_encoder.inverse_transform(raw_predictions.detach().cpu().numpy())
            true_labels = label_encoder.inverse_transform(batch_targets.detach().cpu().numpy())

            refined_indices = []
            inputs_cpu = batch_inputs.detach().cpu()

            for idx, raw_label in enumerate(raw_labels):
                refined_label = raw_label
                if raw_label in tilt_labels:
                    diff_sequence: Sequence[Sequence[float]] = inputs_cpu[idx].tolist()
                    refined_label = refine_tilt_prediction(raw_label, diff_sequence)

                    tilt_cases += 1
                    if refined_label != raw_label:
                        tilt_changed += 1
                    if raw_label != true_labels[idx] and refined_label == true_labels[idx]:
                        tilt_fixed += 1
                    if raw_label == true_labels[idx] and refined_label != true_labels[idx]:
                        tilt_regressed += 1

                refined_indices.append(label_encoder.transform([refined_label])[0])

            refined_tensor = torch.tensor(refined_indices, device=device)
            correct_refined += (refined_tensor == batch_targets).sum().item()

            total_samples += batch_targets.numel()

    acc_rnn = correct_rnn / total_samples if total_samples else 0.0
    acc_refined = correct_refined / total_samples if total_samples else 0.0

    print("=== Evaluation Summary ===")
    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")
    print(f"Samples evaluated: {total_samples}")
    print()
    print(f"RNN-only accuracy:    {acc_rnn:.4%}")
    print(f"Rule-refined accuracy: {acc_refined:.4%}")
    print(f"Delta:                {acc_refined - acc_rnn:+.4%}")

    print()
    print("--- Tilt refinement diagnostics ---")
    print(f"Tilts encountered:              {tilt_cases}")
    if tilt_cases:
        print(f"Changed by refinement:          {tilt_changed}")
        print(f"Errors corrected by refinement: {tilt_fixed}")
        print(f"Correct-to-wrong regressions:   {tilt_regressed}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    evaluate(args.model_path, args.dataset_path, args.batch_size, device)


if __name__ == "__main__":
    main()
