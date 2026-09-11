"""One-pass classification metrics without storing every sample's probabilities.

Accuracy, F1, precision, recall, confidence and calibration error are fractions
in [0, 1], not percentages. Confusion matrices use truth rows / prediction
columns. Balanced accuracy excludes classes with no true support; macro scores
include every configured class with zero-division results set to zero. MCC and
kappa use zero when their denominator is zero. Entropy / CE use natural logs;
the multiclass Brier score is the summed squared probability error in [0, 2].
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from setting.model import SSRNetworks
from setting.precision import full_precision
from ssr.evaluation import _evaluation_images, _validate_model_outputs


_CALIBRATION_BINS = 15
_LABEL_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def metrics_from_confusion_matrix(
    confusion_matrix: Tensor | list[list[int]],
    *,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute JSON-compatible scores; undefined class scores are zero."""
    matrix = torch.as_tensor(confusion_matrix).detach().to("cpu", torch.float64)
    if matrix.ndim != 2 or matrix.size(0) == 0 or matrix.size(0) != matrix.size(1):
        raise ValueError("Confusion matrix must be a nonempty square matrix.")
    if not torch.isfinite(matrix).all() or (matrix < 0).any() or (matrix != matrix.round()).any():
        raise ValueError("Confusion matrix must contain finite nonnegative integer counts.")
    count = int(matrix.sum().item())
    if count == 0:
        raise ValueError("Confusion matrix must contain at least one sample.")
    classes = matrix.size(0)
    names = tuple(str(index) for index in range(classes)) if class_names is None else tuple(class_names)
    if len(names) != classes or any(not isinstance(name, str) for name in names):
        raise ValueError("class_names must contain one string per output class.")

    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    correct = matrix.diagonal()
    precision = correct / predicted.clamp_min(1)
    recall = correct / support.clamp_min(1)
    f1 = 2 * correct / (support + predicted).clamp_min(1)
    weights = support / count
    accuracy = float(correct.sum().item() / count)
    chance = float(torch.dot(support, predicted).item() / count**2)
    mcc_denominator = float(
        ((count**2 - predicted.square().sum()) * (count**2 - support.square().sum()))
        .clamp_min(0).sqrt().item()
    )
    mcc_numerator = float((correct.sum() * count - torch.dot(support, predicted)).item())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall[support > 0].mean().item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float(torch.dot(weights, f1).item()),
        "micro_f1": accuracy,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "weighted_precision": float(torch.dot(weights, precision).item()),
        "weighted_recall": float(torch.dot(weights, recall).item()),
        "multiclass_mcc": mcc_numerator / mcc_denominator if mcc_denominator else 0.0,
        "cohen_kappa": (accuracy - chance) / (1 - chance) if chance < 1 else 0.0,
        "sample_count": count,
        "confusion_matrix": matrix.to(torch.int64).tolist(),
        "per_class": [
            {
                "class_id": index,
                "class_name": names[index],
                "support": int(support[index]),
                "predicted_count": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(classes)
        ],
    }


def _labels_for_batch(labels: Tensor, batch_size: int, classes: int, device: torch.device) -> Tensor:
    # Normal DataLoader labels are still on CPU: reject malformed labels before
    # invoking CUDA gather / bincount kernels, without a device synchronization.
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or labels.numel() != batch_size or labels.dtype not in _LABEL_DTYPES:
        raise ValueError("Evaluation labels must be a one-dimensional integer tensor matching the batch.")
    if ((labels < 0) | (labels >= classes)).any().item():
        raise ValueError(f"Evaluation labels must lie in [0, {classes - 1}].")
    return labels.to(device, dtype=torch.long, non_blocking=True)


@torch.no_grad()
def evaluate_classification(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
    *,
    description: str = "Validation",
    channels_last: bool = False,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate in full precision; one forward pass and O(classes²) memory.

    Networks remain in evaluation mode, matching the previous accuracy helper.
    Top-2 accuracy is None for a single-output classifier. ECE uses 15 equal
    width confidence bins: [i/15, (i+1)/15), with confidence 1 in the last bin.
    FP16/BF16 inputs and logits are promoted to FP32; diagnostic FP64 is kept.
    """
    device = torch.device(device)
    networks.eval()
    # MPS does not support float64; model inference itself remains FP32.
    accumulation_dtype = torch.float32 if device.type == "mps" else torch.float64
    totals = torch.zeros(5, dtype=accumulation_dtype, device=device)
    calibration = torch.zeros((2, _CALIBRATION_BINS), dtype=accumulation_dtype, device=device)
    matrix: Tensor | None = None
    sample_count = 0

    with full_precision(device):
        for images, labels in tqdm(loader, desc=description, leave=False, disable=None):
            if images.size(0) == 0:
                raise ValueError("Evaluation batches must not be empty.")
            images = _evaluation_images(images, device, channels_last)
            features = networks.encoder(images)
            logits = networks.classifier(features)
            if logits.ndim != 2 or logits.size(0) != images.size(0) or logits.size(1) == 0:
                raise ValueError("Classifier logits must have shape (batch_size, num_classes).")
            if not logits.is_floating_point():
                raise ValueError("Classifier logits must be floating point.")
            classes = logits.size(1)
            if matrix is None:
                if class_names is not None and (
                    len(class_names) != classes or any(not isinstance(name, str) for name in class_names)
                ):
                    raise ValueError("class_names must contain one string per output class.")
                matrix = torch.zeros((classes, classes), dtype=torch.int64, device=device)
            elif matrix.size(0) != classes:
                raise ValueError("Classifier output class count changed between batches.")
            labels = _labels_for_batch(labels, images.size(0), classes, device)
            # One explicit output-validity check per batch; no per-metric .item().
            _validate_model_outputs(features, logits)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            log_probabilities = logits.log_softmax(dim=1)
            probabilities = log_probabilities.exp()
            confidence, predictions = probabilities.max(dim=1)
            correct = predictions.eq(labels)
            true_log_probability = log_probabilities.gather(1, labels[:, None]).squeeze(1)
            true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
            top2_correct = (
                logits.topk(2, dim=1).indices.eq(labels[:, None]).any(dim=1).sum()
                if classes >= 2 else logits.new_zeros(())
            )
            totals += torch.stack((
                -true_log_probability.sum(),
                confidence.sum(),
                -(probabilities * log_probabilities).sum(),
                (probabilities.square().sum(dim=1) - 2 * true_probability + 1).clamp_min(0).sum(),
                top2_correct,
            )).to(accumulation_dtype)
            matrix += torch.bincount(
                labels * classes + predictions, minlength=classes * classes
            ).reshape(classes, classes)
            bins = (confidence * _CALIBRATION_BINS).long().clamp_max(_CALIBRATION_BINS - 1)
            calibration[0].scatter_add_(0, bins, confidence.to(accumulation_dtype))
            calibration[1].scatter_add_(0, bins, correct.to(accumulation_dtype))
            sample_count += images.size(0)

    if matrix is None or sample_count == 0:
        raise RuntimeError("Classification evaluation loader is empty.")
    values = totals.cpu().tolist()
    if not torch.isfinite(totals).all().item():
        raise FloatingPointError("Classification evaluation produced non-finite probability metrics.")
    result = metrics_from_confusion_matrix(matrix, class_names=class_names)
    result.update({
        "loss": values[0] / sample_count,
        "confidence_mean": values[1] / sample_count,
        "entropy_mean": values[2] / sample_count,
        "brier_score": values[3] / sample_count,
        "top2_accuracy": values[4] / sample_count if matrix.size(0) >= 2 else None,
        "expected_calibration_error": float((calibration[0] - calibration[1]).abs().sum().item() / sample_count),
        "calibration_bins": _CALIBRATION_BINS,
    })
    return result
