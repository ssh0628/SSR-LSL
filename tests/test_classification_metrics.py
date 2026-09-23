"""Known-count and streamed probability checks for evaluation metrics."""

from __future__ import annotations

import json
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from setting.model import SSRNetworks
from ssr.metrics import evaluate_classification, metrics_from_confusion_matrix


def _networks(encoder: nn.Module | None = None, classifier: nn.Module | None = None) -> SSRNetworks:
    return SSRNetworks(
        nn.Identity() if encoder is None else encoder,
        nn.Identity() if classifier is None else classifier,
        nn.Identity(), nn.Identity(),
    )


class ClassificationMetricsTest(unittest.TestCase):
    def test_unequal_class_counts_match_hand_calculations(self) -> None:
        scores = metrics_from_confusion_matrix([[4, 1, 0], [1, 1, 0], [0, 1, 2]], class_names=("A", "B", "C"))
        self.assertEqual(scores["sample_count"], 10)
        self.assertAlmostEqual(scores["accuracy"], 0.7)
        self.assertAlmostEqual(scores["balanced_accuracy"], (4 / 5 + 1 / 2 + 2 / 3) / 3)
        self.assertAlmostEqual(scores["macro_f1"], (0.8 + 0.4 + 0.8) / 3)
        self.assertAlmostEqual(scores["weighted_f1"], 0.72)
        self.assertAlmostEqual(scores["macro_precision"], (0.8 + 1 / 3 + 1) / 3)
        self.assertAlmostEqual(scores["weighted_precision"], 0.4 + 0.2 / 3 + 0.3)
        self.assertAlmostEqual(scores["weighted_recall"], 0.7)
        self.assertAlmostEqual(scores["micro_f1"], 0.7)
        self.assertAlmostEqual(scores["cohen_kappa"], (0.7 - 0.37) / (1 - 0.37))
        self.assertAlmostEqual(scores["multiclass_mcc"], 33 / math.sqrt(62 * 62))
        self.assertEqual(scores["per_class"][1], {
            "class_id": 1, "class_name": "B", "support": 2, "predicted_count": 3,
            "precision": 1 / 3, "recall": 0.5, "f1": 0.4,
        })
        json.dumps(scores, allow_nan=False)

    def test_absent_classes_have_explicit_macro_and_balanced_conventions(self) -> None:
        scores = metrics_from_confusion_matrix([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(scores["balanced_accuracy"], 1)
        self.assertEqual(scores["macro_recall"], 1 / 3)
        self.assertEqual(scores["macro_f1"], 1 / 3)
        self.assertEqual(scores["weighted_f1"], 1)
        self.assertEqual(scores["multiclass_mcc"], 0)
        self.assertEqual(scores["cohen_kappa"], 0)
        self.assertEqual(scores["per_class"][1]["f1"], 0)

    def test_all_wrong_single_class_predictions_remain_finite(self) -> None:
        scores = metrics_from_confusion_matrix([[0, 4], [0, 0]])
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "multiclass_mcc", "cohen_kappa"):
            self.assertEqual(scores[name], 0)
        inverse = metrics_from_confusion_matrix([[0, 2], [2, 0]])
        self.assertEqual(inverse["multiclass_mcc"], -1)
        self.assertEqual(inverse["cohen_kappa"], -1)
        json.dumps(scores, allow_nan=False)

    def test_confusion_matrix_rejects_bad_counts_and_class_names(self) -> None:
        for matrix in ([], [[1, 2]], [[0, 0], [0, 0]], [[1, -1], [0, 0]], [[0.5]], [[float("nan")]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                metrics_from_confusion_matrix(matrix)
        with self.assertRaisesRegex(ValueError, "class_names"):
            metrics_from_confusion_matrix([[1, 0], [0, 1]], class_names=("A",))

    def test_streaming_metrics_match_direct_probabilities_and_ce(self) -> None:
        probabilities = torch.tensor([[0.8, 0.12, 0.08], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3], [0.15, 0.05, 0.8]])
        logits = probabilities.log()
        labels = torch.tensor([0, 1, 2, 0])
        networks = _networks()
        forwards = []
        handle = networks.encoder.register_forward_hook(lambda _, args, output: forwards.append(output.shape[0]))
        try:
            result = evaluate_classification(
                DataLoader(TensorDataset(logits, labels), batch_size=3), networks,
                torch.device("cpu"), class_names=("A1", "A2", "A3"),
            )
        finally:
            handle.remove()
        self.assertEqual(forwards, [3, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        self.assertAlmostEqual(result["loss"], F.cross_entropy(logits, labels).item(), places=6)
        self.assertAlmostEqual(result["confidence_mean"], 0.725, places=6)
        self.assertAlmostEqual(result["entropy_mean"], -(probabilities * logits).sum(1).mean().item(), places=6)
        self.assertAlmostEqual(result["brier_score"], (probabilities - F.one_hot(labels, 3)).square().sum(1).mean().item(), places=6)
        self.assertAlmostEqual(result["expected_calibration_error"], (0.6 + 0.3 + 0.6) / 4, places=6)
        self.assertEqual(result["top2_accuracy"], 1)
        self.assertEqual(result["calibration_bins"], 15)
        self.assertFalse(networks.encoder.training)
        json.dumps(result, allow_nan=False)

    def test_batch_partition_does_not_change_scores(self) -> None:
        torch.manual_seed(0)
        dataset = TensorDataset(torch.randn(13, 4), torch.arange(13) % 4)
        scores = [evaluate_classification(DataLoader(dataset, batch_size=batch), _networks(), torch.device("cpu")) for batch in (1, 4, 13)]
        for result in scores[1:]:
            self.assertEqual(result["confusion_matrix"], scores[0]["confusion_matrix"])
            for name in ("loss", "confidence_mean", "entropy_mean", "brier_score", "expected_calibration_error", "top2_accuracy", "macro_f1"):
                self.assertAlmostEqual(result[name], scores[0][name], places=6)

    def test_outer_autocast_and_reduced_inputs_still_evaluate_fp32(self) -> None:
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            encoder = nn.Linear(3, 4)
            classifier = nn.Linear(4, 3)
            recorded = []
            handle = classifier.register_forward_hook(lambda _, args, output: recorded.append(output.dtype))
            try:
                inputs = torch.randn(5, 3).to(dtype)
                labels = torch.arange(5) % 3
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    result = evaluate_classification(
                        DataLoader(TensorDataset(inputs, labels), batch_size=2),
                        _networks(encoder, classifier), torch.device("cpu"), channels_last=True,
                    )
                expected = F.cross_entropy(classifier(encoder(inputs.float())), labels).item()
            finally:
                handle.remove()
            self.assertEqual(set(recorded), {torch.float32})
            self.assertAlmostEqual(result["loss"], expected, places=6)

    def test_nonfinite_features_and_logits_are_rejected(self) -> None:
        class NonFinite(nn.Module):
            def forward(self, inputs):
                return inputs * float("nan")

        loader = DataLoader(TensorDataset(torch.ones(2, 3), torch.tensor([0, 1])), batch_size=2)
        for networks in (_networks(NonFinite()), _networks(classifier=NonFinite())):
            with self.assertRaisesRegex(FloatingPointError, "non-finite features or logits"):
                evaluate_classification(loader, networks, torch.device("cpu"))

    def test_empty_loader_invalid_labels_and_output_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "loader is empty"):
            evaluate_classification(DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0, dtype=torch.long))), _networks(), torch.device("cpu"))
        for labels in (torch.tensor([-1, 1]), torch.tensor([0, 3]), torch.tensor([0., 1.]), torch.tensor([[0], [1]])):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                evaluate_classification(DataLoader(TensorDataset(torch.ones(2, 3), labels), batch_size=2), _networks(), torch.device("cpu"))
        loader = DataLoader(TensorDataset(torch.ones(2, 3), torch.tensor([0, 1])), batch_size=2)
        with self.assertRaisesRegex(ValueError, "class_names"):
            evaluate_classification(loader, _networks(), torch.device("cpu"), class_names=("A",))
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_classification(loader, _networks(classifier=nn.Flatten(0)), torch.device("cpu"))

    def test_single_output_has_no_top2_and_exact_confidence_one_is_valid(self) -> None:
        dataset = TensorDataset(torch.ones(3, 1), torch.zeros(3, dtype=torch.long))
        result = evaluate_classification(DataLoader(dataset, batch_size=2), _networks(), torch.device("cpu"))
        self.assertIsNone(result["top2_accuracy"])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["confidence_mean"], 1)
        self.assertEqual(result["expected_calibration_error"], 0)
        self.assertEqual(result["entropy_mean"], 0)
        self.assertEqual(result["brier_score"], 0)

    def test_class_count_must_stay_fixed_across_batches(self) -> None:
        class ChangingClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return inputs[:, :2] if self.calls == 1 else inputs

        loader = DataLoader(TensorDataset(torch.ones(2, 3), torch.tensor([0, 1])), batch_size=1)
        with self.assertRaisesRegex(ValueError, "class count changed"):
            evaluate_classification(loader, _networks(classifier=ChangingClassifier()), torch.device("cpu"))

    def test_double_precision_diagnostic_inputs_are_not_downgraded(self) -> None:
        inputs = torch.tensor([[0.2345678901, 0.9], [0.7654321098, 0.12]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = evaluate_classification(
                DataLoader(TensorDataset(inputs, labels), batch_size=2), _networks(), torch.device("cpu"),
            )
        self.assertAlmostEqual(result["loss"], F.cross_entropy(inputs, labels).item(), places=14)


if __name__ == "__main__":
    unittest.main()
