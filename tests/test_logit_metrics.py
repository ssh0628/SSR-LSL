"""Multi-view aggregate logits use the existing classification metric definitions."""

from __future__ import annotations

import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from setting.model import SSRNetworks
from ssr.metrics import evaluate_classification, metrics_from_logits


class LogitMetricsTest(unittest.TestCase):
    def test_logits_match_streamed_evaluation_for_every_supported_precision(self) -> None:
        source = torch.tensor([
            [0.234567890123, 0.9, -1.0], [0.765432109876, 0.12, 1.5],
            [-0.2, 0.4, 0.3], [0.12, 0.11, 0.1], [1.2, -0.3, 0.22],
        ], dtype=torch.float64)
        labels = torch.tensor([0, 2, 1, 2, 0])
        names = ("A1", "A2", "A3")
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logits = source.to(dtype)
            with self.subTest(dtype=dtype), torch.autocast("cpu", dtype=torch.bfloat16):
                direct = metrics_from_logits(logits, labels, class_names=names)
                for batch_size in (2, 5):
                    streamed = evaluate_classification(
                        DataLoader(TensorDataset(logits, labels), batch_size=batch_size),
                        SSRNetworks(nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()),
                        torch.device("cpu"), class_names=names,
                    )
                    if batch_size == 5:
                        self.assertEqual(direct, streamed)
                    else:
                        self.assertEqual(direct.keys(), streamed.keys())
                        for name, value in direct.items():
                            if isinstance(value, float):
                                self.assertAlmostEqual(value, streamed[name], places=12 if dtype == torch.float64 else 6)
                            else:
                                self.assertEqual(value, streamed[name])
                json.dumps(direct, allow_nan=False)

    def test_float64_precision_and_input_tensors_are_preserved(self) -> None:
        logits = torch.tensor([[0.1234567890123, 0.9], [0.7654321098765, 0.12]], dtype=torch.float64, requires_grad=True)
        original = logits.detach().clone()
        labels = torch.tensor([0, 1], dtype=torch.int32)
        result = metrics_from_logits(logits, labels)
        self.assertAlmostEqual(result["loss"], F.cross_entropy(logits, labels.long()).item(), places=14)
        self.assertTrue(torch.equal(original, logits.detach()))
        self.assertIsNone(logits.grad)
        self.assertEqual(labels.dtype, torch.int32)

    def test_rejects_invalid_shapes_dtypes_and_labels(self) -> None:
        for logits in (torch.ones(2), torch.ones(2, 0), torch.ones(2, 2, 1), torch.ones(2, 3, dtype=torch.long)):
            with self.subTest(shape=logits.shape, dtype=logits.dtype), self.assertRaises(ValueError):
                metrics_from_logits(logits, torch.tensor([0, 1]))
        for labels in (
            torch.tensor([-1, 1]), torch.tensor([0, 3]), torch.tensor([0.0, 1.0]),
            torch.tensor([[0], [1]]), torch.tensor([True, False]), torch.tensor([0]),
        ):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                metrics_from_logits(torch.ones(2, 3), labels)
        for names in (("A",), ("A", "B", 3)):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "class_names"):
                metrics_from_logits(torch.ones(2, 3), torch.tensor([0, 1]), class_names=names)

    def test_rejects_empty_and_nonfinite_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            metrics_from_logits(torch.empty(0, 3), torch.empty(0, dtype=torch.long))
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "non-finite logits"):
                metrics_from_logits(torch.tensor([[0.0, value]]), torch.tensor([0]))

    def test_single_class_and_absent_classes_keep_metric_conventions(self) -> None:
        result = metrics_from_logits(torch.ones(3, 1), torch.zeros(3, dtype=torch.long))
        self.assertIsNone(result["top2_accuracy"])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["expected_calibration_error"], 0)
        self.assertEqual(result["loss"], 0)
        absent = metrics_from_logits(torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([0]))
        self.assertEqual(absent["balanced_accuracy"], 1)
        self.assertEqual(absent["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
