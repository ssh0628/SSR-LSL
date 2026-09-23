"""Selection and Label Wave retain FP32 decisions inside an outer AMP context."""

from __future__ import annotations

from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from setting.config import CONFIG
from setting.model import SSRNetworks
from ssr.evaluation import (
    _extract_features_and_predictions,
    evaluate_epoch,
    predict_training_labels,
)


class SelectionPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.images = torch.randn(9, 6)
        self.labels = torch.arange(9) % 3
        self.networks = SSRNetworks(
            nn.Linear(6, 8), nn.Linear(8, 3), nn.Identity(), nn.Identity()
        )
        self.device = torch.device("cpu")
        self.config = replace(
            CONFIG,
            data=replace(CONFIG.data, class_names=("A", "B", "C")),
            ssr=replace(CONFIG.ssr, neighbors=3, knn_chunks=2),
            structural_labels=replace(
                CONFIG.structural_labels, enabled=True, neighbors=3, knn_chunks=2
            ),
        )

    def loader(self, images: torch.Tensor | None = None) -> DataLoader:
        if images is None:
            images = self.images
        return DataLoader(TensorDataset(images, torch.arange(len(images))), batch_size=3)

    def test_feature_extraction_and_label_wave_ignore_enclosing_autocast(self) -> None:
        expected_features, expected_probabilities = _extract_features_and_predictions(
            self.loader(), self.networks, self.device
        )
        expected_predictions = expected_probabilities.argmax(dim=1)
        seen_dtypes = []
        handle = self.networks.encoder.register_forward_hook(
            lambda module, inputs, output: seen_dtypes.append(output.dtype)
        )
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                features, probabilities = _extract_features_and_predictions(
                    self.loader(), self.networks, self.device
                )
                predictions = predict_training_labels(
                    self.loader(), self.networks, self.device
                )
                self.assertTrue(all(dtype == torch.float32 for dtype in seen_dtypes))
                # The evaluation guard must restore, not erase, the caller's context.
                self.assertEqual(self.networks.encoder(self.images).dtype, torch.bfloat16)
        finally:
            handle.remove()
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(probabilities.dtype, torch.float32)
        torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=0, atol=0)
        self.assertTrue(torch.equal(predictions, expected_predictions))

    def test_selection_and_structural_targets_keep_fp32_similarity(self) -> None:
        expected = evaluate_epoch(
            self.loader(), self.networks, self.labels, self.config, self.device
        )
        result_dtypes = []
        original_mm = torch.mm

        def record_mm(*args, **kwargs):
            result = original_mm(*args, **kwargs)
            result_dtypes.append(result.dtype)
            return result

        with torch.autocast("cpu", dtype=torch.bfloat16):
            with patch("torch.mm", side_effect=record_mm):
                actual = evaluate_epoch(
                    self.loader(), self.networks, self.labels, self.config, self.device
                )
        self.assertTrue(result_dtypes)
        self.assertTrue(all(dtype == torch.float32 for dtype in result_dtypes))
        for field in fields(expected.selection):
            torch.testing.assert_close(
                getattr(actual.selection, field.name),
                getattr(expected.selection, field.name),
                rtol=0,
                atol=0,
            )
        self.assertEqual(actual.structural_targets.dtype, torch.float32)
        torch.testing.assert_close(
            actual.structural_targets, expected.structural_targets, rtol=0, atol=0
        )
        self.assertTrue(torch.equal(actual.predictions, expected.predictions))

    def test_half_inputs_are_promoted_before_model_forward(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                images = self.images.to(dtype)
                expected = _extract_features_and_predictions(
                    self.loader(images.float()), self.networks, self.device
                )
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = _extract_features_and_predictions(
                        self.loader(images), self.networks, self.device
                    )
                    predictions = predict_training_labels(
                        self.loader(images), self.networks, self.device
                    )
                for result, reference in zip(actual, expected):
                    self.assertEqual(result.dtype, torch.float32)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertTrue(torch.equal(predictions, expected[1].argmax(dim=1)))

    def test_double_precision_diagnostics_are_not_downcast(self) -> None:
        for module in self.networks.all_modules():
            module.double()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            features, probabilities = _extract_features_and_predictions(
                self.loader(self.images.double()), self.networks, self.device
            )
        self.assertEqual(features.dtype, torch.float64)
        self.assertEqual(probabilities.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
