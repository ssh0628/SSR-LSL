from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from setting.model import SSRNetworks
from ssr.evaluation import selection_metrics
from ssr.metrics import metrics_from_confusion_matrix
from ssr.selection import SelectionResult
from ssr.trainer import train_epoch


class TrainingCleanupTest(unittest.TestCase):
    def test_selection_logging_preserves_counts_precision_and_rng(self):
        generator = torch.Generator().manual_seed(7)
        samples, classes = 37, 7
        observed = torch.randint(classes, (samples,), generator=generator)
        for dtype in (torch.float32, torch.float64):
            for has_changes in (False, True):
                for selected_count in (0, 19, samples):
                    with self.subTest(dtype=dtype, changes=has_changes, selected=selected_count):
                        modified = observed.clone()
                        if has_changes:
                            modified[::3] = (modified[::3] + 1) % classes
                        selected = torch.arange(selected_count)
                        rejected = torch.arange(selected_count, samples)
                        changed = torch.where(modified.ne(observed))[0]
                        confidences = torch.linspace(0.5, 1.0, samples, dtype=dtype)
                        selection = SelectionResult(
                            selected, rejected, modified, torch.arange(12), changed,
                            confidences, torch.ones(samples),
                        )
                        before = torch.get_rng_state()
                        result = selection_metrics(selection, observed, num_classes=classes)
                        self.assertTrue(torch.equal(before, torch.get_rng_state()))
                        self.assertEqual(result, {
                            "selected": selected_count,
                            "rejected": samples - selected_count,
                            "selected_rate": selected_count / samples,
                            "observed_class_counts": torch.bincount(observed, minlength=classes).tolist(),
                            "modified_class_counts": torch.bincount(modified, minlength=classes).tolist(),
                            "selected_class_counts": torch.bincount(modified[selected], minlength=classes).tolist(),
                            "rejected_class_counts": torch.bincount(modified[rejected], minlength=classes).tolist(),
                            "relabel_candidates": 12,
                            "relabelled": 12,
                            "label_changes": changed.numel(),
                            "label_change_rate": changed.numel() / samples,
                            "label_change_transitions": [
                                {"from": source, "to": target, "count": count}
                                for source in range(classes)
                                for target in range(classes)
                                if source != target and (
                                    count := int(((observed == source) & (modified == target)).sum())
                                )
                            ],
                            "confidence_mean": confidences.mean().item(),
                            "confidence_min": confidences.min().item(),
                            "confidence_max": confidences.max().item(),
                        })

    def test_prediction_counts_reused_from_observed_metrics_match_direct_histogram(self):
        observed = torch.tensor([0, 0, 1, 1, 2, 2, 2])
        predictions = torch.tensor([2, 2, 1, 2, 0, 2, 1])
        classes = 4  # Includes an absent class.
        confusion = torch.bincount(
            observed * classes + predictions, minlength=classes**2,
        ).reshape(classes, classes)
        metrics = metrics_from_confusion_matrix(confusion)
        reused = [item["predicted_count"] for item in metrics["per_class"]]
        reference = torch.bincount(predictions, minlength=classes)
        self.assertEqual(reused, reference.tolist())
        self.assertEqual(max(reused) / len(observed), reference.max().item() / len(observed))

    def test_invalid_view_count_fails_before_forward_bn_update_and_mixup_rng(self):
        config = SimpleNamespace(
            data=SimpleNamespace(num_classes=2),
            ssr=SimpleNamespace(mixup_alpha=4.0, feature_consistency_weight=1.0),
            structural_labels=SimpleNamespace(loss_weight=1.0),
        )
        images = torch.ones(4, 3)
        labels = torch.arange(4) % 2
        indices = torch.arange(4)
        for structural, views, algorithm in ((None, 3, "SSR"), (torch.eye(2)[labels], 2, "LSL")):
            with self.subTest(algorithm=algorithm):
                networks = SSRNetworks(
                    nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)),
                    nn.Linear(4, 2), nn.Linear(4, 4), nn.Linear(4, 4),
                )
                optimizer = torch.optim.SGD(
                    [p for module in networks.all_modules() for p in module.parameters()], lr=0.01,
                )
                before = {
                    (index, key): value.clone()
                    for index, module in enumerate(networks.all_modules())
                    for key, value in module.state_dict().items()
                }
                rng = torch.get_rng_state()
                with patch.object(networks.encoder, "forward", wraps=networks.encoder.forward) as forward:
                    with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                        with self.assertRaisesRegex(RuntimeError, f"{algorithm} requires"):
                            train_epoch(
                                [([images, images], indices)],
                                [([images] * views, indices)],
                                labels, structural, networks, optimizer,
                                config, torch.device("cpu"), epoch=0,
                            )
                        forward.assert_not_called()
                        step.assert_not_called()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for index, module in enumerate(networks.all_modules()):
                    for key, value in module.state_dict().items():
                        self.assertTrue(torch.equal(before[index, key], value))


if __name__ == "__main__":
    unittest.main()
