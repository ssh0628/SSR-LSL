from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from setting.model import SSRNetworks
from ssr.evaluation import _extract_features_and_predictions, predict_training_labels
from ssr.trainer import train_epoch
from ssr.metrics import evaluate_classification


class _NonFinite(nn.Module):
    def forward(self, inputs):
        return inputs * float("nan")


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.tensor([[3., 1., 2.], [1., 3., 2.], [1., 2., 3.], [3., 2., 1.]])
        self.dataset = TensorDataset(self.images, torch.arange(4))
        self.networks = SSRNetworks(nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity())
        self.device = torch.device("cpu")

    def test_out_of_order_batches_are_restored_to_dataset_order(self) -> None:
        loader = DataLoader(self.dataset, batch_size=2, sampler=[3, 1, 0, 2])
        features, probabilities = _extract_features_and_predictions(loader, self.networks, self.device)
        predictions = predict_training_labels(loader, self.networks, self.device)
        self.assertTrue(torch.equal(features, self.images))
        self.assertTrue(torch.equal(probabilities, self.images.softmax(dim=1)))
        self.assertTrue(torch.equal(predictions, self.images.argmax(dim=1)))

    def test_duplicate_and_missing_indices_are_rejected_in_both_paths(self) -> None:
        for extraction in (_extract_features_and_predictions, predict_training_labels):
            for sampler, error in (
                ([0, 0, 1, 2, 3], "duplicate"),
                ([0, 1, 2, 0, 3], "duplicate"),
                ([0, 1, 2], "cover every dataset sample"),
            ):
                with self.subTest(extraction=extraction.__name__, sampler=sampler):
                    loader = DataLoader(self.dataset, batch_size=2, sampler=sampler)
                    with self.assertRaisesRegex(RuntimeError, error):
                        extraction(loader, self.networks, self.device)

    def test_nonfinite_model_outputs_do_not_become_label_wave_predictions(self) -> None:
        loader = DataLoader(self.dataset, batch_size=2)
        for encoder, classifier in ((_NonFinite(), nn.Identity()), (nn.Identity(), _NonFinite())):
            networks = SSRNetworks(encoder, classifier, nn.Identity(), nn.Identity())
            for extraction in (_extract_features_and_predictions, predict_training_labels):
                with self.subTest(extraction=extraction.__name__, encoder=type(encoder).__name__):
                    with self.assertRaisesRegex(FloatingPointError, "non-finite features or logits"):
                        extraction(loader, networks, self.device)

    def test_nonfinite_loss_never_applies_optimizer_step(self) -> None:
        labels = torch.arange(4) % 3
        selected = [([self.images, self.images + 0.1], torch.arange(4))]
        all_samples = [([self.images, self.images + 0.1, self.images + 0.2], torch.arange(4))]
        config = SimpleNamespace(
            data=SimpleNamespace(num_classes=3),
            ssr=SimpleNamespace(mixup_alpha=4.0, feature_consistency_weight=1.0),
            structural_labels=SimpleNamespace(loss_weight=1.0),
        )
        for broken_branch in ("supervised", "structural"):
            with self.subTest(branch=broken_branch):
                torch.manual_seed(0)
                networks = SSRNetworks(nn.Linear(3, 4), nn.Linear(4, 3), nn.Linear(4, 3), nn.Linear(3, 3))
                optimizer = SGD([p for module in networks.all_modules() for p in module.parameters()], lr=0.01)
                before = [p.detach().clone() for module in networks.all_modules() for p in module.parameters()]
                targets = torch.full((4, 3), float("nan"))
                if broken_branch == "supervised":
                    views = [[image * float("nan") for image in selected[0][0]], selected[0][1]]
                    selected_batches = [views]
                else:
                    selected_batches = selected
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(FloatingPointError, f"Non-finite {broken_branch} loss"):
                        train_epoch(selected_batches, all_samples, labels, targets, networks, optimizer, config, self.device, epoch=0)
                    step.assert_not_called()
                for reference, parameter in zip(before, [p for module in networks.all_modules() for p in module.parameters()]):
                    self.assertTrue(torch.equal(reference, parameter))

    def test_accuracy_rejects_nonfinite_logits_and_empty_data(self) -> None:
        loader = DataLoader(self.dataset, batch_size=2)
        networks = SSRNetworks(nn.Identity(), _NonFinite(), nn.Identity(), nn.Identity())
        with self.assertRaisesRegex(FloatingPointError, "non-finite features or logits"):
            evaluate_classification(loader, networks, self.device)
        empty = TensorDataset(self.images[:0], torch.arange(0))
        with self.assertRaisesRegex(RuntimeError, "loader is empty"):
            evaluate_classification(DataLoader(empty), self.networks, self.device)


if __name__ == "__main__":
    unittest.main()
