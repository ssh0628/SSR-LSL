"""Compare memory-bounded execution against the original SSR/LSL operations."""

from __future__ import annotations

import copy
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset

from lsl import extract_structural_labels, structural_mixup_loss
from setting.model import SSRNetworks
from ssr.knn import balanced_knn_scores, hard_knn_scores, similarity_chunk_size
from ssr.losses import (
    mixup_hard_label_views,
    negative_cosine_similarity,
    soft_cross_entropy,
)
from ssr.trainer import train_epoch


class _Views(Dataset):
    def __init__(self, images: torch.Tensor, count: int) -> None:
        self.images = images
        self.count = count

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return [self.images[index] + 0.1 * view for view in range(self.count)], index


def _networks(dtype: torch.dtype) -> SSRNetworks:
    def head(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, output_size),
        ).to(dtype=dtype)

    return SSRNetworks(head(6, 8), head(8, 3), head(8, 4), head(4, 4))


def _reference_epoch(selected, all_samples, labels, targets, networks, optimizer, config):
    """Original aggregated backward; keep forward/RNG order unchanged."""
    networks.train()
    selected_iterator = iter(selected)
    averages = [[], [], []]
    for views, indices in all_samples:
        try:
            selected_views, selected_indices = next(selected_iterator)
        except StopIteration:
            selected_iterator = iter(selected)
            selected_views, selected_indices = next(selected_iterator)
        optimizer.zero_grad(set_to_none=True)
        inputs, mixed_targets, _ = mixup_hard_label_views(
            selected_views[0], selected_views[1], labels[selected_indices],
            num_classes=3, alpha=config.ssr.mixup_alpha,
        )
        ce = soft_cross_entropy(networks.classifier(networks.encoder(inputs)), mixed_targets)
        weak_projection = networks.projector(networks.encoder(views[0]))
        strong_projection = networks.projector(networks.encoder(views[1]))
        networks.predictor(weak_projection)
        prediction = networks.predictor(strong_projection)
        fc = negative_cosine_similarity(prediction, weak_projection)
        loss = ce + config.ssr.feature_consistency_weight * fc
        if targets is not None:
            st = structural_mixup_loss(
                networks.encoder, networks.classifier, views[1], views[2], targets[indices],
                mixup_alpha=config.ssr.mixup_alpha,
            )
            loss = loss + config.structural_labels.loss_weight * st
            averages[2].append(st.item())
        loss.backward()
        optimizer.step()
        averages[0].append(ce.item())
        averages[1].append(fc.item())
    return [float(np.mean(values)) if values else None for values in averages]


class _SavedTensorMemory:
    """Count live autograd-saved bytes, without changing their values."""
    class Saved:
        def __init__(self, tracker, tensor):
            self.tracker = tracker
            self.tensor = tensor.detach()
            self.size = tensor.numel() * tensor.element_size()
            tracker.live += self.size
            tracker.peak = max(tracker.peak, tracker.live)

        def __del__(self):
            self.tracker.live -= self.size

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    def pack(self, tensor):
        return self.Saved(self, tensor)

    @staticmethod
    def unpack(saved):
        return saved.tensor


class MemoryTrainingTest(unittest.TestCase):
    def test_separate_backwards_preserve_updates_bn_and_rng(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for lsl_enabled in (False, True):
                for optimizer_type in (SGD, AdamW):
                    with self.subTest(dtype=dtype, lsl=lsl_enabled, optimizer=optimizer_type):
                        torch.manual_seed(14)
                        images = torch.randn(12, 6, dtype=dtype)
                        labels = torch.arange(12) % 3
                        targets = F.one_hot(labels, 3).to(dtype) * 0.8 + 0.2 / 3
                        if not lsl_enabled:
                            targets = None
                        original = _networks(dtype)
                        optimized = copy.deepcopy(original)
                        config = SimpleNamespace(
                            data=SimpleNamespace(num_classes=3),
                            ssr=SimpleNamespace(mixup_alpha=4.0, feature_consistency_weight=0.7),
                            structural_labels=SimpleNamespace(loss_weight=1.3),
                        )
                        selected = DataLoader(_Views(images[:8], 2), batch_size=4)
                        all_samples = DataLoader(_Views(images, 3 if lsl_enabled else 2), batch_size=4)
                        optimizers = [
                            optimizer_type(
                                [p for module in net.all_modules() for p in module.parameters()],
                                lr=0.003, weight_decay=0.01,
                            )
                            for net in (original, optimized)
                        ]
                        reference_memory, optimized_memory = _SavedTensorMemory(), _SavedTensorMemory()
                        torch.manual_seed(31)
                        np.random.seed(31)
                        with torch.autograd.graph.saved_tensors_hooks(reference_memory.pack, reference_memory.unpack):
                            expected = _reference_epoch(selected, all_samples, labels, targets, original, optimizers[0], config)
                        expected_torch_rng = torch.get_rng_state().clone()
                        expected_numpy_rng = np.random.get_state()
                        torch.manual_seed(31)
                        np.random.seed(31)
                        with torch.autograd.graph.saved_tensors_hooks(optimized_memory.pack, optimized_memory.unpack):
                            actual = train_epoch(
                                selected, all_samples, labels, targets, optimized, optimizers[1],
                                config, torch.device("cpu"), epoch=0,
                            )
                        self.assertTrue(torch.equal(expected_torch_rng, torch.get_rng_state()))
                        numpy_rng = np.random.get_state()
                        self.assertEqual(expected_numpy_rng[0], numpy_rng[0])
                        np.testing.assert_array_equal(expected_numpy_rng[1], numpy_rng[1])
                        self.assertEqual(expected_numpy_rng[2:], numpy_rng[2:])
                        # Separate backward changes floating-point summation order.
                        tolerance = 2e-6 if dtype == torch.float32 else 1e-9
                        for reference_module, actual_module in zip(original.all_modules(), optimized.all_modules()):
                            for key, value in reference_module.state_dict().items():
                                other = actual_module.state_dict()[key]
                                if value.is_floating_point():
                                    torch.testing.assert_close(value, other, atol=tolerance, rtol=tolerance)
                                else:
                                    self.assertTrue(torch.equal(value, other), key)
                            for parameter, other in zip(reference_module.parameters(), actual_module.parameters()):
                                if parameter.grad is None:
                                    self.assertIsNone(other.grad)
                                else:
                                    torch.testing.assert_close(parameter.grad, other.grad, atol=tolerance, rtol=tolerance)
                        for value, reference in zip((actual.supervised_loss, actual.feature_consistency_loss, actual.structural_loss), expected):
                            if reference is None:
                                self.assertIsNone(value)
                            else:
                                self.assertAlmostEqual(value, reference, delta=tolerance)
                        self.assertLess(optimized_memory.peak, reference_memory.peak)

    def test_bounded_knn_matches_full_matrix_without_retaining_chunks(self) -> None:
        torch.manual_seed(12)
        features = torch.randn(23, 7, dtype=torch.float64)
        labels = torch.arange(23) % 3
        normalized = F.normalize(features, dim=1)
        similarity = normalized @ normalized.T
        neighbors = 5
        indices = similarity.topk(neighbors, dim=1).indices
        scores = F.one_hot(labels[indices], 3).to(features.dtype).mean(dim=1)
        prior = torch.bincount(labels, minlength=3).to(features.dtype) + 1e-10
        expected_scores = scores / (prior / prior.sum())
        expected_scores /= expected_scores.sum(dim=1, keepdim=True)
        similarity.fill_diagonal_(torch.inf)
        structural_indices = similarity.topk(neighbors, dim=1).indices
        expected_targets = torch.zeros(23, 3, dtype=features.dtype)
        for source, destinations in enumerate(structural_indices):
            for destination in destinations:
                expected_targets[destination, labels[source]] += 1
        expected_targets /= expected_targets.sum(dim=1, keepdim=True)

        matrix_bytes = 23 * features.element_size() * 4
        references = []
        original_mm = torch.mm

        def checked_mm(first, second):
            self.assertTrue(all(reference() is None for reference in references))
            result = original_mm(first, second)
            self.assertLessEqual(result.numel() * result.element_size(), matrix_bytes)
            references.append(weakref.ref(result))
            return result

        with patch("ssr.knn.MAX_SIMILARITY_BYTES", matrix_bytes), patch("torch.mm", side_effect=checked_mm):
            actual_votes = hard_knn_scores(normalized, normalized.T, labels, 3, neighbors)
            actual_scores = balanced_knn_scores(features, features, labels, num_classes=3, neighbors=neighbors, chunks=1)
            actual_targets = extract_structural_labels(features, labels, num_classes=3, neighbors=neighbors, chunks=1)
            # Explicitly requesting more chunks must still be respected.
            self.assertEqual(similarity_chunk_size(23, 23, 8, chunks=23), 1)
        self.assertTrue(torch.equal(actual_votes, scores))
        self.assertTrue(torch.equal(actual_scores, expected_scores))
        self.assertTrue(torch.equal(actual_targets, expected_targets))


if __name__ == "__main__":
    unittest.main()
