from __future__ import annotations

import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from lsl import extract_structural_labels
from setting.config import CONFIG, TrainingConfig
from setting.model import SSRNetworks
from ssr.engine import _build_optimizer, _build_scheduler
from ssr.evaluation import selection_metrics
from ssr.knn import balanced_knn_scores, hard_knn_scores
from ssr.losses import mixup_soft_target_views
from ssr.selection import SelectionResult


class SSRCoreTest(unittest.TestCase):
    def test_memory_efficient_hard_vote_matches_one_hot_reference(self) -> None:
        torch.manual_seed(0)
        queries = F.normalize(torch.randn(5, 4), dim=1)
        bank = F.normalize(torch.randn(9, 4), dim=1)
        labels = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0])
        neighbors = 4

        actual = hard_knn_scores(queries, bank.T, labels, 3, neighbors)
        indices = (queries @ bank.T).topk(neighbors, dim=1).indices
        expected = F.one_hot(labels[indices], num_classes=3).float().mean(dim=1)

        self.assertTrue(torch.equal(actual, expected))

    def test_balanced_knn_is_invariant_to_feature_scale(self) -> None:
        torch.manual_seed(1)
        features = torch.randn(8, 5)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        baseline = balanced_knn_scores(
            features,
            features,
            labels,
            num_classes=3,
            neighbors=3,
            chunks=3,
        )
        scaled = balanced_knn_scores(
            features * 7.0,
            features * 0.25,
            labels,
            num_classes=3,
            neighbors=3,
            chunks=3,
        )

        self.assertTrue(torch.allclose(baseline, scaled, atol=1e-6))
        self.assertTrue(torch.allclose(baseline.sum(dim=1), torch.ones(8)))

    def test_reverse_knn_targets_are_finite_probabilities(self) -> None:
        features = torch.eye(4)
        labels = torch.tensor([0, 1, 2, 1])
        targets = extract_structural_labels(
            features,
            labels,
            neighbors=1,
            chunks=3,
            num_classes=3,
        )

        self.assertTrue(torch.equal(targets, F.one_hot(labels, 3).float()))
        self.assertTrue(torch.isfinite(targets).all())
        self.assertTrue(torch.equal(targets.sum(dim=1), torch.ones(4)))

    def test_invalid_scheduler_ratio_and_mixup_alpha_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "eta_min_ratio"):
            TrainingConfig(scheduler_eta_min_ratio=1.1).validate()
        with self.assertRaisesRegex(ValueError, "alpha"):
            mixup_soft_target_views(
                torch.randn(2, 3),
                torch.randn(2, 3),
                torch.eye(2),
                alpha=0.0,
            )

    def test_relabel_candidates_and_actual_changes_are_logged_separately(self) -> None:
        noisy = torch.tensor([0, 1, 2, 0])
        modified = torch.tensor([0, 2, 1, 0])
        selection = SelectionResult(
            selected_indices=torch.tensor([0, 1, 2]),
            rejected_indices=torch.tensor([3]),
            modified_labels=modified,
            relabelled_indices=torch.tensor([0, 1, 2]),
            changed_indices=torch.tensor([1, 2]),
            confidences=torch.tensor([0.95, 0.99, 0.91, 0.40]),
            consistency=torch.ones(4),
        )

        metrics = selection_metrics(
            selection,
            noisy,
            num_classes=3,
        )

        self.assertEqual(metrics["relabelled"], 3)
        self.assertEqual(metrics["relabel_candidates"], 3)
        self.assertEqual(metrics["label_changes"], 2)
        self.assertEqual(metrics["label_change_rate"], 0.5)
        self.assertEqual(
            metrics["label_change_transitions"],
            [
                {"from": 1, "to": 2, "count": 1},
                {"from": 2, "to": 1, "count": 1},
            ],
        )

    def test_cosine_schedule_preserves_each_parameter_group_lr_ratio(self) -> None:
        networks = SSRNetworks(
            encoder=nn.Linear(2, 2),
            classifier=nn.Linear(2, 2),
            projector=nn.Linear(2, 2),
            predictor=nn.Linear(2, 2),
        )
        config = replace(
            CONFIG,
            training=TrainingConfig(
                epochs=4,
                optimizer="adamw",
                learning_rate=1e-2,
                encoder_learning_rate=1e-3,
                scheduler_eta_min_ratio=0.1,
            ),
        )
        optimizer = _build_optimizer(networks, config)
        self.assertIsInstance(optimizer, AdamW)
        scheduler = _build_scheduler(optimizer, config)

        for _ in range(config.training.epochs):
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-3)


if __name__ == "__main__":
    unittest.main()
