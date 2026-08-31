from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from setting.config import CONFIG, SSRConfig, StructuralLabelsConfig
from setting.model import SSRNetworks
from ssr.evaluation import evaluate_epoch


class LabelWaveIntegrationTest(unittest.TestCase):
    def test_ssr_evaluation_exposes_raw_predictions_in_dataset_order(self) -> None:
        torch.manual_seed(0)
        images = torch.randn(10, 3, 4, 4)
        loader = DataLoader(
            TensorDataset(images, torch.arange(len(images))),
            batch_size=4,
            shuffle=False,
        )
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 8))
        classifier = nn.Linear(8, 10)
        networks = SSRNetworks(
            encoder=encoder,
            classifier=classifier,
            projector=nn.Identity(),
            predictor=nn.Identity(),
        )
        config = replace(
            CONFIG,
            ssr=SSRConfig(neighbors=3, knn_chunks=2),
            structural_labels=StructuralLabelsConfig(enabled=False),
        )
        noisy_labels = torch.arange(10, dtype=torch.long)

        supervision = evaluate_epoch(
            loader,
            networks,
            noisy_labels,
            config,
            torch.device("cpu"),
        )
        expected = classifier(encoder(images)).argmax(dim=1)

        self.assertTrue(torch.equal(supervision.predictions, expected))
        self.assertEqual(supervision.predictions.shape, noisy_labels.shape)


if __name__ == "__main__":
    unittest.main()
