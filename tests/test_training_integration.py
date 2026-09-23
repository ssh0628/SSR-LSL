from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from setting.config import CONFIG, StructuralLabelsConfig, TrainingConfig
from setting.model import SSRNetworks
from ssr.trainer import train_epoch


class _ViewDataset(Dataset):
    def __init__(self, images: torch.Tensor, views: int) -> None:
        self.images = images
        self.views = views

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        return [image.clone() for _ in range(self.views)], index


class TrainingIntegrationTest(unittest.TestCase):
    def _networks(self) -> SSRNetworks:
        return SSRNetworks(
            encoder=nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 8)),
            classifier=nn.Linear(8, 10),
            projector=nn.Linear(8, 4),
            predictor=nn.Linear(4, 4),
        )

    def test_ssr_and_lsl_training_paths_update_parameters(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        images = torch.randn(8, 3, 4, 4)
        labels = torch.arange(8) % 10

        for lsl_enabled in (False, True):
            with self.subTest(lsl_enabled=lsl_enabled):
                networks = self._networks()
                optimizer = SGD(
                    [
                        parameter
                        for module in networks.all_modules()
                        for parameter in module.parameters()
                    ],
                    lr=0.01,
                )
                config = replace(
                    CONFIG,
                    data=replace(CONFIG.data, class_names=tuple(str(index) for index in range(10))),
                    structural_labels=StructuralLabelsConfig(enabled=lsl_enabled),
                    training=TrainingConfig(
                        epochs=1,
                        batch_size=4,
                        num_workers=0,
                        persistent_workers=False,
                    ),
                )
                selected = DataLoader(_ViewDataset(images, 2), batch_size=4)
                all_samples = DataLoader(
                    _ViewDataset(images, 3 if lsl_enabled else 2),
                    batch_size=4,
                )
                structural_targets = (
                    torch.nn.functional.one_hot(labels, 10).float()
                    if lsl_enabled
                    else None
                )
                before = networks.classifier.weight.detach().clone()

                losses = train_epoch(
                    selected,
                    all_samples,
                    labels,
                    structural_targets,
                    networks,
                    optimizer,
                    config,
                    device,
                    epoch=0,
                )

                self.assertTrue(torch.isfinite(torch.tensor(losses.supervised_loss)))
                self.assertTrue(
                    torch.isfinite(torch.tensor(losses.feature_consistency_loss))
                )
                self.assertEqual(losses.structural_loss is not None, lsl_enabled)
                self.assertFalse(torch.equal(before, networks.classifier.weight))


if __name__ == "__main__":
    unittest.main()
