from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from setting.model import calibrate_batch_norm


class BatchNormCalibrationTest(unittest.TestCase):
    def test_calibration_updates_statistics_without_parameters_or_mode_changes(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        encoder = nn.Sequential(nn.BatchNorm2d(3, momentum=0.2), nn.Flatten()).eval()
        images = torch.randn(16, 3, 8, 8) + torch.tensor([5.0, 8.0, -4.0])[None, :, None, None]
        loader = DataLoader(
            TensorDataset(images, torch.arange(len(images))),
            batch_size=4,
            shuffle=False,
        )
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in encoder.named_parameters()
        }
        batches = calibrate_batch_norm(loader, encoder, device)

        self.assertEqual(batches, 4)
        torch.testing.assert_close(encoder[0].running_mean, images.mean(dim=(0, 2, 3)))
        expected_variance = torch.stack([
            batch.var(dim=(0, 2, 3)) for batch in images.split(4)
        ]).mean(dim=0)
        torch.testing.assert_close(encoder[0].running_var, expected_variance)
        self.assertEqual(encoder[0].momentum, 0.2)
        self.assertTrue(all(not module.training for module in encoder.modules()))
        for name, parameter in encoder.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters_before[name]), name)

        tracked_batches = [
            int(module.num_batches_tracked)
            for module in encoder.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        self.assertTrue(tracked_batches)
        self.assertTrue(all(count == batches for count in tracked_batches))


if __name__ == "__main__":
    unittest.main()
