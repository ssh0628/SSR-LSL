"""Mixed-precision policy, FP32 loss arithmetic, and unchanged CPU training."""

from __future__ import annotations

import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from lsl.loss import structural_mixup_loss
from setting.model import SSRNetworks
from setting.precision import PrecisionPolicy, full_precision
from ssr.losses import (
    mixup_soft_target_views,
    negative_cosine_similarity,
    soft_cross_entropy,
)
from ssr.trainer import train_epoch


class _Images(Dataset):
    def __init__(self, views: int):
        self.images = torch.linspace(-1, 1, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
        self.views = views

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return [self.images[index] + view * 0.1 for view in range(self.views)], index


def _networks(device):
    encoder = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    return SSRNetworks(
        encoder.to(device), nn.Linear(4, 3).to(device),
        nn.Linear(4, 4).to(device), nn.Linear(4, 4).to(device),
    )


def _config(amp):
    return SimpleNamespace(
        training=SimpleNamespace(amp=amp, channels_last=True, log_interval=20),
        data=SimpleNamespace(num_classes=3),
        ssr=SimpleNamespace(mixup_alpha=4.0, feature_consistency_weight=0.7),
        structural_labels=SimpleNamespace(loss_weight=1.0),
    )


def _train(networks, config, device):
    optimizer = torch.optim.SGD(
        [p for module in networks.all_modules() for p in module.parameters()], lr=0.01
    )
    labels = torch.tensor([0, 1, 2, 0], device=device)
    targets = F.one_hot(labels, 3).float()
    return train_epoch(
        DataLoader(_Images(2), batch_size=2),
        DataLoader(_Images(3), batch_size=2),
        labels, targets, networks, optimizer, config, device, epoch=0,
    )


class PrecisionTest(unittest.TestCase):
    def test_cpu_and_mps_policy_disable_cuda_optimizations(self):
        for device in ("cpu", "mps"):
            policy = PrecisionPolicy.from_config(_config(True).training, device)
            self.assertFalse(policy.amp)
            self.assertFalse(policy.channels_last)
        policy = PrecisionPolicy.from_config(None, "cpu")
        source = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        self.assertEqual(policy.to_device(source).dtype, torch.float64)
        self.assertTrue(policy.to_device(source).is_contiguous())
        layer = nn.Linear(4, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16), policy.autocast():
            self.assertEqual(layer(torch.randn(2, 4)).dtype, torch.float32)

    def test_loss_arithmetic_is_fp32_for_reduced_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            logits = torch.tensor([[50, -40, 2], [0.2, 0.3, -0.6]], dtype=dtype)
            logits.requires_grad_()
            targets = torch.tensor([[0.2, 0.5, 0.3], [0.5, 0.25, 0.25]], dtype=dtype)
            prediction = torch.randn(2, 4, dtype=dtype, requires_grad=True)
            projection = torch.randn(2, 4, dtype=dtype, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                ce = soft_cross_entropy(logits, targets)
                fc = negative_cosine_similarity(prediction, projection)
            self.assertEqual(ce.dtype, torch.float32)
            self.assertEqual(fc.dtype, torch.float32)
            torch.testing.assert_close(
                ce, -(F.log_softmax(logits.float(), dim=1) * targets.float()).sum(1).mean(),
                rtol=0, atol=0,
            )
            torch.testing.assert_close(
                fc, -F.cosine_similarity(prediction.float(), projection.float(), dim=-1).mean(),
                rtol=0, atol=0,
            )
            (ce + fc).backward()
            self.assertIsNone(projection.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_unsupported_cuda_amp_fails_with_clear_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                PrecisionPolicy.from_config(_config(True).training, "cuda")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "training.amp=False"):
                PrecisionPolicy.from_config(_config(True).training, "cuda")

    def test_float64_loss_is_not_downgraded(self):
        logits = torch.randn(2, 3, dtype=torch.float64)
        targets = F.softmax(torch.randn_like(logits), dim=1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(soft_cross_entropy(logits, targets).dtype, torch.float64)
            self.assertEqual(negative_cosine_similarity(logits, targets).dtype, torch.float64)

    def test_structural_forward_can_use_bf16_but_ce_is_fp32(self):
        encoder, classifier = nn.Linear(4, 6), nn.Linear(6, 3)
        outputs = []
        handle = classifier.register_forward_hook(lambda _, args, output: outputs.append(output))
        targets = torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = structural_mixup_loss(
                    encoder, classifier, torch.randn(2, 4), torch.randn(2, 4),
                    targets, mixup_alpha=4.0,
                )
        finally:
            handle.remove()
        self.assertEqual(outputs[0].dtype, torch.bfloat16)
        self.assertEqual(loss.dtype, torch.float32)
        expected = -(F.log_softmax(outputs[0].float(), dim=1) * targets[0]).sum(1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(encoder.weight.grad).all())

    def test_channels_last_survives_two_view_mixup(self):
        views = [torch.randn(3, 3, 8, 8).to(memory_format=torch.channels_last) for _ in range(2)]
        mixed, _, _ = mixup_soft_target_views(*views, torch.eye(3), alpha=4.0)
        self.assertEqual(mixed.shape, (6, 3, 8, 8))
        self.assertTrue(mixed.is_contiguous(memory_format=torch.channels_last))

    def test_cpu_amp_config_keeps_exact_fp32_training(self):
        first = _networks("cpu")
        second = copy.deepcopy(first)
        losses = []
        for networks, amp in ((first, False), (second, True)):
            torch.manual_seed(0)
            np.random.seed(0)
            losses.append(_train(networks, _config(amp), torch.device("cpu")))
        self.assertEqual(losses[0], losses[1])
        for left, right in zip(first.all_modules(), second.all_modules()):
            for name, tensor in left.state_dict().items():
                torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0, atol=0)

    def test_bf16_training_branches_accumulate_finite_gradients(self):
        # CUDA가 없는 CI에서도 BF16 forward + FP32 loss의 전체 경로 검증
        networks = _networks("cpu")
        output_dtypes = []
        output_grad_flags = []
        def record_forward(_, args, output):
            output_dtypes.append(output.dtype)
            output_grad_flags.append(output.requires_grad)

        handle = networks.encoder.register_forward_hook(
            record_forward
        )
        try:
            with patch.object(
                PrecisionPolicy, "autocast",
                lambda _: torch.autocast("cpu", dtype=torch.bfloat16),
            ):
                losses = _train(networks, _config(False), torch.device("cpu"))
        finally:
            handle.remove()
        self.assertTrue(output_dtypes)
        self.assertEqual(set(output_dtypes), {torch.bfloat16})
        # CE, no-grad weak, grad strong, LSL 순서; autocast cache 재사용 버그 감지
        self.assertEqual(output_grad_flags, [True, False, True, True] * 2)
        self.assertTrue(np.isfinite(losses.supervised_loss))
        self.assertTrue(np.isfinite(losses.feature_consistency_loss))
        self.assertTrue(np.isfinite(losses.structural_loss))
        for module in networks.all_modules():
            for parameter in module.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_bf16_optimizer_step_keeps_fp32_parameters(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 support required")
        device = torch.device("cuda")
        networks = _networks(device)
        networks.encoder.to(memory_format=torch.channels_last)
        parameters = [p for module in networks.all_modules() for p in module.parameters()]
        before = [p.detach().clone() for p in parameters]
        policy = PrecisionPolicy.from_config(_config(True).training, device)
        images = policy.to_device(torch.randn(2, 3, 8, 8))
        self.assertTrue(images.is_contiguous(memory_format=torch.channels_last))
        with policy.autocast():
            self.assertEqual(networks.encoder(images).dtype, torch.bfloat16)
        with full_precision(device):
            self.assertEqual(networks.encoder(images).dtype, torch.float32)
        losses = _train(networks, _config(True), device)
        self.assertTrue(np.isfinite(losses.supervised_loss))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, parameters)))
        for parameter in parameters:
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
