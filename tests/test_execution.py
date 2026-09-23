"""Device-safe performance settings and independent evaluation loaders."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from setting.config import CONFIG
from setting.model import SSRNetworks
from ssr.engine import _build_optimizer, _loader_options, configure_execution


class ExecutionTest(unittest.TestCase):
    def test_example_defaults_keep_loader_prefetch_bounded(self):
        train = _loader_options(CONFIG, torch.device("cuda"))
        evaluation = _loader_options(CONFIG, torch.device("cuda"), evaluation=True)
        self.assertEqual(2 * train["num_workers"], 8)
        self.assertEqual(evaluation["num_workers"], 4)
        self.assertEqual(train["prefetch_factor"], 2)
        self.assertEqual(train["num_workers"] * train["prefetch_factor"], 8)
        self.assertEqual((train["batch_size"], evaluation["batch_size"]), (32, 128))

    def test_separate_train_and_evaluation_loader_limits(self):
        config = replace(CONFIG, training=replace(
            CONFIG.training, batch_size=128, num_workers=16,
            eval_batch_size=256, eval_num_workers=8, prefetch_factor=1,
        ))
        train = _loader_options(config, torch.device("cuda"))
        evaluation = _loader_options(config, torch.device("cuda"), evaluation=True)
        self.assertEqual((train["batch_size"], train["num_workers"]), (128, 16))
        self.assertEqual((evaluation["batch_size"], evaluation["num_workers"]), (256, 8))
        self.assertTrue(train["pin_memory"])
        self.assertTrue(train["persistent_workers"])
        self.assertFalse(evaluation["persistent_workers"])
        self.assertEqual(train["prefetch_factor"], 1)

    def test_zero_workers_omit_worker_only_options(self):
        config = replace(CONFIG, training=replace(
            CONFIG.training, num_workers=0, eval_num_workers=0,
        ))
        for evaluation in (False, True):
            options = _loader_options(config, torch.device("cpu"), evaluation=evaluation)
            self.assertNotIn("prefetch_factor", options)
            self.assertNotIn("persistent_workers", options)
            self.assertFalse(options["pin_memory"])

    def test_cpu_does_not_use_fused_optimizer_or_channels_last(self):
        networks = SSRNetworks(*(nn.Linear(3, 3) for _ in range(4)))
        configure_execution(CONFIG, networks, torch.device("cpu"))
        optimizer = _build_optimizer(networks, CONFIG)
        self.assertFalse(optimizer.defaults.get("fused"))
        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertEqual(optimizer.param_groups[0]["lr"], CONFIG.training.encoder_learning_rate)

    def test_cuda_configuration_keeps_selection_tf32_disabled(self):
        previous = (
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        )
        networks = Mock()
        try:
            with patch("ssr.engine.PrecisionPolicy.from_config", return_value=SimpleNamespace(channels_last=True)):
                for deterministic in (False, True):
                    config = replace(CONFIG, runtime=replace(CONFIG.runtime, deterministic=deterministic))
                    configure_execution(config, networks, torch.device("cuda"))
                    self.assertEqual(torch.backends.cudnn.deterministic, deterministic)
                    self.assertEqual(torch.backends.cudnn.benchmark, not deterministic)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
            networks.encoder.to.assert_called_with(memory_format=torch.channels_last)
        finally:
            (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = previous

    def test_invalid_performance_settings_fail_fast(self):
        for field, value in (("eval_batch_size", 0), ("eval_num_workers", -1), ("log_interval", 0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(CONFIG.training, **{field: value}).validate()


if __name__ == "__main__":
    unittest.main()
