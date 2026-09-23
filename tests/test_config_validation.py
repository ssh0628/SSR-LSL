from __future__ import annotations

import unittest

from setting import config


class ConfigValidationTest(unittest.TestCase):
    def test_nonfinite_optimization_values_fail_before_training(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            cases = (
                config.TrainingConfig(learning_rate=value),
                config.TrainingConfig(encoder_learning_rate=value),
                config.TrainingConfig(weight_decay=value),
                config.SSRConfig(mixup_alpha=value),
                config.SSRConfig(feature_consistency_weight=value),
                config.StructuralLabelsConfig(loss_weight=value),
            )
            for candidate in cases:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        candidate.validate()

    def test_nonfinite_image_preprocessing_is_rejected(self):
        for candidate in (
            config.DataConfig(mean=(float("nan"), 0.5, 0.5)),
            config.DataConfig(std=(float("inf"), 0.5, 0.5)),
            config.AugmentationConfig(weak_rotation=float("nan")),
            config.AugmentationConfig(color_jitter=(0.1, float("inf"), 0.1, 0.0)),
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    candidate.validate()


if __name__ == "__main__":
    unittest.main()
