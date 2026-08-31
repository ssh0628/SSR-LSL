"""Class-balanced oversampling retained from the official SSR implementation."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    def __init__(self, labels: torch.Tensor, num_classes: int = 10, num_fold: int = 1) -> None:
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.classes = torch.arange(num_classes)
        class_counts = torch.as_tensor(
            [torch.sum(self.labels == class_id) for class_id in self.classes],
            dtype=torch.int,
        )
        if class_counts.max().item() == 0:
            raise ValueError("SSR selected no samples; a balanced loader cannot be constructed.")

        maximum_count = int(class_counts.max().item())
        target_count = maximum_count * num_fold
        sampled_ids: list[torch.Tensor] = []
        for class_id, class_count_tensor in zip(self.classes, class_counts, strict=True):
            class_count = int(class_count_tensor.item())
            if class_count == 0:
                continue
            folds = (target_count + class_count - 1) // class_count
            class_ids = torch.where(self.labels == class_id)[0].repeat(folds)
            random_tail = torch.randperm(maximum_count)
            class_ids[-maximum_count:] = class_ids[-maximum_count:][random_tail]
            sampled_ids.append(class_ids[:target_count])
        self.ids = torch.cat(sampled_ids)

    def __iter__(self):
        permutation = torch.randperm(len(self.ids))
        return iter(self.ids[permutation].tolist())

    def __len__(self) -> int:
        return len(self.ids)
