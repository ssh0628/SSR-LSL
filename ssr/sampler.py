"""Class-balanced oversampling retained from the official SSR implementation."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    def __init__(self, labels: torch.Tensor, num_classes: int = 10) -> None:
        labels = torch.as_tensor(labels, dtype=torch.long, device="cpu").flatten()
        if labels.numel() == 0:
            raise ValueError("SSR selected no samples; a balanced loader cannot be constructed.")
        if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
            raise ValueError("labels contain an invalid class index.")

        class_counts = torch.bincount(labels, minlength=num_classes)
        maximum_count = int(class_counts.max().item())
        sampled_ids: list[torch.Tensor] = []
        for class_id, class_count_tensor in enumerate(class_counts):
            class_count = int(class_count_tensor.item())
            if class_count == 0:
                continue
            folds = (maximum_count + class_count - 1) // class_count
            class_ids = torch.where(labels == class_id)[0].repeat(folds)
            random_tail = torch.randperm(maximum_count)
            class_ids[-maximum_count:] = class_ids[-maximum_count:][random_tail]
            sampled_ids.append(class_ids[:maximum_count])
        self.ids = torch.cat(sampled_ids)

    def __iter__(self) -> Iterator[int]:
        permutation = torch.randperm(len(self.ids))
        return iter(self.ids[permutation].tolist())

    def __len__(self) -> int:
        return len(self.ids)
