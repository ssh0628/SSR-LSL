"""Single-loop k-NN must retain the previous query partitions and scores."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from ssr.knn import balanced_knn_scores, hard_knn_scores, similarity_chunk_size


def _legacy_hard(queries, bank, labels, classes, neighbors):
    """Previous standalone voter, including its own memory-bound inner loop."""
    scores = torch.zeros(len(queries), classes, device=labels.device, dtype=queries.dtype)
    step = similarity_chunk_size(len(queries), bank.size(1), queries.element_size(), 1)
    for start in range(0, len(queries), step):
        end = min(start + step, len(queries))
        similarity = torch.mm(queries[start:end], bank)
        indices = similarity.topk(k=neighbors, dim=-1).indices
        del similarity
        neighbor_labels = labels[indices]
        scores[start:end].scatter_add_(
            1, neighbor_labels, torch.ones_like(neighbor_labels, dtype=scores.dtype)
        )
    return scores / neighbors


def _legacy_balanced(queries, bank, labels, classes, neighbors, chunks):
    """Previous outer loop plus per-chunk vote division and concatenation."""
    counts = torch.bincount(labels, minlength=classes).to(bank.dtype) + 1e-10
    prior = counts / counts.sum()
    normalized_bank = F.normalize(bank, dim=1)
    queries = normalized_bank if queries is bank else F.normalize(queries, dim=1)
    step = similarity_chunk_size(len(queries), len(bank), bank.element_size(), chunks)
    parts = [
        _legacy_hard(queries[start:start + step], normalized_bank.T, labels, classes, neighbors)
        for start in range(0, len(queries), step)
    ]
    if not parts:
        return torch.empty((0, classes), device=bank.device)
    scores = torch.cat(parts, dim=0) / prior
    return scores / scores.sum(dim=1, keepdim=True)


def _trace_matmuls(call):
    inputs = []
    original_mm = torch.mm

    def record(queries, bank):
        inputs.append((queries.clone(), bank.shape))
        return original_mm(queries, bank)

    with patch("torch.mm", side_effect=record):
        output = call()
    return output, inputs


class KNNCleanupTest(unittest.TestCase):
    def test_balanced_matches_previous_scores_and_every_query_chunk_exactly(self):
        generator = torch.Generator().manual_seed(34)
        for dtype in (torch.float32, torch.float64):
            bank = torch.randn(23, 7, dtype=dtype, generator=generator)
            # Preserve existing handling of zero features and tied neighbors.
            bank[0].zero_()
            bank[1] = bank[2]
            labels = torch.arange(len(bank)) % 3
            for same_bank in (False, True):
                queries = bank if same_bank else bank[:13].clone() * 2
                for chunks in (1, 3, 10, 100):
                    for memory_rows in (1, 5, 100):
                        for neighbors in (1, 5, len(bank)):
                            with self.subTest(dtype=dtype, same_bank=same_bank, chunks=chunks, memory_rows=memory_rows, neighbors=neighbors):
                                limit = len(bank) * bank.element_size() * memory_rows
                                with patch("ssr.knn.MAX_SIMILARITY_BYTES", limit):
                                    expected, old_chunks = _trace_matmuls(
                                        lambda: _legacy_balanced(queries, bank, labels, 4, neighbors, chunks)
                                    )
                                    actual, new_chunks = _trace_matmuls(
                                        lambda: balanced_knn_scores(
                                            queries, bank, labels, num_classes=4,
                                            neighbors=neighbors, chunks=chunks,
                                        )
                                    )
                                self.assertTrue(torch.equal(actual, expected))
                                self.assertEqual(len(new_chunks), len(old_chunks))
                                for (new_query, new_bank), (old_query, old_bank) in zip(new_chunks, old_chunks):
                                    self.assertTrue(torch.equal(new_query, old_query))
                                    self.assertEqual(new_bank, old_bank)

    def test_standalone_five_positional_arguments_keep_default_partitions(self):
        generator = torch.Generator().manual_seed(5)
        for dtype in (torch.float32, torch.float64):
            bank = F.normalize(torch.randn(11, 5, dtype=dtype, generator=generator), dim=1)
            queries = bank[:7]
            labels = torch.arange(len(bank)) % 3
            for neighbors in (1, len(bank)):
                with self.subTest(dtype=dtype, neighbors=neighbors):
                    with patch("ssr.knn.MAX_SIMILARITY_BYTES", 3 * len(bank) * bank.element_size()):
                        expected, old_chunks = _trace_matmuls(lambda: _legacy_hard(queries, bank.T, labels, 4, neighbors))
                        actual, new_chunks = _trace_matmuls(lambda: hard_knn_scores(queries, bank.T, labels, 4, neighbors))
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual([len(value[0]) for value in new_chunks], [len(value[0]) for value in old_chunks])

    def test_empty_queries_keep_existing_output_dtypes(self):
        bank = torch.eye(3, dtype=torch.float64)
        queries = bank[:0]
        labels = torch.arange(3)
        with patch("torch.mm") as multiply:
            votes = hard_knn_scores(queries, bank.T, labels, 3, 1)
            balanced = balanced_knn_scores(queries, bank, labels, num_classes=3, neighbors=1)
        multiply.assert_not_called()
        self.assertEqual(votes.shape, (0, 3))
        self.assertEqual(votes.dtype, bank.dtype)
        self.assertEqual(balanced.shape, (0, 3))
        self.assertEqual(balanced.dtype, torch.get_default_dtype())

    def test_balanced_rejects_invalid_inputs_before_matrix_multiplication(self):
        bank = torch.eye(3)
        labels = torch.arange(3)
        cases = (
            (bank[0], bank, labels, {}, "rank-2"),
            (bank[:, :2], bank, labels, {}, "dimensions"),
            (bank, bank, labels[:2], {}, "one entry"),
            (bank, bank, labels[:, None], {}, "rank-1"),
            (bank, bank, labels, {"neighbors": 0}, "neighbors"),
            (bank, bank, labels, {"neighbors": 4}, "neighbors"),
            (bank[:0], bank[:0], labels[:0], {}, "neighbors"),
            (bank, bank, labels, {"chunks": 0}, "chunks"),
            (bank, bank, torch.tensor([-1, 1, 2]), {}, "class index"),
            (bank, bank, torch.tensor([0, 1, 3]), {}, "class index"),
        )
        for queries, feature_bank, feature_labels, options, error in cases:
            with self.subTest(error=error, options=options):
                with patch("torch.mm") as multiply:
                    with self.assertRaisesRegex(ValueError, error):
                        balanced_knn_scores(
                            queries, feature_bank, feature_labels,
                            **{"num_classes": 3, "neighbors": 1, **options},
                        )
                multiply.assert_not_called()
        with self.assertRaisesRegex(ValueError, "chunks"):
            hard_knn_scores(bank, bank.T, labels, 3, 1, chunks=0)


if __name__ == "__main__":
    unittest.main()
