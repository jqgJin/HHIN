import unittest

import numpy as np

import hhin_core as core


class RankingTieTests(unittest.TestCase):
    def test_topk_uses_node_id_for_equal_scores(self):
        scores = np.array([0.7, 0.9, 0.9, 0.2])
        ids = np.array([40, 20, 10, 30])
        positions = core.deterministic_topk_indices(scores, 2, ids)
        self.assertEqual(ids[positions].tolist(), [10, 20])

    def test_topk_is_invariant_to_input_permutation(self):
        scores = np.array([0.9, 0.9, 0.7, 0.4])
        ids = np.array([20, 10, 30, 40])
        first = ids[core.deterministic_topk_indices(scores, 2, ids)]
        permutation = np.array([3, 1, 0, 2])
        second = ids[permutation][
            core.deterministic_topk_indices(
                scores[permutation], 2, ids[permutation]
            )
        ]
        self.assertEqual(first.tolist(), second.tolist())

    def test_tie_averaged_ndcg_is_permutation_invariant(self):
        predicted = np.array([0.8, 0.8, 0.3, 0.1])
        truth = np.array([1.0, 0.2, 0.4, 0.0])
        permutation = np.array([1, 0, 3, 2])
        first = core.ndcg_at_k(predicted, truth, 3)
        second = core.ndcg_at_k(predicted[permutation], truth[permutation], 3)
        self.assertAlmostEqual(first, second, places=14)

    def test_tie_aware_overlap_is_one_for_identical_scores(self):
        scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
        self.assertAlmostEqual(core.tie_aware_overlap_at_k(scores, scores, 2), 1.0)

    def test_identical_rankings_have_perfect_metrics(self):
        scores = np.array([1.0, 0.8, 0.8, 0.2])
        metrics = core.ranking_metrics(scores, scores, self_idx=0)
        self.assertEqual(metrics["top10"], 1.0)
        self.assertAlmostEqual(metrics["ndcg10"], 1.0)


if __name__ == "__main__":
    unittest.main()
