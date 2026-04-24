# test_evaluate.py
# Unit tests for evaluation metrics.
# Owner: Colton Spurgin

import pytest
from evaluation.evaluate import recall_at_k, precision_at_k, ndcg_at_k, mrr


class TestRecallAtK:
    def test_full_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c", "d", "e"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(3 / 5)

    def test_no_recall(self):
        retrieved = ["a", "b"]
        relevant = {"c", "d"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_k_limits_retrieved(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(2 / 4)


class TestPrecisionAtK:
    def test_full_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_precision(self):
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_no_precision(self):
        retrieved = ["a", "b"]
        relevant = {"c", "d"}
        assert precision_at_k(retrieved, relevant, 2) == 0.0

    def test_k_zero(self):
        retrieved = ["a"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, 0) == 0.0

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, 10) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_degraded_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_relevant_at_end(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        dcg = 1.0 / 2
        idcg = 1.0
        expected = dcg / idcg
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0


class TestMrr:
    def test_first_is_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == pytest.approx(1.0)

    def test_second_is_relevant(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == pytest.approx(1.0 / 2)

    def test_no_relevant_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert mrr(retrieved, relevant) == 0.0

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a"}
        assert mrr(retrieved, relevant) == 0.0

    def test_multiple_relevant_returns_first(self):
        retrieved = ["a", "b", "c"]
        relevant = {"b", "c"}
        assert mrr(retrieved, relevant) == pytest.approx(1.0 / 2)
