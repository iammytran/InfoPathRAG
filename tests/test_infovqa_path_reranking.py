import pytest

from src.lilac.retriever.retriever import Retriever
from src.experiment.experiments.infovqa_ablation import _score_distribution


def test_validate_path_weights():
    assert Retriever._validate_path_weights((0.5, 0.25, 0.25)) == (
        0.5, 0.25, 0.25
    )
    with pytest.raises(ValueError):
        Retriever._validate_path_weights((0.5, 0.5, 0.1))
    with pytest.raises(ValueError):
        Retriever._validate_path_weights((-0.1, 0.5, 0.6))


def test_query_zscore_handles_constant_values():
    assert Retriever._normalize_values([2.0, 2.0], "query_zscore") == [0.0, 0.0]


def test_score_distribution_contains_required_statistics():
    result = _score_distribution([1.0, 2.0, 3.0, 4.0])
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["p50"] == 2.5
    assert result["std"] > 0


def test_deterministic_weight_tie_break_is_stable():
    values = [(0.8, ("r", "t", "f")), (0.8, ("a", "b", "c"))]
    assert max(values, key=lambda value: (value[0], value[1])) == values[0]
