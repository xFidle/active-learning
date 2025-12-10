import numpy as np
import pytest

from src.forest.util import highest_probability_arg, majority_vote, split_at_threshold


@pytest.mark.parametrize(
    "dataset, feature_index, threshold, expected_above, expected_below",
    [
        (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            0,
            4,
            np.array([[5, 6], [7, 8]]),
            np.array([[1, 2], [3, 4]]),
        ),
        (np.array([[1, 5], [2, 3], [3, 7]]), 1, 5, np.array([[1, 5], [3, 7]]), np.array([[2, 3]])),
        (np.array([[10, 20], [15, 25]]), 0, 5, np.array([[10, 20], [15, 25]]), np.empty((0, 2))),
        (np.array([[1, 2], [2, 3]]), 0, 5, np.empty((0, 2)), np.array([[1, 2], [2, 3]])),
    ],
)
def test_split_at_threshold(
    dataset: np.ndarray,
    feature_index: int,
    threshold: float,
    expected_above: np.ndarray,
    expected_below: np.ndarray,
):
    above, below = split_at_threshold(dataset, feature_index, threshold)

    assert (above == expected_above).all()
    assert (below == expected_below).all()


@pytest.mark.parametrize(
    "proba, expected",
    [
        (np.array([1]), (0,)),
        (np.array([0.5, 0.5]), (0, 1)),
        (np.array([0.9, 0.1]), (0,)),
        (np.array([0.1, 0.9]), (1,)),
        (np.array([0.33, 0.33, 0.33]), (0, 1, 2)),
    ],
)
def test_highest_probability(proba: np.ndarray, expected: tuple[int]):
    arg = highest_probability_arg(proba)
    assert arg in expected


@pytest.mark.parametrize(
    "votes, expected",
    [
        (np.array([1]), (1,)),
        (np.array([0, 1]), (0, 1)),
        (np.array([1, 1]), (1,)),
        (np.array([0, 1, 1]), (1,)),
        (np.array([1, 1, 1]), (1,)),
        (np.array([0, 1, 0, 1]), (0, 1)),
        (np.array([0, 0, 1, 0]), (0,)),
    ],
)
def test_majority_vote(votes: np.ndarray, expected: tuple[int]):
    label = majority_vote(votes)
    assert label in expected
