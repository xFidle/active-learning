import numpy as np


def highest_probability_arg(proba: np.ndarray) -> int:
    return int(np.argmax(proba))


def majority_vote(votes: np.ndarray) -> int:
    unique, counts = np.unique(votes, return_counts=True)
    return unique[np.argmax(counts)]


def gini_impurity(counts: np.ndarray) -> float:
    probabilities = counts / np.sum(counts)
    if probabilities.size == 0:
        return 0.0

    return float(1 - np.sum(np.power(probabilities, 2)))
