import numpy as np


def split_at_threshold(
    dataset: np.ndarray, feature_index: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    feature_column = dataset[:, feature_index]
    return dataset[feature_column >= threshold], dataset[feature_column < threshold]


def highest_probability_arg(proba: np.ndarray) -> int:
    most_frequent = np.argwhere(proba == np.max(proba)).flatten()
    return np.random.choice(most_frequent)


def majority_vote(votes: np.ndarray) -> int:
    unique, counts = np.unique(votes, return_counts=True)
    most_frequent = unique[counts == np.max(counts)]
    return np.random.choice(most_frequent)
