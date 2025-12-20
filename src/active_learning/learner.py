from dataclasses import dataclass

import numpy as np

from src.models.classifier import Classifier

from .selector import Selector


@dataclass
class PredictionSnapshot:
    labeled_ratio: float
    proba: np.ndarray


@dataclass
class ExperimentResults:
    y_test: np.ndarray
    snapshots: list[PredictionSnapshot]


@dataclass
class ActiveLearnerConfig:
    classifier: Classifier
    selector: Selector
    batch_size: int
    store_results: bool


@dataclass
class LearningData:
    X_train: np.ndarray
    y_train: np.ndarray
    labeled_mask: np.ndarray  # True for already labled samples, False otherwise


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig, data: LearningData) -> None:
        self.classifier = config.classifier
        self.selector = config.selector
        self.batch_size = config.batch_size
        self.store_results = config.store_results
        self.data = data

    def loop(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        if self.store_results:
            self.experiment_results: ExperimentResults = ExperimentResults(y_test, [])

        self.classifier.fit(
            self.data.X_train[self.data.labeled_mask], self.data.y_train[self.data.labeled_mask]
        )

        while np.flatnonzero(~self.data.labeled_mask).size != 0:
            print("Unlabeled remaining:", np.flatnonzero(~self.data.labeled_mask).size)
            samples_indices = self.selector(
                self.data.X_train, self.data.labeled_mask, self.batch_size
            )

            self.data.labeled_mask[samples_indices] = True

            self.classifier.fit(
                self.data.X_train[self.data.labeled_mask], self.data.y_train[self.data.labeled_mask]
            )

            if self.store_results:
                self._store_experiment_results(X_test)

    def _store_experiment_results(self, X_test: np.ndarray) -> None:
        proba = self.classifier.predict_proba(X_test)
        majority_class_proba = proba[:, 1]
        self.experiment_results.snapshots.append(
            PredictionSnapshot(self._get_labeled_ratio(), majority_class_proba)
        )

    def _get_labeled_ratio(self) -> float:
        labeled_indices = np.flatnonzero(self.data.labeled_mask)
        unlabeled_indices = np.flatnonzero(~self.data.labeled_mask)
        return labeled_indices.size / (labeled_indices.size + unlabeled_indices.size)
