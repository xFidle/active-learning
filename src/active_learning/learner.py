from dataclasses import dataclass
from multiprocessing import Event
from multiprocessing.managers import DictProxy
from typing import Any

import numpy as np

from src.models.classifier import Classifier

from .selector import Selector


@dataclass
class ExperimentResults:
    labeled_ratio: list[float]
    y_test: np.ndarray
    proba: np.ndarray


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
    labeled_mask: np.ndarray


@dataclass
class MultiprocessingContext:
    learner_id: int
    proxy: DictProxy[Any, Any]
    event: Event


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig, data: LearningData) -> None:
        self.classifier = config.classifier
        self.selector = config.selector
        self.batch_size = config.batch_size
        self.store_results = config.store_results
        self.data = data

    def loop(
        self, X_test: np.ndarray, y_test: np.ndarray, ctx: MultiprocessingContext | None = None
    ) -> None:
        self.classifier.fit(
            self.data.X_train[self.data.labeled_mask], self.data.y_train[self.data.labeled_mask]
        )

        unlabeled = int(np.flatnonzero(~self.data.labeled_mask).shape[0])
        n_iter = unlabeled // self.batch_size + 2

        if self.store_results:
            self._prepare_results_arrays(X_test, y_test, n_iter)

        for i in range(1, n_iter):
            samples_indices = self.selector(
                self.data.X_train, self.data.labeled_mask, self.batch_size
            )

            self.data.labeled_mask[samples_indices] = True

            self.classifier.fit(
                self.data.X_train[self.data.labeled_mask], self.data.y_train[self.data.labeled_mask]
            )

            if self.store_results:
                self._store_results(X_test, i)

            if ctx is not None:
                ctx.proxy[ctx.learner_id] = {"total": n_iter - 1, "completed": i}
                ctx.event.set()

    def _prepare_results_arrays(self, X_test: np.ndarray, y_test: np.ndarray, n_iter: int) -> None:
        self.results = ExperimentResults(
            y_test=y_test, labeled_ratio=[0] * n_iter, proba=np.empty((n_iter, y_test.shape[0]))
        )

        self._store_results(X_test, 0)

    def _store_results(self, X_test: np.ndarray, iteration: int) -> None:
        proba = self.classifier.predict_proba(X_test)
        majority_class_proba = proba[:, 1]
        self.results.labeled_ratio[iteration] = self._get_labeled_ratio()
        self.results.proba[iteration] = majority_class_proba

    def _get_labeled_ratio(self) -> float:
        labeled_indices = np.flatnonzero(self.data.labeled_mask)
        unlabeled_indices = np.flatnonzero(~self.data.labeled_mask)
        return labeled_indices.size / (labeled_indices.size + unlabeled_indices.size)
