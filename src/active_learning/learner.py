from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any

import numpy as np

from src.active_learning.selector import Selector, SelectorName, resolve_selector
from src.config import register_config
from src.models.classifier import Classifier, resolve_classifier
from src.models.forest.forest import RandomForest


@dataclass
class ExperimentResults:
    labeled_ratio: list[float]
    y_test: np.ndarray
    proba: np.ndarray


@register_config(
    name="active_learner",
    field_mappings={"_selector_name": "selector"},
    field_parsers={"classifier": lambda name, parser: resolve_classifier(name, parser)},
    field_serializers={"classifier": lambda classifier: classifier.name},
)
@dataclass
class ActiveLearnerConfig:
    classifier: Classifier = field(default_factory=RandomForest)
    _selector_name: SelectorName = "uncertainty"
    _selector_seed: int = 42
    _selector: Selector | None = None
    batch_size: int = 10
    should_store_results: bool = True

    @property
    def selector(self) -> Selector:
        if self._selector is None:
            return resolve_selector(self._selector_name, self.classifier, self._selector_seed)
        return self._selector


@dataclass
class LearningData:
    X_train: np.ndarray
    y_train: np.ndarray
    labeled_mask: np.ndarray


@dataclass
class MultiprocessingContext:
    learner_id: int
    queue: "Queue[dict[str, Any]]"


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig, data: LearningData) -> None:
        self._classifier = config.classifier
        self._selector = config.selector
        self._batch_size = config.batch_size
        self._should_store_results = config.should_store_results
        self._data = data

    def loop(
        self, X_test: np.ndarray, y_test: np.ndarray, ctx: MultiprocessingContext | None = None
    ) -> None:
        self._classifier.fit(
            self._data.X_train[self._data.labeled_mask], self._data.y_train[self._data.labeled_mask]
        )

        unlabeled = int(np.flatnonzero(~self._data.labeled_mask).shape[0])
        n_iter = unlabeled // self._batch_size + 2

        if self._should_store_results:
            self._prepare_results_arrays(X_test, y_test, n_iter)

        for i in range(1, n_iter):
            samples_indices = self._selector(
                self._data.X_train, self._data.labeled_mask, self._batch_size
            )

            self._data.labeled_mask[samples_indices] = True

            self._classifier.fit(
                self._data.X_train[self._data.labeled_mask],
                self._data.y_train[self._data.labeled_mask],
            )

            if self._should_store_results:
                self._store_results(X_test, i)

            if ctx is not None:
                ctx.queue.put({"task_id": ctx.learner_id, "total": n_iter - 1, "completed": i})

    def _prepare_results_arrays(self, X_test: np.ndarray, y_test: np.ndarray, n_iter: int) -> None:
        self.results = ExperimentResults(
            y_test=y_test, labeled_ratio=[0] * n_iter, proba=np.empty((n_iter, y_test.shape[0]))
        )

        self._store_results(X_test, 0)

    def _store_results(self, X_test: np.ndarray, iteration: int) -> None:
        proba = self._classifier.predict_proba(X_test)
        majority_class_proba = proba[:, 1]
        self.results.labeled_ratio[iteration] = self._get_labeled_ratio()
        self.results.proba[iteration] = majority_class_proba

    def _get_labeled_ratio(self) -> float:
        labeled_indices = np.flatnonzero(self._data.labeled_mask)
        unlabeled_indices = np.flatnonzero(~self._data.labeled_mask)
        return labeled_indices.size / (labeled_indices.size + unlabeled_indices.size)
