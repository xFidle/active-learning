from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.config import ConfigParser, register_config
from src.models.classifier import CLASSIFIERS, Classifier, ClassifierName, resolve_classifier
from src.models.forest.forest import RandomForestConfig
from src.models.svm.svm import SVMConfig

type Results = dict[str, dict[str, list[float]]]


@register_config(name="aggregator")
@dataclass
class AggregatorConfig:
    runs: int = 10
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    output: str = "results/compare"


class Aggregator:
    def __init__(self, config: AggregatorConfig) -> None:
        self._runs = config.runs
        self._metrics = config.metrics
        self._output = Path(config.output)
        self._output.mkdir(parents=True, exist_ok=True)

    def aggregate(
        self, X: np.ndarray, y: np.ndarray, classifiers: list[str], p: ConfigParser
    ) -> None:
        all_results: Results = {}
        for name in classifiers:
            results: dict[str, list[float]] = {m: [] for m in self._metrics}

            for seed in range(self._runs):
                X_train, X_test, y_train, y_test = cast(
                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y),
                )

                if name in CLASSIFIERS:
                    clf = resolve_classifier(cast(ClassifierName, name), p)
                    clf.set_rng(seed)
                else:
                    clf = resolve_sklearn_classifier(
                        cast(ClassifierName, name.split("-")[1]), p, seed
                    )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                results["accuracy"].append(float(accuracy_score(y_test, y_pred)))
                results["precision"].append(float(precision_score(y_test, y_pred, zero_division=0)))
                results["recall"].append(float(recall_score(y_test, y_pred, zero_division=0)))
                results["f1"].append(float(f1_score(y_test, y_pred, zero_division=0)))

            all_results[name] = results

        self._save_results(all_results, f"{classifiers[0]}")

    def _save_results(self, data: Results, filename: str) -> None:
        for metric in self._metrics:
            table = {}
            for model_name, results in data.items():
                values = np.array(results[metric])
                table[model_name] = {
                    "Name": model_name,
                    "Mean": values.mean(),
                    "Std": values.std(),
                    "Max": values.max(),
                    "Min": values.min(),
                }

            df = pd.DataFrame(table).T
            df.to_csv(self._output / f"{filename}-{metric}.csv", float_format="%.3f", index=False)


def resolve_sklearn_classifier(
    classifier: ClassifierName, config_parser: ConfigParser, seed: int
) -> Classifier:
    match classifier:
        case "svm":
            svm_config = config_parser.get(SVMConfig)
            return cast(Classifier, LinearSVC(C=svm_config.penalty, max_iter=svm_config.iter_count))
        case "forest":
            forest_config = config_parser.get(RandomForestConfig)
            return cast(
                Classifier,
                RandomForestClassifier(
                    random_state=seed,
                    n_estimators=forest_config.n_trees,
                    criterion="gini",
                    max_features="sqrt",
                    max_depth=forest_config.tree_config.max_depth,
                    min_samples_split=forest_config.tree_config.min_samples_split,
                ),
            )
