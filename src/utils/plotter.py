from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.active_learning.selector import SelectorName
from src.config.base import register_config
from src.initializer.initializer import InitializerName
from src.models.classifier import ClassifierName


@register_config(name="plotter")
@dataclass
class PlotterConfig:
    results: str = "results"


class Plotter:
    def __init__(self, config: PlotterConfig) -> None:
        self._results = Path(config.results)

    def plot_learning_curves(
        self, classifier: ClassifierName, initializer: InitializerName
    ) -> None:
        for path in self._results.glob("*"):
            elements = path.stem.split("_")
            if not path.is_dir() or elements[0] != classifier or elements[1] != initializer:
                continue
            auc_results = path / "auc-results.csv"
            df = pd.read_csv(auc_results)  # pyright: ignore
            labeled_ratio, mean = (
                np.array(list(map(lambda x: x * 100, df["labeled_ratio"]))),
                df["mean"],
            )
            plt.plot(labeled_ratio, mean, label=elements[2])
        plt.title(f"{classifier} Active Learning Curves ({initializer} initalization)")
        plt.xlabel("% Data")
        plt.ylabel("PR AUC")
        plt.legend()
        plt.grid(visible=True)  # pyright: ignore
        plt.savefig(self._results / f"{classifier}-{initializer}-learning-curve.png")

    def plot_pr_auc(
        self,
        classifier: ClassifierName,
        initializer: InitializerName,
        selector: SelectorName,
        ths: list[str],
    ) -> None:
        dir_name = f"{classifier}_{initializer}_{selector}"
        for file in (self._results / dir_name).glob("*.csv"):
            elements = file.stem.split("-")

            if elements[0] == "precision" and elements[1] == "recall" and elements[2] in ths:
                df = pd.read_csv(file)  # pyright: ignore
                precision, recall = df["precision"], df["recall"]
                auc = np.trapezoid(recall, precision)
                label = f"{int(float(elements[2]))}% data AUC={auc:.3f}"
                plt.plot(recall, precision, label=label)
        plt.title(f"{classifier} PR Curves ({initializer} initalization, {selector} sampling)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(visible=True)  # pyright: ignore
        plt.savefig(self._results / Path(dir_name) / "precision-recall-curve.png")
