import argparse
from typing import cast

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import ConfigParser
from src.models.classifier import CLASSIFIERS
from src.utils.aggregator import Aggregator, AggregatorConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Results comparision")
    parser.add_argument("classifier", choices=CLASSIFIERS)
    args = parser.parse_args()
    return args


def prepare_dataset() -> tuple[np.ndarray, np.ndarray]:
    wine_quality = fetch_ucirepo(id=186)
    x = cast(pd.DataFrame, wine_quality.data.features)  # pyright: ignore
    y = cast(pd.DataFrame, wine_quality.data.targets)  # pyright: ignore

    y = (y.iloc[:, 0] >= 6).astype(int)
    x = x.to_numpy()
    y = y.to_numpy()

    return x, y


def main():
    args = get_args()
    X, y = prepare_dataset()

    config_parser = ConfigParser()
    aggregator_config = config_parser.get(AggregatorConfig)
    aggregator = Aggregator(aggregator_config)

    aggregator.aggregate(X, y, [f"{args.classifier}", f"SKLEARN-{args.classifier}"], config_parser)


if __name__ == "__main__":
    main()
