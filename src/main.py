from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.forest.cart import CARTConfig
from src.forest.forest import RandomForest, RandomForestConfig
from src.learner.learner import ActiveLearner, ActiveLearnerConfig, LearningData
from src.selector.selector import UncertaintySelector
from src.utils.config_parser import ConfigParser
from src.utils.logger import setup_logger


def parse_args():
    config_parser = ConfigParser()
    logger_config = config_parser.get_logger_config()
    setup_logger(logger_config)


def main():
    parse_args()

    session_name = "forest-test"

    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 6, 1, 0).ravel()

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.8)
    )

    cart_config = CARTConfig(10, 2)
    forest_config = RandomForestConfig(100, cart_config)
    forest = RandomForest(forest_config)

    learner_config = ActiveLearnerConfig(
        forest,
        UncertaintySelector(forest),
        Path(session_name),
        LearningData(x_train[:-10, :], y_train[:-10], x_train[-10:, :]),
    )

    learner = ActiveLearner(learner_config)
    learner.loop(x_test, y_test)


if __name__ == "__main__":
    main()
