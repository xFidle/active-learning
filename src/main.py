from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.active_learning.learner import ActiveLearnerConfig, LearnerTester, TesterConfig
from src.active_learning.selector import resolve_selector
from src.config import ConfigParser, LoggerConfig
from src.models.classifier import resolve_model
from src.utils.logger import setup_logger


def main():
    config_parser = ConfigParser()
    logger_config = config_parser.get(LoggerConfig)
    setup_logger(logger_config)
    df = pd.read_csv("data/active_learning/flowers.csv")

    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    x = x.to_numpy()
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.8)
    )

    classifier = resolve_model("forest")
    selector = resolve_selector("random", classifier)

    learner_config = ActiveLearnerConfig(classifier, selector, 5, True)
    tester_config = TesterConfig(Path("abc"), n_splits=2, n_repeats=1)
    tester = LearnerTester(learner_config, tester_config)
    print(tester.aggregate_results(x, y))


if __name__ == "__main__":
    main()
