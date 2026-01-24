import argparse

import pandas as pd

from src.active_learning.tester import ActiveLearnerConfig, LearnerTester, TesterConfig
from src.config import ConfigParser
from src.utils.logger import LoggerConfig, setup_root_logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Active Learning")
    parser.add_argument("-c", "--config", default="config")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config_parser = ConfigParser(args.config)
    logger_config = config_parser.get(LoggerConfig)
    learner_config = config_parser.get(ActiveLearnerConfig)
    tester_config = config_parser.get(TesterConfig)
    setup_root_logger(logger_config)

    df = pd.read_csv("data/active_learning/flowers.csv")  # pyright: ignore

    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    x = x.to_numpy()
    y = y.to_numpy()

    tester = LearnerTester(learner_config, tester_config)
    tester.run(x, y)


if __name__ == "__main__":
    main()
