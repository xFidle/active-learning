import argparse

from src.config import ConfigParser
from src.initializer.initializer import INITIALIZERS
from src.models.classifier import CLASSIFIERS
from src.utils.plotter import Plotter, PlotterConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="PR Curves")
    parser.add_argument("classifier", type=str, choices=CLASSIFIERS, help="ClassifierName")
    parser.add_argument("initializer", type=str, choices=INITIALIZERS, help="InitializerName")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    config_parser = ConfigParser()
    plotter_config = config_parser.get(PlotterConfig)

    plotter = Plotter(plotter_config)
    plotter.plot_learning_curves(args.classifier, args.initializer)


if __name__ == "__main__":
    main()
