import argparse

from src.active_learning.selector import SELECTORS
from src.active_learning.tester import TesterConfig
from src.config import ConfigParser
from src.initializer.initializer import INITIALIZERS
from src.models.classifier import CLASSIFIERS
from src.utils.plotter import Plotter, PlotterConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="PR Curves")
    parser.add_argument("classifier", type=str, choices=CLASSIFIERS, help="ClassifierName")
    parser.add_argument("initializer", type=str, choices=INITIALIZERS, help="InitializerName")
    parser.add_argument("selector", type=str, choices=SELECTORS, help="SelectorName")
    parser.add_argument("--ths", type=float, nargs="+", help="List of thresholds")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    config_parser = ConfigParser()
    tester_config = config_parser.get(TesterConfig)
    plotter_config = config_parser.get(PlotterConfig)

    plotter = Plotter(plotter_config)
    ths = args.ths if args.ths is not None else tester_config.thresholds
    ths = list(map(str, ths))
    plotter.plot_pr_auc(args.classifier, args.initializer, args.selector, ths)


if __name__ == "__main__":
    main()
