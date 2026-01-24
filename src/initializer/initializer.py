from typing import Literal, Protocol

import numpy as np

from src.config.parser import ConfigParser
from src.initializer.cluster import ClusterInitializer, ClusterInitializerConfig
from src.initializer.random import RandomInitalizerConfig, RandomInitializer

type InitializerName = Literal["random", "cluster"]
INITIALIZERS = ["random", "cluster"]


class Initializer(Protocol):
    name: InitializerName

    def __call__(self, x: np.ndarray, y: np.ndarray, n_labeled: int) -> np.ndarray: ...


def resolve_initializer(name: InitializerName, p: ConfigParser) -> Initializer:
    match name:
        case "random":
            return RandomInitializer(p.get(RandomInitalizerConfig))

        case "cluster":
            return ClusterInitializer(p.get(ClusterInitializerConfig))
