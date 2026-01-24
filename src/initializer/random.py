from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.config import register_config

if TYPE_CHECKING:
    from src.initializer.initializer import InitializerName


@register_config(name="random_initalizer")
@dataclass
class RandomInitalizerConfig:
    seed: int = 42


class RandomInitializer:
    name: "InitializerName" = "random"

    def __init__(self, config: RandomInitalizerConfig) -> None:
        self.seed = config.seed

    def __call__(self, x: np.ndarray, y: np.ndarray, n_labeled: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n_samples = x.shape[0]

        labeled_mask = np.zeros(n_samples, dtype=bool)
        labeled_mask[:n_labeled] = True
        rng.shuffle(labeled_mask)

        return labeled_mask
