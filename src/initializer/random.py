from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.initializer.initializer import InitializerName


class RandomInitializer:
    name: "InitializerName" = "random"

    def __call__(self, x: np.ndarray, y: np.ndarray, n_labeled: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_samples = x.shape[0]

        labeled_mask = np.zeros(n_samples, dtype=bool)
        labeled_mask[:n_labeled] = True
        rng.shuffle(labeled_mask)

        return labeled_mask
