from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import skfuzzy as fuzz
from sklearn.decomposition import PCA

from src.config.base import register_config

if TYPE_CHECKING:
    from src.initializer.initializer import InitializerName


@register_config(name="cluster_initializer")
@dataclass
class ClusterInitializerConfig:
    clusters: int = 2
    pca_components: int | None = None
    center_ratio: float = 0.6
    border_ratio: float = 0.4


class ClusterInitializer:
    name: "InitializerName" = "cluster"

    def __init__(self, config: ClusterInitializerConfig | None = None) -> None:
        if config is None:
            config = ClusterInitializerConfig()

        self.clusters = config.clusters
        self.pca_components = config.pca_components
        self.center_ratio = config.center_ratio
        self.border_ratio = config.border_ratio

    def __call__(
        self, x: np.ndarray, y: np.ndarray, n_labeled: int, rng: np.random.Generator
    ) -> np.ndarray:
        while True:
            labeled_mask = self._select_instances(x, n_labeled, rng)

            labeled_y = y[labeled_mask]
            if len(np.unique(labeled_y)) == len(np.unique(y)):
                return labeled_mask

    def _select_instances(
        self, x: np.ndarray, n_labeled: int, rng: np.random.Generator
    ) -> np.ndarray:
        n_samples, n_features = x.shape

        n_components = self.pca_components
        if n_components is None:
            n_components = min(n_features, n_samples) // 2

        pca = PCA(n_components=n_components, random_state=rng.integers(0, 2**31))
        x_reduced = pca.fit_transform(x)

        _, u, _, _, _, _, _ = fuzz.cmeans(
            x_reduced.T,
            c=self.clusters,
            m=2,
            error=0.005,
            maxiter=1000,
            seed=rng.integers(0, 2**31),
        )

        n_center = int(n_labeled * self.center_ratio)
        n_border = n_labeled - n_center

        center_indices = self._select_cluster_centers(u, n_center)

        border_indices = self._select_cluster_borders(u, n_border)

        selected_indices = np.unique(np.concatenate([center_indices, border_indices]))

        if len(selected_indices) > n_labeled:
            selected_indices = rng.choice(selected_indices, size=n_labeled, replace=False)
        elif len(selected_indices) < n_labeled:
            remaining = n_labeled - len(selected_indices)
            available_indices = np.setdiff1d(np.arange(n_samples), selected_indices)
            additional_indices = rng.choice(available_indices, size=remaining, replace=False)
            selected_indices = np.concatenate([selected_indices, additional_indices])

        labeled_mask = np.zeros(n_samples, dtype=bool)
        labeled_mask[selected_indices] = True

        return labeled_mask

    def _select_cluster_centers(self, u: np.ndarray, n_total: int) -> np.ndarray:
        n_clusters, _ = u.shape
        n_per_cluster = n_total // n_clusters
        center_indices = []

        for cluster_idx in range(n_clusters):
            cluster_memberships = u[cluster_idx, :]
            top_indices = np.argsort(cluster_memberships)[-n_per_cluster:]
            center_indices.extend(top_indices)

        return np.array(center_indices)

    def _select_cluster_borders(self, u: np.ndarray, n_total: int) -> np.ndarray:
        sorted_memberships = np.sort(u, axis=0)
        top_two_diff = np.abs(sorted_memberships[-1, :] - sorted_memberships[-2, :])

        border_indices = np.argsort(top_two_diff)[:n_total]

        return border_indices
