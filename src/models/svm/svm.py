from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.config.base import register_config

if TYPE_CHECKING:
    from src.models.classifier import ClassifierName


@register_config("svm")
@dataclass
class SVMConfig:
    learning_rate: float = 0.1
    penalty: float = 100  # C param
    iter_count: int = 1000


class SVM:
    name: "ClassifierName" = "svm"

    def __init__(self, config: SVMConfig | None = None) -> None:
        if config is None:
            config = SVMConfig()

        self.learning_rate: float = config.learning_rate
        self.penalty: float = config.penalty
        self.iter_count: int = config.iter_count
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.platt_a: float | None = None
        self.platt_b: float | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.rng: np.random.Generator = np.random.default_rng()

    def set_rng(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n_samples, n_features = X_train.shape

        feature_mean = X_train.mean(axis=0)
        feature_scale = X_train.std(axis=0)
        feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        X_scaled = self._standardize(X_train)

        w = np.zeros(n_features)
        b = 0.0

        y = np.where(y_train <= 0, -1, 1)
        lambda_param = 1.0 / (self.penalty * n_samples)
        step = 0

        for _ in range(self.iter_count):
            for idx in self.rng.permutation(n_samples):
                step += 1
                x_i = X_scaled[idx]
                condition = y[idx] * (np.dot(x_i, w) + b) >= 1
                eta = self.learning_rate / (1.0 + self.learning_rate * lambda_param * step)

                if condition:
                    w = (1 - eta * lambda_param) * w
                else:
                    w = (1 - eta * lambda_param) * w + eta * y[idx] * x_i
                    b += eta * y[idx]

        self.w = w
        self.b = b

        self._fit_platt_scaling(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.sign(self._decision_function(X))
        return np.where(pred <= 0, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.platt_a is None or self.platt_b is None:
            raise ValueError("Model not trained. Call fit() first.")

        decision = self._decision_function(X)
        proba_class_1 = expit(-(self.platt_a * decision + self.platt_b))
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise ValueError("Model not trained. Call fit() first.")

        return (X - self.feature_mean) / self.feature_scale

    def _fit_platt_scaling(self, X: np.ndarray, Y: np.ndarray) -> None:
        decision_values = self._decision_function(X)

        y_binary = np.where(Y <= 0, 0, 1).flatten()

        n_pos = np.sum(y_binary == 1)
        n_neg = np.sum(y_binary == 0)
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)

        targets = np.where(y_binary == 1, t_pos, t_neg)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params
            pred_proba = expit(-(a * decision_values + b))
            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
            nll = -np.sum(targets * np.log(pred_proba) + (1 - targets) * np.log(1 - pred_proba))
            return nll

        result = minimize(neg_log_likelihood, x0=[0.0, 0.0], method="BFGS")
        self.platt_a, self.platt_b = result.x

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self._standardize(X)
        return np.dot(X_scaled, self.w) + self.b
