from dataclasses import dataclass

import numpy as np


@dataclass
class SVMConfig:
    learning_rate: float
    penalty: float  # C param
    iter_count: int


class SVM:
    def __init__(self, config: SVMConfig) -> None:
        self.learning_rate: float = config.learning_rate
        self.penalty: float = config.penalty
        self.iter_count: int = config.iter_count
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        _, n_features = X_train.shape

        w = np.zeros(n_features)
        b = 0.0

        y = np.where(Y_train <= 0, -1, 1)

        for _ in range(self.iter_count):
            for idx, x_i in enumerate(X_train):
                condition = y[idx] * (np.dot(x_i, w) + b) >= 1

                if condition:
                    w -= self.learning_rate * (2 * (1 / self.iter_count) * w)
                else:
                    w -= self.learning_rate * (
                        2 * (1 / self.iter_count) * w - self.penalty * y[idx] * x_i
                    )
                    b -= self.learning_rate * (-self.penalty * y[idx])

        self.w = w
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.sign(self._decision_function(X))
        return np.where(pred <= 0, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        decision = self._decision_function(X)
        proba_class_1 = 1 / (1 + np.exp(-decision))
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")

        return np.dot(X, self.w) + self.b
