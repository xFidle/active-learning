import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from src.models.classifier import Classifier


def calculate_pr_metrics(
    classifier: Classifier, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    proba = classifier.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    auc_score = auc(recall, precision)

    return precision, recall, float(auc_score)
