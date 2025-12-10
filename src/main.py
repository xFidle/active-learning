from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.forest.cart import CART, CARTConfig, PredictionMode
from src.forest.forest import RandomForest, RandomForestConfig


# Proof of working classifier
def main():
    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 6, 1, 0)

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.2)
    )

    config = CARTConfig(10, 2)
    proba = PredictionMode.PROBABILITIES
    classes = PredictionMode.LABELS

    tree = CART(config)
    tree.fit(x_train, y_train)

    forest_config = RandomForestConfig(100, config)
    forest = RandomForest(forest_config)
    forest.fit(x_train, y_train)

    sklearn_forest = RandomForestClassifier(100, random_state=42)
    sklearn_forest.fit(x_train, y_train)

    forest_proba = forest.predict(x_test, proba)
    sklearn_forest_proba = sklearn_forest.predict_proba(x_test)

    print(forest_proba[:10])
    print(sklearn_forest_proba[:10])

    print("My single tree: ", accuracy_score(y_test, tree.predict(x_test, classes)))
    print("My forest: ", accuracy_score(y_test, forest.predict(x_test, classes)))
    print("SKLEARN forest: ", accuracy_score(y_test, sklearn_forest.predict(x_test)))


if __name__ == "__main__":
    main()
