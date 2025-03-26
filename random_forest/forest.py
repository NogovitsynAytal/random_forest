import numpy as np
from typing import List, Tuple
from .tree import DecisionTree


class RandomForest:
    """
    Случайный лес для задачи классификации.
    """
    def __init__(self, n_trees: int = 10, max_depth: int = None,
                 min_samples_split: int = 2, n_features: int = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение случайного леса.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание меток для новых данных.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(row).argmax() for row in predictions.T])

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация бутстрап-выборки.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]