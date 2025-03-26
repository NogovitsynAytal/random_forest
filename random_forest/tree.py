from typing import List, Tuple
import numpy as np


class DecisionTreeNode:
    """
    Узел дерева решений.
    """
    def __init__(self, feature_index: int = None, threshold: float = None,
                 left=None, right=None, label: int = None):
        self.feature_index = feature_index  # Индекс признака для разделения
        self.threshold = threshold  # Пороговое значение
        self.left = left  # Левый потомок
        self.right = right  # Правый потомок
        self.label = label  # Метка класса (если это лист)


class DecisionTree:
    """
    Дерево решений для задачи классификации.
    """
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение дерева решений.
        :param X: Признаки (numpy array).
        :param y: Метки классов (numpy array).
        """
        self.root = self._build_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание меток для новых данных.
        :param X: Признаки (numpy array).
        :return: Массив предсказанных меток.
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Рекурсивное построение дерева.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Критерии остановки
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_label = self._most_common_label(y)
            return DecisionTreeNode(label=leaf_label)

        # Выбор наилучшего разделения
        feature_index, threshold = self._find_best_split(X, y, n_features)
        if feature_index is None:
            leaf_label = self._most_common_label(y)
            return DecisionTreeNode(label=leaf_label)

        # Разделение данных
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionTreeNode(feature_index, threshold, left, right)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[int, float]:
        """
        Поиск наилучшего разбиения.
        """
        best_gini = float('inf')
        split_index, split_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._calculate_gini(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _calculate_gini(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float) -> float:
        """
        Расчет коэффициента Джини.
        """
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        n = len(y)
        n_left, n_right = sum(left_indices), sum(right_indices)

        if n_left == 0 or n_right == 0:
            return float('inf')

        gini_left = 1 - sum((np.sum(y[left_indices] == c) / n_left) ** 2 for c in np.unique(y))
        gini_right = 1 - sum((np.sum(y[right_indices] == c) / n_right) ** 2 for c in np.unique(y))

        return (n_left / n) * gini_left + (n_right / n) * gini_right

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Наиболее часто встречающаяся метка.
        """
        return np.bincount(y).argmax()

    def _predict_sample(self, sample: np.ndarray, node: DecisionTreeNode) -> int:
        """
        Рекурсивное предсказание для одного образца.
        """
        if node.label is not None:
            return node.label
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)