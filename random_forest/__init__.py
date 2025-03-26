# random_forest/__init__.py

"""
RandomForest - A simple implementation of Random Forest in Python.
"""

from .forest import RandomForest
from .tree import DecisionTree
from .utils import train_test_split
from .metrics import accuracy

__version__ = "0.1"

__all__ = [
    "RandomForest",
    "DecisionTree",
    "train_test_split",
    "accuracy"
]