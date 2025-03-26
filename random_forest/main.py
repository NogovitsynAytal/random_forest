from random_forest.forest import RandomForest
from random_forest.utils import train_test_split
from random_forest.metrics import accuracy
import numpy as np


# Генерация данных
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создание и обучение модели
model = RandomForest(n_trees=10, max_depth=5)
model.fit(X_train, y_train)

# Предсказание и оценка качества
y_pred = model.predict(X_test)
print("Accuracy:", accuracy(y_test, y_pred))