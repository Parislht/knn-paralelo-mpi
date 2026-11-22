import time
from collections import Counter

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def predecirEnunciado(k=3):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    start_time = time.time()
    y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]
    accuracy = np.mean(np.array(y_pred) == y_test)
    end_time = time.time()
    return np.array(y_pred), accuracy, end_time - start_time


if __name__ == "__main__":
    prediccionesEnunciado, accuracyEnunciado, tiempoEnunciado = predecirEnunciado()
    print(f"Accuracy: {accuracyEnunciado:.4f}")
    print(f"Execution time (sequential): {tiempoEnunciado:.4f} sec")
