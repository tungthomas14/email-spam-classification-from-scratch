import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) == 0:
            raise ValueError("X must not be empty")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            probabilities = self.predict_proba(X)
            errors = probabilities - y

            dw = (X.T @ errors) / n_samples
            db = np.mean(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._binary_cross_entropy(y, probabilities)
            self.loss_history.append(loss)

        return self

    def predict_proba_one(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        score = vector @ self.weights + self.bias
        return self._sigmoid(score)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        scores = X @ self.weights + self.bias
        return self._sigmoid(scores)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
