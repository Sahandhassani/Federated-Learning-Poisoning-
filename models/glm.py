import numpy as np

class LinearRegressionGLM:
    def initialize(self, dim):
        return np.zeros(dim)

    def prediction(self, X, theta):
        return X @ theta

    def loss(self, X, y, theta):
        preds = self.prediction(X, theta)
        return np.mean((preds - y) ** 2)

    def gradient(self, X, y, theta):
        preds = self.prediction(X, theta)
        return (2 / len(y)) * X.T @ (preds - y)
