import numpy as np

class LogisticRegressionModel:
    def initialize(self, d, c):
        return np.zeros((d, c))

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def gradient(self, X, y, theta):
        probs = self.softmax(X @ theta)
        y_onehot = np.eye(theta.shape[1])[y]
        return X.T @ (probs - y_onehot) / len(X)

    def accuracy(self, X, y, theta):
        preds = np.argmax(X @ theta, axis=1)
        return np.mean(preds == y)
