# federated/aggregation.py
import numpy as np

class Aggregator:
    def __init__(self, method="fedavg"):
        self.method = method

    def aggregate(self, thetas, weights=None):
        thetas = np.array(thetas)  # (K, ...)

        K = thetas.shape[0]

        if weights is None:
            weights = np.ones(K) / K

        weights = np.array(weights)

        # reshape weights to (K, 1, 1, ..., 1)
        reshape_dims = (K,) + (1,) * (thetas.ndim - 1)
        weights = weights.reshape(reshape_dims)

        theta_global = np.sum(weights * thetas, axis=0)
        return theta_global
