import numpy as np

def split_clients_only(X, y, client_percentages, shuffle=True, seed=42):
    """
    Split the full training dataset across clients using percentages.

    Args:
        X (np.ndarray): training data
        y (np.ndarray): labels
        client_percentages (list): e.g. [20, 10, 25, 25, 20]
        shuffle (bool): shuffle before split
        seed (int): reproducibility

    Returns:
        list of (Xi, yi) tuples for each client
    """

    assert sum(client_percentages) == 100, \
        "client_percentages must sum to 100"

    n = len(X)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    splits = []
    start = 0

    for pct in client_percentages:
        size = int((pct / 100) * n)
        end = start + size

        idx = indices[start:end]
        splits.append((X[idx], y[idx]))

        start = end

    # safety: add leftovers to last client
    if start < n:
        X_last, y_last = splits[-1]
        extra_idx = indices[start:]
        splits[-1] = (
            np.concatenate([X_last, X[extra_idx]]),
            np.concatenate([y_last, y[extra_idx]])
        )

    return splits
