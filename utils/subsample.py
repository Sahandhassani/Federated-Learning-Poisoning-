import numpy as np

def subsample_dataset(
    X,
    y,
    fraction=None,
    n_samples=None,
    balanced=True,
    seed=42
):
    """
    Subsample a dataset.

    Args:
        X (np.ndarray): data array
        y (np.ndarray): labels
        fraction (float): fraction of data to keep (0 < fraction <= 1)
        n_samples (int): absolute number of samples to keep
        balanced (bool): class-balanced subsampling
        seed (int): random seed

    Returns:
        X_sub, y_sub
    """
    np.random.seed(seed)

    N = len(X)

    if fraction is None and n_samples is None:
        raise ValueError("Specify either fraction or n_samples")

    if fraction is not None:
        n_samples = int(fraction * N)

    if not balanced:
        idx = np.random.choice(N, n_samples, replace=False)
        return X[idx], y[idx]

    # -------- class-balanced subsampling --------
    classes = np.unique(y)
    per_class = n_samples // len(classes)

    idx_all = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = np.random.choice(
            idx_c,
            per_class,
            replace=len(idx_c) < per_class
        )
        idx_all.append(idx_c)

    idx_all = np.concatenate(idx_all)
    np.random.shuffle(idx_all)

    return X[idx_all], y[idx_all]
