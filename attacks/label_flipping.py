# attacks/label_flipping.py
import numpy as np

class LabelFlippingAttack:
    """
    Label flipping attack for classification.
    """

    def __init__(
        self,
        flip_fraction: float,
        num_classes: int,
        target_class: int = None,
        seed: int = 42
    ):
        """
        Args:
            flip_fraction (float): fraction of samples to flip (0–1)
            num_classes (int): number of classes
            target_class (int): if set, flip all labels to this class
            seed (int): random seed
        """
        assert 0.0 <= flip_fraction <= 1.0
        self.flip_fraction = flip_fraction
        self.num_classes = num_classes
        self.target_class = target_class
        self.rng = np.random.RandomState(seed)

    def apply(self, y):
        """
        Apply label flipping to a label array.

        Args:
            y (np.ndarray): labels (N,)

        Returns:
            np.ndarray: poisoned labels
        """
        y = y.copy()
        n = len(y)
        n_flip = int(self.flip_fraction * n)

        idx = self.rng.choice(n, n_flip, replace=False)

        if self.target_class is not None:
            # Targeted attack
            y[idx] = self.target_class
        else:
            # Untargeted random flipping
            for i in idx:
                old = y[i]
                new = self.rng.choice(
                    [c for c in range(self.num_classes) if c != old]
                )
                y[i] = new

        return y
