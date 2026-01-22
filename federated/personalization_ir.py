class Personalizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def apply(self, global_theta, local_theta):
        """
        Blend local and global models.
        alpha = 1.0 -> fully local
        alpha = 0.0 -> fully global
        """
        return self.alpha * local_theta + (1 - self.alpha) * global_theta
