class Client:
    def __init__(self, client_id, X, y, model, lr=0.1):
        self.id = client_id
        self.X = X
        self.y = y
        self.model = model
        self.lr = lr

    def train(self, theta, epochs=1):
        theta_local = theta.copy()
        for _ in range(epochs):
            grad = self.model.gradient(self.X, self.y, theta_local)
            theta_local -= self.lr * grad
        return theta_local
