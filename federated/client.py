import torch
import torch.nn as nn

class Client:
    def __init__(
        self,
        client_id,
        dataloader,
        model_fn,
        device,
        lr,
        optimizer_name="adam"
    ):
        self.id = client_id
        self.device = device
        self.loader = dataloader
        self.active = True
        self.model = model_fn().to(device)
        self.criterion = nn.CrossEntropyLoss()

        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train(self, theta_global, epochs=1):
        self.model.load_state_dict(theta_global)
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = correct / max(1, total)

        return self.model.state_dict(), avg_loss, avg_acc
