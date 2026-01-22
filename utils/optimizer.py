
import torch

def make_optimizer(model, lr, optimizer_name):
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer")
