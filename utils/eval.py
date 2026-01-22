import torch

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

import torch

def evaluate_avg_clients(clients, model_fn, test_loader, device, criterion=None):
    """
    Evaluate averaged model across ACTIVE clients.
    Returns (loss, accuracy) if criterion is given, else accuracy only.
    """

    active_clients = [c for c in clients if c.active]
    if len(active_clients) == 0:
        return (None, None) if criterion else None

    avg_state = {}
    ref_state = active_clients[0].model.state_dict()

    for k, v in ref_state.items():
        avg_state[k] = torch.zeros_like(v) if torch.is_floating_point(v) else v.clone()

    for c in active_clients:
        state = c.model.state_dict()
        for k in avg_state:
            if torch.is_floating_point(state[k]):
                avg_state[k] += state[k]

    for k in avg_state:
        if torch.is_floating_point(avg_state[k]):
            avg_state[k] /= len(active_clients)

    model = model_fn().to(device)
    model.load_state_dict(avg_state)

    if criterion is None:
        acc = evaluate(model, test_loader, device)
        return acc

    loss, acc = evaluate_with_loss(model, test_loader, criterion, device)
    return loss, acc

def evaluate_clients(clients, model_fn, test_loader, device):
    """
    Returns dict {client_id: accuracy}
    """
    accs = {}

    for c in clients:
        if not c.active:
            continue

        model = model_fn().to(device)
        model.load_state_dict(c.model.state_dict())

        acc = evaluate(model, test_loader, device)
        accs[c.id] = acc

    return accs

def evaluate_with_loss(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return (
        loss_sum / max(1, n_batches),
        correct / max(1, total)
    )


