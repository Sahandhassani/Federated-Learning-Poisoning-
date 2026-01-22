import torch

class FedAvg:
    def aggregate(self, updates):
        avg = {}
        for k in updates[0]:
            avg[k] = torch.stack([u[k].float() for u in updates]).mean(0)
        return avg
