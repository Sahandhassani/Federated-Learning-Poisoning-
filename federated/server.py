class Server:
    def __init__(self, clients, aggregator, model_fn, device):
        self.clients = clients
        self.aggregator = aggregator
        self.device = device
        self.theta_global = model_fn().to(device).state_dict()

    def federated_round(self, local_epochs=1, logger=None, round_idx=None):
        updates = []

        for client in self.clients:
            # Client trains locally
            theta_local, train_loss = client.train(
                self.theta_global,
                epochs=local_epochs
            )
            updates.append(theta_local)

            # Log client-side loss (optional)
            if logger is not None and round_idx is not None:
                logger.log_client(
                    round_idx=round_idx,
                    client_id=client.id,
                    loss=train_loss
                )

        # Aggregate client models (FedAvg)
        self.theta_global = self.aggregator.aggregate(updates)
