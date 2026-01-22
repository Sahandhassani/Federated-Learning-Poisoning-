import random
import copy
import torch

class P2PAggregator:
    def __init__(self, num_peers=2, self_weight=0.5):
        self.num_peers = num_peers
        self.self_weight = self_weight

    def aggregate(self, client_states):
        import torch
        client_ids = list(client_states.keys())
        new_states = {}

        for cid in client_ids:
            neighbors = [
                i for i in client_ids if i != cid
            ]

            if len(neighbors) > self.num_peers:
                neighbors = neighbors[:self.num_peers]

            # Prepare new state
            agg_state = {}

            for k, v in client_states[cid].items():

                if not torch.is_floating_point(v):
                    # Copy integer buffers directly
                    agg_state[k] = v.clone()
                    continue

                # Floating-point parameter → average
                agg_state[k] = self.self_weight * v.clone()

                for nid in neighbors:
                    agg_state[k] += (
                        (1.0 - self.self_weight) / len(neighbors)
                    ) * client_states[nid][k]

            new_states[cid] = agg_state

        return new_states

class P2PTrainer:
    def __init__(self, clients, aggregator):
        self.clients = clients
        self.aggregator = aggregator

    def p2p_round(self, local_epochs=1, logger=None, round_idx=None):
        # Step 1: local training
        client_states = {}
        for c in self.clients:
            theta, loss = c.train(
                c.model.state_dict(),
                epochs=local_epochs
            )
            client_states[c.id] = theta

            if logger is not None:
                logger.log_client(round_idx, c.id, loss)

        # Step 2: peer aggregation
        new_states = self.aggregator.aggregate(client_states)

        # Step 3: update clients
        for c in self.clients:
            c.model.load_state_dict(new_states[c.id])
    
