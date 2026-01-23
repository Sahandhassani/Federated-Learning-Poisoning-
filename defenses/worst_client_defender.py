import torch
from torch.utils.data import DataLoader

# ==================================================
# DEFENDER (CORRECTED – WILL PRODUCE RESULTS)
# ==================================================
class WorstClientDefender:
    def __init__(self, patience=3, acc_threshold=0.05):
        self.patience = patience
        self.acc_threshold = acc_threshold
        self.worst_counter = {}
        self.quarantined = set()

    def observe(self, round_idx, fed_client_accs, p2p_client_accs):
        suspects = []

        # means
        fed_mean = sum(fed_client_accs.values()) / len(fed_client_accs)
        p2p_mean = sum(p2p_client_accs.values()) / len(p2p_client_accs)

        # worst clients
        fed_worst, fed_worst_acc = min(fed_client_accs.items(), key=lambda x: x[1])
        p2p_worst, p2p_worst_acc = min(p2p_client_accs.items(), key=lambda x: x[1])

        fed_gap = fed_mean - fed_worst_acc
        p2p_gap = p2p_mean - p2p_worst_acc

        for cid in fed_client_accs:
            if cid not in self.worst_counter:
                self.worst_counter[cid] = 0

            if (
                cid == fed_worst and
                cid == p2p_worst and
                fed_gap >= self.acc_threshold and
                p2p_gap >= self.acc_threshold
            ):
                self.worst_counter[cid] += 1
            else:
                self.worst_counter[cid] = max(0, self.worst_counter[cid] - 1)

            if self.worst_counter[cid] >= self.patience:
                suspects.append(cid)

        print(
            f"[DEFENDER] Round {round_idx} | "
            f"Fed mean={fed_mean:.4f}, worst={fed_worst}({fed_worst_acc:.4f}), gap={fed_gap:.4f} | "
            f"P2P mean={p2p_mean:.4f}, worst={p2p_worst}({p2p_worst_acc:.4f}), gap={p2p_gap:.4f} | "
            f"Counters={self.worst_counter}"
        )

        return suspects

    def quarantine(self, clients, suspects):
        for c in clients:
            if c.id in suspects and c.id not in self.quarantined:
                print(f"[DEFENDER] Quarantining client {c.id}")
                c.active = False
                self.quarantined.add(c.id)
