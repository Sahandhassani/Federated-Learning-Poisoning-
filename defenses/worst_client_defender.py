# defenses/worst_client_defender.py
class WorstClientDefender:
    def __init__(self, patience=2):
        self.patience = patience
        self.worst_counter = {}
        self.quarantined = set()

    def observe(self, round_idx, fed_client_accs, p2p_client_accs, logger):
        suspects = []

        fed_worst = min(fed_client_accs, key=fed_client_accs.get)
        p2p_worst = min(p2p_client_accs, key=p2p_client_accs.get)

        for cid in fed_client_accs:
            if cid not in self.worst_counter:
                self.worst_counter[cid] = 0

            if cid == fed_worst and cid == p2p_worst:
                self.worst_counter[cid] += 1
            else:
                self.worst_counter[cid] = 0

            logger.log_defender(
                round_idx,
                cid,
                fed_client_accs[cid],
                p2p_client_accs[cid],
                self.worst_counter[cid],
                cid in self.quarantined,
            )

            if self.worst_counter[cid] >= self.patience:
                suspects.append(cid)

        return suspects


    def quarantine(self, clients, suspects):
        for c in clients:
            if c.id in suspects and c.id not in self.quarantined:
                print(f"[DEFENDER] Quarantining client {c.id}")
                c.active = False
                self.quarantined.add(c.id)
