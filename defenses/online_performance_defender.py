class OnlinePerformanceDefender:
    def __init__(
        self,
        drop_threshold=0.02,
        gap_threshold=0.03,
        patience=2
    ):
        self.drop_threshold = drop_threshold
        self.gap_threshold = gap_threshold
        self.patience = patience

        self.prev_fed = None
        self.prev_p2p = None

        self.suspicion = {}  # client_id -> count
        self.quarantined = set()

    def observe(
        self,
        fed_acc,
        p2p_acc,
        client_losses,
        clients
    ):
        """
        client_losses: dict {client_id: loss}
        clients: list of Client objects
        """

        if self.prev_fed is None:
            self.prev_fed = fed_acc
            self.prev_p2p = p2p_acc
            return []

        delta_fed = fed_acc - self.prev_fed
        delta_p2p = p2p_acc - self.prev_p2p
        gap = fed_acc - p2p_acc

        suspects = []

        # global degradation detected
        if (
            delta_fed < -self.drop_threshold and
            delta_p2p < -self.drop_threshold and
            abs(gap) > self.gap_threshold
        ):
            # rank clients by loss (high loss = suspicious)
            ranked = sorted(
                client_losses.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for cid, _ in ranked:
                self.suspicion[cid] = self.suspicion.get(cid, 0) + 1

                if self.suspicion[cid] >= self.patience:
                    suspects.append(cid)

        self.prev_fed = fed_acc
        self.prev_p2p = p2p_acc
        return suspects

    def quarantine(self, clients, suspects):
        for c in clients:
            if c.id in suspects and c.active:
                print(f"[DEFENDER] Client {c.id} quarantined")
                c.active = False
                self.quarantined.add(c.id)