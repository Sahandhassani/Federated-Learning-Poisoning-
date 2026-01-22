# utils/logger.py
import os
import csv
import json

class ExperimentLogger:
    def __init__(self, exp_dir, config):
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # ------------------------
    # internal writer
    # ------------------------
    def _write(self, fname, header, row):
        path = os.path.join(self.exp_dir, fname)
        write_header = not os.path.exists(path)

        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
            
    def _write_row(self, fname, header, row):
        path = os.path.join(self.exp_dir, fname)
        write_header = not os.path.exists(path)

        with open(path, "a") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            f.write(",".join(map(str, row)) + "\n")


    # ==================================================
    # CENTRALIZED
    # ==================================================
    def log_centralized(self, step, train_loss, train_acc, test_loss, test_acc):
        self._write(
            "centralized.csv",
            ["step", "train_loss", "train_acc", "test_loss", "test_acc"],
            [step, train_loss, train_acc, test_loss, test_acc],
        )

    # ==================================================
    # FEDERATED (PER ROUND)
    # ==================================================
    def log_fed_round(
        self,
        round_idx,
        client_stats,   # dict: cid -> (acc, loss)
        mean_acc,
        mean_loss,
        test_acc,
        test_loss,
    ):
        header = ["round"]
        row = [round_idx]

        for cid in sorted(client_stats):
            header += [f"client_{cid}_acc", f"client_{cid}_loss"]
            row += [client_stats[cid][0], client_stats[cid][1]]

        header += ["mean_acc", "mean_loss", "test_acc", "test_loss"]
        row += [mean_acc, mean_loss, test_acc, test_loss]

        self._write("fed_rounds.csv", header, row)

    # ==================================================
    # P2P (PER ROUND)
    # ==================================================
    def log_p2p_round(
        self,
        round_idx,
        client_stats,
        mean_acc,
        mean_loss,
        test_acc,
        test_loss,
    ):
        header = ["round"]
        row = [round_idx]

        for cid in sorted(client_stats):
            header += [f"client_{cid}_acc", f"client_{cid}_loss"]
            row += [client_stats[cid][0], client_stats[cid][1]]

        header += ["mean_acc", "mean_loss", "test_acc", "test_loss"]
        row += [mean_acc, mean_loss, test_acc, test_loss]

        self._write("p2p_rounds.csv", header, row)
        
    def log_defender(
        self,
        round_idx,
        client_id,
        fed_acc,
        p2p_acc,
        bad_count,
        quarantined,
    ):
        """
        Logs defender decisions per round and per client
        """
        self._write_row(
            "defender.csv",
            [
                "round",
                "client_id",
                "fed_acc",
                "p2p_acc",
                "bad_count",
                "quarantined",
            ],
            [
                round_idx,
                client_id,
                fed_acc,
                p2p_acc,
                bad_count,
                int(quarantined),
            ],
        )
