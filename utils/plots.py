# utils/plots.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def _safe_read(path):
    if not os.path.exists(path):
        print(f"[PLOT WARNING] Missing {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[PLOT WARNING] Failed to read {path}: {e}")
        return None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ==================================================
# CENTRALIZED
# ==================================================
def plot_centralized(exp_dir):
    df = _safe_read(os.path.join(exp_dir, "centralized.csv"))
    if df is None:
        return

    out_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(out_dir)

    plt.figure()
    if "train_acc" in df.columns:
        plt.plot(df["step"], df["train_acc"], label="Train Acc")
    if "test_acc" in df.columns:
        plt.plot(df["step"], df["test_acc"], label="Test Acc")

    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title("Centralized Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "centralized.png"))
    plt.close()


# ==================================================
# FEDERATED (FedAvg)
# ==================================================
def plot_fed_global(exp_dir):
    df = _safe_read(os.path.join(exp_dir, "fed_rounds.csv"))
    if df is None:
        return

    out_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(out_dir)

    plt.figure()
    if "mean_acc" in df.columns:
        plt.plot(df["round"], df["mean_acc"], label="Mean Train Acc")
    if "test_acc" in df.columns:
        plt.plot(df["round"], df["test_acc"], label="Test Acc")

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg Global Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_global.png"))
    plt.close()


def plot_fed_clients(exp_dir):
    df = _safe_read(os.path.join(exp_dir, "fed_rounds.csv"))
    if df is None:
        return

    out_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(out_dir)

    plt.figure()
    for col in df.columns:
        if col.startswith("client_") and col.endswith("_acc"):
            plt.plot(df["round"], df[col], label=col)

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg Client Accuracies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_clients.png"))
    plt.close()


# ==================================================
# P2P
# ==================================================
def plot_p2p_global(exp_dir):
    df = _safe_read(os.path.join(exp_dir, "p2p_rounds.csv"))
    if df is None:
        return

    out_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(out_dir)

    plt.figure()
    if "mean_acc" in df.columns:
        plt.plot(df["round"], df["mean_acc"], label="Mean Train Acc")
    if "test_acc" in df.columns:
        plt.plot(df["round"], df["test_acc"], label="Test Acc")

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("P2P Global Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p2p_global.png"))
    plt.close()


def plot_p2p_clients(exp_dir):
    df = _safe_read(os.path.join(exp_dir, "p2p_rounds.csv"))
    if df is None:
        return

    out_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(out_dir)

    plt.figure()
    for col in df.columns:
        if col.startswith("client_") and col.endswith("_acc"):
            plt.plot(df["round"], df[col], label=col)

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("P2P Client Accuracies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p2p_clients.png"))
    plt.close()
