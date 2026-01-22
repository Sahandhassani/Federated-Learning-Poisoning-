import torch
from torch.utils.data import DataLoader

from attacks.label_flipping import LabelFlippingAttack

from utils.CIFAR.data_utils import load_CIFAR10
from utils.dataset import CIFARDataset
from utils.split import split_clients_only
from utils.eval import evaluate, evaluate_avg_clients, evaluate_clients , evaluate_with_loss
from utils.subsample import subsample_dataset
from utils.optimizer import make_optimizer
from utils.logger import ExperimentLogger
from defenses.worst_client_defender import WorstClientDefender
from utils.config import (
    BATCH_SIZE,
    LR,
    LOCAL_EPOCHS,
    ROUNDS,
    OPTIMIZER,
    ATTACK_CLIENT_ID,         
    FLIP_FRACTION,           
    NUM_CLASSES,
    subsample_fraction,
)
from utils.plots import (
    plot_centralized,
    plot_fed_global,
    plot_fed_clients,
    plot_p2p_global,
    plot_p2p_clients
)

from models.alexnet import AlexNetCIFAR
from models.cifar_cnn import CIFAR10_Model
from models.cifar_cnn2 import CIFAR_CustomCNN

from federated.client import Client
from federated.server import Server
from federated.aggregation import FedAvg
from federated.p2p import P2PAggregator, P2PTrainer

def main():
    
    attack = LabelFlippingAttack(
        flip_fraction=0.4,
        num_classes=NUM_CLASSES
    )
    
    defender = WorstClientDefender(patience=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================================================
    # EXPERIMENT SETUP
    # ==================================================
    EXP_DIR = "experiments/exp_040"
    config = {
        "dataset": "CIFAR-10",
        "model": "CIFAR10_Model",
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "local_epochs": LOCAL_EPOCHS,
        "rounds": ROUNDS,
        "optimizer": OPTIMIZER,
        "subsample_fraction": subsample_fraction,
        "local_ratio": 0.1,
        "client_percentages": [20, 10, 25, 25, 20],
    }

    logger = ExperimentLogger(EXP_DIR, config)
    model = CIFAR10_Model

    # ==================================================
    # LOAD CIFAR-10
    # ==================================================
    X_train, y_train, X_test, y_test = load_CIFAR10(
        "data/CIFAR/datasets/cifar-10-batches-py"
    )

    print("Original data:")
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ==================================================
    # SUBSAMPLE (DEBUG / FAST EXPERIMENTS)
    # ==================================================
    X_train, y_train = subsample_dataset(
        X_train, y_train, fraction=1, balanced=True
    )
    X_test, y_test = subsample_dataset(
        X_test, y_test, fraction=1, balanced=True
    )

    print("\nAfter subsampling:")
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ==================================================
    # SPLIT LOCAL VS FEDERATED DATA
    # ==================================================
    client_data = split_clients_only(
        X_train,
        y_train,
        client_percentages=[20, 10, 25, 25, 20]
    )

    # ==================================================
    # CENTRALIZED (LOCAL) BASELINE
    # ==================================================
       
    print("\nLocal (centralized) training...")

    local_dataset = CIFARDataset(X_train, y_train)
    local_loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        CIFARDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    local_model = model().to(device)
    local_optimizer = make_optimizer(local_model, LR, OPTIMIZER)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Match training effort with FL
    CENTRAL_EPOCHS = LOCAL_EPOCHS * ROUNDS

    for epoch in range(CENTRAL_EPOCHS):

        local_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for x, y in local_loader:
            x, y = x.to(device), y.to(device)

            local_optimizer.zero_grad()
            logits = local_model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            local_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = epoch_loss / max(1, n_batches)
        train_acc = correct / max(1, total)

        # ------------------------
        # Test evaluation
        # ------------------------
        test_loss, test_acc = evaluate_with_loss(
            local_model,
            test_loader,
            loss_fn,
            device
        )

        step = epoch
        logger.log_centralized(step, train_loss, train_acc, test_loss, test_acc)


    local_acc = evaluate(local_model, test_loader, device)
    print("Local training done.")
    
    # ==================================================
    # FEDERATED + P2P TRAINING (SINGLE CELL)
    # ==================================================
    print("\nFederated training...")

    # ------------------------
    # Build FedAvg system
    # ------------------------
    fed_clients = []
    for i, (Xi, yi) in enumerate(client_data):
        yi_fed = attack.apply(yi) if i == ATTACK_CLIENT_ID else yi
        fed_clients.append(
            Client(
                client_id=i,
                dataloader=DataLoader(
                    CIFARDataset(Xi, yi_fed),
                    batch_size=BATCH_SIZE,
                    shuffle=True
                ),
                model_fn=model,
                device=device,
                lr=LR,
                optimizer_name=OPTIMIZER
            )
        )

    server = Server(fed_clients, FedAvg(), model, device)
    fed_model = model().to(device)

    # ------------------------
    # Build P2P system
    # ------------------------
    p2p_clients = []
    for i, (Xi, yi) in enumerate(client_data):
        yi_p2p = attack.apply(yi) if i == ATTACK_CLIENT_ID else yi
        p2p_clients.append(
            Client(
                client_id=i,
                dataloader=DataLoader(
                    CIFARDataset(Xi, yi_p2p),
                    batch_size=BATCH_SIZE,
                    shuffle=True
                ),
                model_fn=model,
                device=device,
                lr=LR,
                optimizer_name=OPTIMIZER
            )
        )

    p2p_trainer = P2PTrainer(
        clients=p2p_clients,
        aggregator=P2PAggregator(num_peers=2, self_weight=0.5)
    )

    print("\nSynchronous FedAvg + P2P training...")

    for r in range(ROUNDS):
        print(f"\n=== Round {r} ===")

        # ==================================================
        # FedAvg round
        # ==================================================
        fed_updates = []
        fed_client_stats = {}

        for client in server.clients:
            if not client.active:
                continue

            state, loss, acc = client.train(
                server.theta_global,
                epochs=LOCAL_EPOCHS
            )
            fed_updates.append(state)
            fed_client_stats[client.id] = (acc, loss)

        server.theta_global = server.aggregator.aggregate(fed_updates)
        fed_model.load_state_dict(server.theta_global)

        fed_test_loss, fed_test_acc = evaluate_with_loss(
            fed_model,
            test_loader,
            loss_fn,
            device
        )

        fed_mean_acc = sum(a for a, _ in fed_client_stats.values()) / len(fed_client_stats)
        fed_mean_loss = sum(l for _, l in fed_client_stats.values()) / len(fed_client_stats)

        logger.log_fed_round(
            r,
            fed_client_stats,
            fed_mean_acc,
            fed_mean_loss,
            fed_test_acc,
            fed_test_loss
        )

        # ==================================================
        # P2P round
        # ==================================================
        p2p_client_stats = {}
        client_states = {}

        for client in p2p_clients:
            if not client.active:
                continue

            state, loss, acc = client.train(
                client.model.state_dict(),
                epochs=LOCAL_EPOCHS
            )
            client_states[client.id] = state
            p2p_client_stats[client.id] = (acc, loss)

        new_states = p2p_trainer.aggregator.aggregate(client_states)
        for client in p2p_clients:
            if client.id in new_states:
                client.model.load_state_dict(new_states[client.id])

        p2p_test_acc = evaluate_avg_clients(
            p2p_clients,
            model,
            test_loader,
            device
        )
        p2p_test_loss = None

        p2p_mean_acc = sum(a for a, _ in p2p_client_stats.values()) / len(p2p_client_stats)
        p2p_mean_loss = sum(l for _, l in p2p_client_stats.values()) / len(p2p_client_stats)

        logger.log_p2p_round(
            r,
            p2p_client_stats,
            p2p_mean_acc,
            p2p_mean_loss,
            p2p_test_acc,
            p2p_test_loss
        )

        # ==================================================
        # Defender
        # ==================================================
        fed_client_accs = evaluate_clients(
            server.clients, model, test_loader, device
        )
        p2p_client_accs = evaluate_clients(
            p2p_clients, model, test_loader, device
        )

        suspects = defender.observe(
            round_idx=r,
            fed_client_accs=fed_client_accs,
            p2p_client_accs=p2p_client_accs,
            logger=logger
        )

        defender.quarantine(server.clients, suspects)
        defender.quarantine(p2p_clients, suspects)

        # ==================================================
        # Monitor
        # ==================================================
        print(
            f"Round {r}: "
            f"FedAvg Test Acc={fed_test_acc:.4f}, "
            f"P2P Test Acc={p2p_test_acc:.4f}, "
            f"Quarantined={defender.quarantined}"
        )
    
    print("FL training done.") 
    # ==================================================
    # RESULTS
    # ==================================================
    print("\n===== RESULTS =====")
    print(f"Local model accuracy     : {local_acc:.4f}")
    print(f"Federated model accuracy : {fed_test_acc:.4f}")

    # ==================================================
    # PLOTTING (AFTER TRAINING ONLY)
    # ==================================================

    print("\nGenerating plots...")
    plot_centralized(EXP_DIR)
    plot_fed_global(EXP_DIR)
    plot_fed_clients(EXP_DIR)
    plot_p2p_global(EXP_DIR)
    plot_p2p_clients(EXP_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
