import numpy as np

from federated.client import Client
from federated.server import Server
from federated.aggregation import Aggregator
from federated.personalization import Personalizer
from models.default_model import LogisticRegressionModel
from data.load_iris import load_iris_splits


def train_local(X_train, y_train, X_val, y_val, model, lr=0.1, epochs=200):
    theta = model.initialize(X_train.shape[1], len(set(y_train)))

    for _ in range(epochs):
        grad = model.gradient(X_train, y_train, theta)
        theta -= lr * grad

    return {
        "val_acc": model.accuracy(X_val, y_val, theta),
        "theta": theta
    }


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_iris_splits()

    model = LogisticRegressionModel()

    # =====================
    # Local Training
    # =====================
    local_result = train_local(
        X_train, y_train, X_val, y_val, model
    )

    # =====================
    # Federated Training
    # =====================
    clients = []
    n_clients = 6

    splits = np.array_split(range(len(X_train)), n_clients)

    for i, idx in enumerate(splits):
        clients.append(
            Client(
                client_id=i,
                X=X_train[idx],
                y=y_train[idx],
                model=model,
                lr=0.1
            )
        )

    server = Server(
        clients=clients,
        aggregator=Aggregator(),
        personalizer=Personalizer(alpha=0.6),
        participation_rate=1.0
    )
    
    server.theta_global = model.initialize(X_train.shape[1], 3)
    
    for _ in range(50):
        server.federated_round(local_epochs=1)


    theta_fed = server.theta_global

    fed_val_acc = model.accuracy(X_val, y_val, theta_fed)
    fed_test_acc = model.accuracy(X_test, y_test, theta_fed)
    local_test_acc = model.accuracy(X_test, y_test, local_result["theta"])

    # =====================
    # Results
    # =====================
    print("\n===== Iris Results =====")
    print(f"Local   Val Acc : {local_result['val_acc']:.3f}")
    print(f"Local   Test Acc: {local_test_acc:.3f}")
    print(f"Fed     Val Acc : {fed_val_acc:.3f}")
    print(f"Fed     Test Acc: {fed_test_acc:.3f}")


if __name__ == "__main__":
    main()