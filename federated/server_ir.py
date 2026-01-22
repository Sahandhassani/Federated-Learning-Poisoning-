class Server:
    def __init__(self, clients, aggregator, personalizer, participation_rate=1.0):
        self.clients = clients
        self.aggregator = aggregator
        self.personalizer = personalizer
        self.participation_rate = participation_rate
        self.theta_global = None

    def federated_round(self, local_epochs=1):
        updates = []

        for client in self.clients:
            theta_local = client.train(self.theta_global, epochs=local_epochs)
            theta_personal = self.personalizer.apply(self.theta_global, theta_local)
            updates.append(theta_personal)

        self.theta_global = self.aggregator.aggregate(updates)
