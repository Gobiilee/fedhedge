import flwr as fl

class HedgingFlowerClient(fl.client.NumPyClient):
    """
    Unified Core FL Client. Doesn't care if it's running LSTM, RL, or Transformer.
    """
    def __init__(self, client_id: str, agent, train_loader, val_loader):
        self.client_id = client_id
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return self.agent.get_weights()

    def set_parameters(self, parameters):
        self.agent.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.agent.train_epoch(self.train_loader)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"metric": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_metric = self.agent.evaluate(self.val_loader)
        return float(val_metric), len(self.val_loader.dataset), {"metric": val_metric}