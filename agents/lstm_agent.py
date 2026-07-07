import torch
import numpy as np
from collections import OrderedDict
from agents.base_agent import BaseAgent
from models.deep_hedging_net import DeepHedgingLSTM
from models.loss_functions import HedgingVarianceLoss

class LSTMAgent(BaseAgent):
    def __init__(self, input_dim=4, hidden_dim=32, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepHedgingLSTM(input_dim, hidden_dim).to(self.device)
        self.criterion = HedgingVarianceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_weights(self) -> list[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for features, target_returns in dataloader:
            features = features.to(self.device)
            target_returns = target_returns.to(self.device)
            
            self.optimizer.zero_grad()
            deltas = self.model(features)
            loss = self.criterion(deltas, target_returns)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, target_returns in dataloader:
                features = features.to(self.device)
                target_returns = target_returns.to(self.device)
                deltas = self.model(features)
                loss = self.criterion(deltas, target_returns)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)