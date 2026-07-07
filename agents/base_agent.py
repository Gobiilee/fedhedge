from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    """
    Abstract Base Class for all Hedging Agents (LSTM, RL, etc.).
    Ensures a unified interface for the Flower FL Core.
    """
    
    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        """Extract the model's weights into a list of NumPy arrays to send to the Server."""
        pass

    @abstractmethod
    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Load the weights (aggregated from the Server) back into the local model."""
        pass

    @abstractmethod
    def train_epoch(self, dataloader) -> float:
        """Train the model for 1 epoch (round). Returns the metric (Loss or Reward)."""
        pass

    @abstractmethod
    def evaluate(self, dataloader) -> float:
        """Evaluates the model on the test set. Returns the metric (Loss or Reward)."""
        pass