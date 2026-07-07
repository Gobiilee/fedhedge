import torch
import numpy as np
from collections import OrderedDict
from agents.base_agent import BaseAgent
from models.rl_hedging_net import RLHedgeActor, RLHedgeCritic

class RLAgent(BaseAgent):
    def __init__(self, input_dim=4, action_dim=1, hidden_dim=64, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize 2 neural networks
        self.actor = RLHedgeActor(input_dim, hidden_dim).to(self.device)
        self.critic = RLHedgeCritic(input_dim, action_dim, hidden_dim).to(self.device)
        
        #2. Two independent optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.model = self.actor
        self.mse_loss = torch.nn.MSELoss()

    def get_weights(self) -> list[np.ndarray]:
        # RL FL WEIGHT: Extract only the weights of the Actor network (Transaction Strategy) to send to the Server.
        return [val.cpu().numpy() for _, val in self.actor.state_dict().items()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        # Only load aggregate weights from the Server into the local Actor network.
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.actor.state_dict().keys(), weights)}
        )
        self.actor.load_state_dict(state_dict, strict=True)

    def _compute_reward(self, raw_returns, deltas):
        """ 
        Need to improve 
        Reward definition: Heavy penalty if the variance of the Hedged portfolio is high. 
        Reward = - (raw_return - delta * raw_return)^2 
        """
        hedged_returns = raw_returns - (deltas * raw_returns)
        risk_penalty = torch.where(
            hedged_returns < 0, 
            (hedged_returns ** 2) * 5.0, 
            (hedged_returns ** 2) * 1.0  
        )
        funding_cost = deltas * 0.0005
        reward = -(risk_penalty + funding_cost)
        
        return reward

    def train_epoch(self, dataloader) -> float:
        self.actor.train()
        self.critic.train()
        total_actor_loss = 0.0
        
        for features, target_returns in dataloader:
            features = features.to(self.device)
            target_returns = target_returns.to(self.device)
            
            # Since the current dataset returns a 3D sequence (batch, seq_length, features), 
            # RL Actor often uses last step state:
            current_state = features[:, -1, :] 

            self.critic_optimizer.zero_grad()
            with torch.no_grad():
                deltas = self.actor(current_state)

            rewards = self._compute_reward(target_returns, deltas)
            q_values = self.critic(current_state, deltas)
            
            critic_loss = self.mse_loss(q_values, rewards)
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_deltas = self.actor(current_state)

            actor_loss = -self.critic(current_state, actor_deltas).mean()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            
        return total_actor_loss / len(dataloader)

    def evaluate(self, dataloader) -> float:
        self.actor.eval()
        total_reward = 0.0
        
        with torch.no_grad():
            for features, target_returns in dataloader:
                features = features.to(self.device)
                target_returns = target_returns.to(self.device)
                
                current_state = features[:, -1, :]
                deltas = self.actor(current_state)
                rewards = self._compute_reward(target_returns, deltas)
                
                total_reward += rewards.mean().item()
                
        return total_reward / len(dataloader)