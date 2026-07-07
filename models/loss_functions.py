import torch
import torch.nn as nn

class HedgingVarianceLoss(nn.Module):
    """
    Custom financial loss function. 
    Aims to minimize the variance of the hedged portfolio returns 
    while penalizing aggressive trading (transaction costs).
    """
    def __init__(self, cost_coefficient=0.001):
        super(HedgingVarianceLoss, self).__init__()
        self.cost_coefficient = cost_coefficient

    def forward(self, hedge_ratio, asset_returns):
        """
        Calculates the financial risk loss.
        hedge_ratio: Predicted delta by the model (batch_size, 1)
        asset_returns: Actual price returns of the underlying asset (batch_size, 1)
        """
        # Hedged PnL = Action (Hedge Ratio) * Asset Return
        # If price drops and you shorted (-1), Hedged PnL is positive.
        hedged_returns = hedge_ratio * asset_returns
        
        # 1. Minimize Variance: We want the hedged returns to stabilize around a target
        mean_return = torch.mean(hedged_returns)
        variance = torch.mean((hedged_returns - mean_return) ** 2)
        
        # 2. Transaction Cost Penalty: Penalize high-frequency flipping of positions
        # Trading cost is proportional to the absolute size of the position
        tx_cost = torch.mean(torch.abs(hedge_ratio)) * self.cost_coefficient
        
        # Total Loss to minimize
        total_loss = variance + tx_cost
        return total_loss* 100000.0