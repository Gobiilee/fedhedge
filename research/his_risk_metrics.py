import numpy as np
import pandas as pd
from typing import Tuple


def calculate_historical_risk_metrics(
    returns: pd.Series, 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate the Historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).

    Args:
        returns (pd.Series): A pandas Series of historical portfolio returns.
        confidence_level (float): The confidence level for the risk metrics 
                                  (e.g., 0.95 for 95%, 0.99 for 99%).

    Returns:
        Tuple[float, float]: A tuple containing (VaR, CVaR). Both are returned 
                             as positive numbers representing the loss magnitude.
                             Returns (0.0, 0.0) if the input series is empty.
    """
    if returns.empty:
        return 0.0, 0.0

    # Ensure the returns are sorted in ascending order (worst losses at the top)
    sorted_returns: np.ndarray = np.sort(returns.values)
    
    # Calculate the index corresponding to the given confidence level
    # E.g., for 95% confidence, we look at the 5th percentile
    percentile: float = 1.0 - confidence_level
    var_index: int = int(np.floor(percentile * len(sorted_returns)))
    
    # Value at Risk is the return at the threshold index
    # We take the negative to represent loss as a positive value
    var_value: float = -sorted_returns[var_index]
    
    # CVaR (Expected Shortfall) is the expected return given that the loss is worse than VaR
    # We slice the array up to the var_index and calculate the mean
    tail_losses: np.ndarray = sorted_returns[:var_index]
    
    if len(tail_losses) == 0:
        cvar_value: float = var_value
    else:
        cvar_value: float = -np.mean(tail_losses)

    return var_value, cvar_value


# --- Example Usage ---
if __name__ == "__main__":
    # Generate random synthetic daily returns for a dummy portfolio
    np.random.seed(42)
    # Mean = 0.0005, Standard Deviation = 0.02, 1000 days
    simulated_returns = pd.Series(np.random.normal(0.0005, 0.02, 1000))
    
    # Calculate 99% VaR and CVaR
    conf_level = 0.99
    portfolio_var, portfolio_cvar = calculate_historical_risk_metrics(
        returns=simulated_returns, 
        confidence_level=conf_level
    )
    
    print(f"99% Value at Risk (VaR): {portfolio_var:.4f}")
    print(f"99% Conditional VaR (CVaR): {portfolio_cvar:.4f}")