import numpy as np

class FinancialMetrics:
    """
    Computes professional risk management and performance metrics 
    to evaluate the hedging strategy.
    """
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculates the maximum peak-to-trough decline in portfolio value."""
        cumulative_pnl = np.cumsum(returns)
        # Convert cumulative sum to a wealth index starting at 1
        wealth_index = 1 + cumulative_pnl 
        historical_max = np.maximum.accumulate(wealth_index)
        # Avoid division by zero if historical_max is 0
        historical_max = np.where(historical_max == 0, 1e-8, historical_max)
        drawdowns = (wealth_index - historical_max) / historical_max
        return float(np.min(drawdowns))

    @staticmethod
    def calculate_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Calculates Conditional Value at Risk (cVaR) / Expected Shortfall.
        Represents the average loss in the worst (alpha)% of cases.
        """
        if len(returns) == 0:
            return 0.0
        sorted_returns = np.sort(returns)
        cutoff_idx = int(np.floor(alpha * len(sorted_returns)))
        if cutoff_idx == 0:
            return float(sorted_returns[0])
        # Average of the worst returns up to the cutoff
        worst_returns = sorted_returns[:cutoff_idx]
        return float(-np.mean(worst_returns))

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculates the annualized Sharpe Ratio (assuming hourly data)."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        # Hourly return to annualized scale (24 * 365 = 8760 hours/year)
        mean_excess_return = np.mean(returns) - (risk_free_rate / 8760)
        hourly_sharpe = mean_excess_return / np.std(returns)
        return float(hourly_sharpe * np.sqrt(8760))