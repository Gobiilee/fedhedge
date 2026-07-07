import os
import sys
import flwr as fl
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

class SaveModelStrategy(fl.server.strategy.FedProx):
    """
    Custom FL Strategy that extends FedProx to save the global model 
    weights to disk after the final federated round.
    """
    def aggregate_fit(self, server_round, results, failures):
        # 1. Call the parent class to perform standard FedProx aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. Check if it is the final round (Assuming 5 rounds total)
        if aggregated_parameters is not None and server_round == 5:
            print(f"\n[SERVER] --- Saving Global Model at Round {server_round} ---")
            
            # Convert Flower's byte-format parameters to a list of NumPy arrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Ensure the output directory exists
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            
            # Save the arrays to a compressed .npz file
            save_path = config.FEDHEDGE_MODEL
            np.savez(save_path, *aggregated_ndarrays)
            print(f"[SERVER] --- Successfully saved AI weights to {save_path} ---\n")

        return aggregated_parameters, aggregated_metrics
    
def get_fedprox_strategy():
    """
    Returns the custom strategy for the global server.
    """
    return SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        proximal_mu=0.1,  # Penalty term for non-IID data
    )