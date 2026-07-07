import flwr as fl
from fedprox_strategy import get_fedprox_strategy

def main():
    print("==================================================")
    print("🛡️ FEDHEDGE GLOBAL SERVER LISTENING ON PORT 8080 🛡️")
    print("==================================================\n")
    
    strategy = get_fedprox_strategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1000),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()