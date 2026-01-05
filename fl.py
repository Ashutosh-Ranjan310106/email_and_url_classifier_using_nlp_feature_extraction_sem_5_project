import flwr as fl

if __name__ == "__main__":
    # Start Flower server with FedAvg
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3)
    )

