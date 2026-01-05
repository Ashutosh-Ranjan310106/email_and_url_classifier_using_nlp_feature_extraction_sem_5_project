import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=5)
    )

