import flwr as fl
from client import client_fn

strategy = fl.server.strategy.FedAvg()

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
