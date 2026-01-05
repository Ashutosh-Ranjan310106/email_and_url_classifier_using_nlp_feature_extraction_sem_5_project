import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import XORModel

# --- Local XOR datasets ---
client_data = {
    0: {
        "x": torch.tensor([[0.,0.], [1.,1.]]),
        "y": torch.tensor([[0.], [0.]])
    },
    1: {
        "x": torch.tensor([[0.,1.], [1.,0.]]),
        "y": torch.tensor([[1.], [1.]])
    }
}

class XORClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = XORModel()
        self.x = client_data[cid]["x"]
        self.y = client_data[cid]["y"]
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    def get_parameters(self, config=None):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        for _ in range(200):            # Train locally
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(), len(self.x), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model(self.x)
        loss = self.loss_fn(y_pred, self.y)
        correct = ((y_pred > 0.5) == self.y).float().mean().item()
        return loss.item(), len(self.x), {"accuracy": correct}

def client_fn(cid):
    return XORClient(int(cid))
