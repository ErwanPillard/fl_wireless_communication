import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.mnist_net import Net

DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Données locales (chaque client a son propre subset)
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    return trainset, testset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(DEVICE)
        self.trainset, self.testset = load_data()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(self.model(x), y)
            loss.backward()
            optimizer.step()

        return self.get_parameters({}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.testset, batch_size=32)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = self.model(x)
                loss += loss_fn(preds, y).item()
                correct += (preds.argmax(1) == y).sum().item()

        accuracy = correct / len(self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)