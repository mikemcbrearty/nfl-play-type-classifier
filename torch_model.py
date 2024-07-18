import numpy as np
import torch

from data_loader import load_data
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Neural network model using PyTorch
# Adapted from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

class NflPlayDataset(Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)
        self.data = np.float32(data)
        self.targets = np.float32(targets)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


(train_data, val_data, test_data), (train_targets, val_targets, test_targets) = load_data()

train_dataset = NflPlayDataset(train_data, train_targets)
val_dataset = NflPlayDataset(val_data, val_targets)
test_dataset = NflPlayDataset(test_data, test_targets)

batch_size = 512
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


epochs = 20
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done training!")

print("Test against test dataset:")
test(test_dataloader, model, loss_fn)
