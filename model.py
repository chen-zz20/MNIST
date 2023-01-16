import torch
import torch.utils.data
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )


    def forward(self, x:torch.tensor):
        return self.net(x)


def calculate_acc(output, label):
    correct = torch.sum(torch.argmax(output, dim=1) == label)
    return correct / len(label)


def train_epoch(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, loss_metrics: torch.nn.Module, optimizer:torch.optim, device:str):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        times += 1
        loss = loss_metrics(torch.argmax(outputs, dim=1), labels)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        train_loss += loss.item()
        train_acc += calculate_acc(outputs, labels)
        optimizer.step()
    
    train_loss = train_loss / times
    train_acc = train_acc / times

    return train_loss, train_acc


def test_epoch(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, loss_metrics: torch.nn.Module, device:str):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        times += 1
        loss = loss_metrics(torch.argmax(outputs, dim=1), labels)

        test_loss += loss.item()
        test_acc += calculate_acc(outputs, labels).cpu().data.numpy()
    
    test_loss = test_loss / times
    test_acc = test_acc / times

    return test_loss, test_acc
