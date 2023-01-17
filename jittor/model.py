import jittor as jt
from jittor.dataset.dataset import DataLoader, Dataset
import jittor.nn as nn


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


    def execute(self, x:jt.Var):
        return self.net(x)


def calculate_acc(output, label):
    correct = jt.sum(jt.argmax(output, dim=1)[0] == label)
    return correct / len(label)


def train_epoch(model:nn.Module, data_loader: Dataset, loss_metrics: nn.Module, optimizer:jt.optim):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):

        outputs = model(inputs)
        times += 1
        loss = loss_metrics(outputs, nn.one_hot(labels, 10))

        optimizer.zero_grad()
        train_loss += loss.item()
        train_acc += calculate_acc(outputs, labels).numpy()
        optimizer.step(loss)
    
    train_loss = train_loss / times
    train_acc = train_acc / times

    return train_loss, train_acc


def test_epoch(model:nn.Module, data_loader:Dataset, loss_metrics: nn.Module):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):

        outputs = model(inputs)
        times += 1
        loss = loss_metrics(outputs, nn.one_hot(labels, 10))

        test_loss += loss.item()
        test_acc += calculate_acc(outputs, labels).numpy()
    
    test_loss = test_loss / times
    test_acc = test_acc / times

    return test_loss, test_acc
