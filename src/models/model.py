import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self, perc_dropout: float):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=perc_dropout)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class M(nn.Module):
    def __init__(self, perc_dropout: float):
        super(M, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)

        self.dequant = torch.quantization.DeQuantStub()

        self.dropout = nn.Dropout(p=perc_dropout)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.quant(x)
        x = self.relu(self.fc3(x))
        x = self.dequant(x)
        x = self.dropout(x)
        x = F.log_softmax(self.dequant(self.fc4(x)), dim=1)

        return x
