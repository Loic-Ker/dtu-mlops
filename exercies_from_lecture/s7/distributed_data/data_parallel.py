import torch.nn.functional as F
from torch import nn
import torch


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

net = Classifier(0.15)
device = torch.device('cpu') #implement to start with gpu but I dont have one

net_para = nn.DataParallel(net, device_ids=[0])

train = torch.load("../../../data/processed/train.pt")
train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

import time
start = time.time()

count=0
for images, labels in train_set:
    count+=1
    log_ps = net(images)
    if count==20:
        break

end = time.time()
print(end-start)

start = time.time()

count=0
for images, labels in train_set:
    count+=1
    log_ps = net_para(images)
    if count==20:
        break

end = time.time()
print(end-start)

