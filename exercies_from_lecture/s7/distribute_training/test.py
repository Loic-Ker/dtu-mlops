



import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])

dataset = datasets.ImageFolder('lfw',lfw_trans)

dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=9)
