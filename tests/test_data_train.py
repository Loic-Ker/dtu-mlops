import torch
import random
import os
import pytest


@pytest.mark.skipif(not os.path.exists("data/processed/train.pt"), reason="No Datas")
def test_length():
    data = torch.load("data/processed/train.pt")  # datas
    assert len(set([label for _, label in data])) == 10, "Not the good number of classes"


def test_shape():
    data = torch.load("data/processed/train.pt")  # datas
    randomlist = random.sample(range(0, len(data) - 1), 200)
    for i in randomlist:  # good size
        assert data[i][0].shape == torch.Size([1, 28, 28]), "No the good image size"
