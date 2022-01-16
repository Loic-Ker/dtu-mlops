import torch
from src.models.model import Classifier
import os
import pytest


@pytest.mark.skipif(not os.path.exists("data/processed/test.pt"), reason="No Datas")
def test_model():
    data = torch.load("data/processed/train.pt")
    data = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)  # datas
    model = Classifier()
    assert model(iter(data).next()[0]).shape == torch.Size(
        [64, 10]
    ), "Not the good output"
