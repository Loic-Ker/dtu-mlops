import logging

import hydra
import torch
from model import Classifier
from omegaconf import OmegaConf
from torch import nn, optim

import wandb

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default.yaml")
def train(config):
    log.info("Training")
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    dropout = hparams["perc_dropout"]

    wandb.init(
        project="MNIST",
        config=hparams,
    )

    model = Classifier(dropout)

    train = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.watch(model, criterion, log="all", log_freq=100)

    print("Start training")

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"Training loss": loss.item()})
        # turn off gradients
        with torch.no_grad():
            # validation pass here
            count = 0
            accuracy_epoch = 0
            for images, labels in test_set:
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                count += 1
                accuracy_epoch += accuracy
                wandb.log({"Validation accuracy": accuracy})
        print(f"Accuracy, validation: {(accuracy_epoch/count)*100}%")

    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Training finished !")


if __name__ == "__main__":
    train()
