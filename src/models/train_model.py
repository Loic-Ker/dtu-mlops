import torch
from torch import nn, optim
from model import Classifier


def train(epochs=20, lr=0.001):
    print("Training")

    model = Classifier()

    train = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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

        ## TODO: Implement the validation pass and print out the validation accuracy
        print(f"Accuracy, validation: {(accuracy_epoch/count)*100}%")


if __name__ == "__main__":
    train()
