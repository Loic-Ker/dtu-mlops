import timeit

import torch
from model import Classifier, M
from torch import nn, optim


def train(model):
    lr = 0.001
    epochs = 5

    model = model

    train = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"Accuracy, validation: {(accuracy_epoch/count)*100}%")

    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Training finished !")


t0 = timeit.Timer(
    stmt="train(model)",
    setup="from __main__ import train",
    globals={"model": Classifier(0.15)},
)

model_fp32 = M(0.15)
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [["fc3", "relu"]])
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

t1 = timeit.Timer(
    stmt="train_quantized(model)",
    setup="from __main__ import train",
    globals={"model": model_fp32_prepared},
)

print(t0.timeit(2))
print(t1.timeit(2))
