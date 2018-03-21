"""
Ugur Gudelek
run
ugurgudelek
08-Mar-18
finance-cnn
"""

from dataset import IndicatorDataset
from model import CNN, LSTM
from config import Config

from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader

from torch import nn
from torch import optim
import numpy as np

import torch.nn.functional as F


def main():
    config = Config()
    dataset = IndicatorDataset(config=config, stock_names=['spy'], label_after=30)
    model = LSTM(input_size=28, seq_length=28, num_layers=1, out_size=3)
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=100, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(dataset.valid_dataset, batch_size=100, shuffle=False, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    # Train
    for epoch in range(50):
        acc = []
        valid_acc = []
        for i, (image, label) in enumerate(train_dataloader):
            image, label = Variable(image.float()), Variable(label.long()).squeeze(1)

            optimizer.zero_grad()

            # forward + backward + optimize
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward(retain_graph=True)
            optimizer.step()

            true_val = label.data.numpy()
            pred_val = output.data.numpy().argmax(axis=1)
            mul_bin = (true_val == pred_val)

            acc.append(mul_bin.mean())
            if i % 10 == 0:


                print(
                    'i: {:<5} | Epoch: {:<3} | Actual: {:07.3f} | Pred: {:07.3f} | TLoss: {:010.3f} | TAcc: {} | VLoss: {} | VAcc: {}'.format(
                        i * label.size()[-1],
                        epoch, loss.data[0],
                        true_val.mean(),
                        pred_val.mean(),
                        np.array(acc).mean(),
                        validate(
                            model,
                            valid_dataloader,
                            criterion, valid_acc), np.array(valid_acc).mean()
                    )
                )


def validate(model, valid_dataloader, criterion, valid_acc):
    # Validate
    model.eval()
    losses = []

    for i, (image, label) in enumerate(valid_dataloader):
        image, label = Variable(image.float()), Variable(label.long().squeeze(1))
        output = model.forward(image)
        loss = criterion(output, label)
        losses.append(loss.data[0])

        true_val = label.data.numpy()
        pred_val = output.data.numpy().argmax(axis=1)
        mul_bin = (true_val == pred_val)
        if i%5 == 0:
            print('1:',' '.join(map(str, true_val)),'\n2:', ' '.join(map(str,pred_val)))

        valid_acc.append(mul_bin)

    model.train()

    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
