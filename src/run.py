"""
Ugur Gudelek
run
ugurgudelek
08-Mar-18
finance-cnn
"""

from dataset import IndicatorDataset
from model import CNN
from config import Config

from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader

from torch import nn
from torch import optim

def main():
    config = Config()
    dataset = IndicatorDataset(config=config, stock_names=['spy'], row_len=28)
    model = CNN()
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=100, shuffle=False)
    valid_dataloader = DataLoader(dataset.valid_dataset, batch_size=10, shuffle=False)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    # Train
    for epoch in range(20):
        for i, (image, label) in enumerate(train_dataloader):
            image, label = Variable(image.float()), Variable(label.float())

            optimizer.zero_grad()

            # forward + backward + optimize
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('i: {:<3} | Epoch: {:<3} | Loss: {:010.3f} | Valid_loss: {:010.3f}'.format(i*label.size()[-1], epoch, loss.data[0], validate(model, valid_dataloader, criterion)))




def validate(model, valid_dataloader, criterion):
    # Validate
    model.eval()
    losses = []
    for i, (image, label) in enumerate(valid_dataloader):
        image, label = Variable(image.float()), Variable(label.float())
        output = model.forward(image)
        loss = criterion(output, label)
        losses.append(loss.data[0])

    model.train()

    return sum(losses)/len(losses)

if __name__ == "__main__":
    main()