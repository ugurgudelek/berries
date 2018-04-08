from model import CNN, LSTM
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torch.autograd import Variable

import numpy as np
class Estimator:
    """

    """

    def __init__(self, dataset, model_config, dataloader_config):
        self.model = LSTM(input_size=model_config['input_size'],
                     seq_length=model_config['seq_length'],
                     num_layers=model_config['num_layers'],
                     out_size=model_config['out_size'])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.train_dataloader = DataLoader(dataset.train_dataset,
                                      batch_size=dataloader_config['train_batch_size'],
                                      shuffle=dataloader_config['train_shuffle'],
                                      drop_last=True)
        self.valid_dataloader = DataLoader(dataset.valid_dataset,
                                      batch_size=dataloader_config['valid_batch_size'],
                                      shuffle=dataloader_config['valid_shuffle'],
                                      drop_last=True)


    def run_epoch(self, epoch):

        # Train
        # tobecontinued...
        for i, (timage, tlabel) in enumerate(self.train_dataloader):
            timage, tlabel = Variable(timage.float()), Variable(tlabel.float())

            toutput, tloss = self.train_on_batch(timage, tlabel)


        # Validate
        for i, (vimage, vlabel) in enumerate(self.valid_dataloader):
            vimage, vlabel = Variable(vimage.float()), Variable(vlabel.float())
            voutput, vloss = self.validate_on_batch(vimage, vlabel)

        return (toutput, tloss, voutput, vloss)



    def train_on_batch(self, Xs, ys):
        self.optimizer.zero_grad()

        # forward + backward + optimize
        output = self.model.forward(Xs)
        loss = self.criterion(output, ys)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return output, loss

    def validate_on_batch(self, Xs, ys):
        self.model.eval()

        output, loss = self.train_on_batch(Xs, ys)

        self.model.train()

        return output, loss