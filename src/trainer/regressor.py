__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from history.history import History

from collections import defaultdict
import matplotlib.pyplot as plt

class RegressorTrainer:
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None, use_cuda=True):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(), lr=hyperparams['lr'])
        self.hyperparams = hyperparams
        self.params = params

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

        self.train_loader = DataLoader(self.dataset.trainset, batch_size=self.hyperparams['train_batch_size'],
                                       **self.loader_kwargs)
        self.test_loader = DataLoader(self.dataset.testset, batch_size=self.hyperparams['test_batch_size'],
                                      **self.loader_kwargs)

        self.history = History()

    def _on_epoch(self, epoch, train=True):
        self.model.train(train)
        loader = self.train_loader if train else self.test_loader

        self.model.hidden = None if train else self.model.hidden
        logs = dict()
        logs['loss'] = list()

        for batch_ix, (data, targets) in enumerate(loader):
            data, targets = data.to(self.device), targets.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, targets)
            if train:
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.
                loss.backward()
                self.optimizer.step()

            if batch_ix % self.params['log_interval'] == 0:
                print('\t'.join((
                    f"{'Train' if train else 'Test'}",
                    f"Epoch: {epoch} [{batch_ix * len(data)}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f}%)]",
                    f"Loss: {loss.item():.6f}",
                    # f"Proba: {self.proba(output)}",
                )))

                logs['loss'].append(loss.item())

                yield loss

        self.history.append(phase='train' if train else 'test',
                            log_dict={'epoch':epoch,
                                      'loss': np.mean(logs['loss']),
                                     })

        self.model.train(not train)

    def fit(self):
        # # See what the scores are before training
        # with torch.no_grad():
        #     for loss in self._on_epoch(train=False, epoch=0):
        #         pass

        for epoch in range(1, self.hyperparams['epoch']+1):
            for loss in self._on_epoch(train=True, epoch=epoch):
                    pass
            # with torch.no_grad():
            #     for loss in self._on_epoch(train=False, epoch=epoch):
            #         pass






    @staticmethod
    def proba(output):
        return torch.nn.functional.softmax(output.detach(), dim=1).cpu().numpy()

    def predict(self, data=None):
        if data is None:
            data = self.dataset.testset.data

        self.model.hidden = None
        pred = self.model(data)
        return pred.detach().numpy()

    def predict_log_proba(self):
        pass

    def predict_proba(self):
        pass

    # def score(self, data=None, targets=None, kind='accuracy'):
    #     if data is None:
    #         data = self.dataset.testset.data
    #     if targets is None:
    #         targets = self.dataset.testset.targets
    #
    #     output = self.model(data)
    #     if kind == 'accuracy':
    #         return self.accuracy(output, targets)
    #     raise RuntimeError(f"{kind} is not recognized in available kinds.")
