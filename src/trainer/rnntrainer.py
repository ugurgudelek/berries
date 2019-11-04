__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pathlib import Path

from history.history import History

from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision


# class TimeSeriesDataloader():
#     def __init__(self, dataset, batch_size, seq_len, **kwargs):
#         self.dataset = dataset
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.time_skip = len(self.dataset) // self.batch_size
#
#     def __iter__(self):
#         for ix in range(len(self.dataset)):
#
#             col_ix = ix % self.batch_size
#             batch_ix = ix // self.batch_size
#
#             real_ix = col_ix * self.time_skip + batch_ix * self.seq_len
#             yield self.dataset[real_ix]

# class TimeSeriesDatasetSampler(torch.utils.data.sampler.Sampler):
#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices (list, optional): a list of indices
#         num_samples (int, optional): number of samples to draw
#
#     Examples:
#         train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         sampler=ImbalancedDatasetSampler(train_dataset),
#         batch_size=args.batch_size,
#         **kwargs
#     )
#     """
#
#     def __init__(self, num_samples, time_skip):
#         self.num_samples = num_samples
#         self.time_skip = time_skip
#
#     def __iter__(self):
#         for i in range(0, self.num_samples, self.time_skip):
#             yield i
#
#     def __len__(self):
#         return self.num_samples


class RNNTrainer:
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None, use_cuda=True):
        self.hyperparams = hyperparams
        self.params = params

        self.model = model
        self.dataset = dataset
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(),
                                           lr=self.hyperparams['lr'],
                                           weight_decay=self.hyperparams['weight_decay'])
        self.device = self.params['device']
        self.model = self.model.to(self.device)
        # self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.loader_kwargs = {'drop_last': False, 'shuffle': False}


        # self.train_loader = TimeSeriesDataloader(self.dataset.trainset,
        #                                batch_size=self.hyperparams['train_batch_size'],
        #                                          seq_len=self.hyperparams['seq_len'],
        #                                # sampler=TimeSeriesDatasetSampler(13056, 204),
        #                                **self.loader_kwargs)
        # self.test_loader = TimeSeriesDataloader(self.dataset.testset,
        #                               batch_size=self.hyperparams['test_batch_size'],
        #                                         seq_len=self.hyperparams['seq_len'],
        #                               **self.loader_kwargs)

        self.history = History()

    @staticmethod
    def get_batch(source, i):
        seq_len = min(50, len(source) - 1 - i)
        data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
        target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
        return data, target

    def _on_epoch(self, epoch, train=True):
        # Train and test differences
        with_grad_or_not = torch.enable_grad if train else torch.no_grad
        # loader = self.train_loader if train else self.test_loader
        dataset = self.dataset.trainset.batched_data if train else self.dataset.testset.batched_data
        self.model.train(train)  # enable or disable dropout

        hidden = self.model.init_hidden(self.hyperparams['train_batch_size'])

        with with_grad_or_not():

            logs = dict()
            logs['loss'] = list()
            for batch_ix, i in enumerate(range(0, len(dataset), 50)):

                data, targets = self.get_batch(dataset, i)
                data, targets = data.to(self.device), targets.to(self.device)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = self.model.repackage_hidden(hidden)
                hidden_ = self.model.repackage_hidden(hidden)
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.

                # Loss1: Free Running Loss
                outVal = data[0].unsqueeze(0)
                outVals = []
                hids1 = []
                for i in range(data.size(0)):
                    outVal, hidden_, hid = self.model.forward(outVal, hidden_, return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals, dim=0)
                hids1 = torch.cat(hids1, dim=0)
                loss1 = self.criterion(outSeq1.view(self.hyperparams['train_batch_size'], -1),
                                       targets.contiguous().view(self.hyperparams['train_batch_size'], -1))

                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model.forward(data, hidden, return_hiddens=True)
                loss2 = self.criterion(outSeq2.view(self.hyperparams['train_batch_size'], -1),
                                       targets.contiguous().view(self.hyperparams['train_batch_size'], -1))

                '''Loss3: Simplified Professor forcing loss'''
                loss3 = self.criterion(hids1.view(self.hyperparams['train_batch_size'], -1),
                                       hids2.view(self.hyperparams['train_batch_size'], -1).detach())

                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1 + loss2 + loss3

                if train:
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams['clip'])
                    self.optimizer.step()

                logs['loss'].append(loss.item())

                if batch_ix % self.params['log_interval'] == 0 and batch_ix > 0:
                    print('\t'.join((
                        f"{'Train' if train else 'Test'}",
                        f"Epoch: {epoch} ", #[{batch_ix * len(data)}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f} % )]
                f"Loss: {np.array(logs['loss']).mean().item():.6f}",
                # f"Proba: {self.proba(output)}",
                )))

            self.history.append(phase='train' if train else 'test',
                                log_dict={'epoch': epoch,
                                          'loss': np.mean(logs['loss']),
                                          })

            self.model.train(not train)

        return np.mean(logs['loss'])


    def fit(self):
        # # See what the scores are before training
        # with torch.no_grad():
        #     for loss in self._on_epoch(train=False, epoch=0):
        #         pass

        for epoch in range(1, self.hyperparams['epoch'] + 1):
            train_loss = self._on_epoch(train=True, epoch=epoch)
            print(f'Train| end of epoch {epoch:3d} | | {train_loss:5.4f} |')
            val_loss = self._on_epoch(train=False, epoch=epoch)
            print(f'Val | end of epoch {epoch:3d} | | {val_loss:5.4f} |')

            self.generate_output(self.hyperparams, self.params, epoch, self.model, self.dataset.testset,
                                 startPoint=1500)

            if epoch % self.params['save_interval'] == 0:
                # Save the model if the validation loss is the best we've seen so far.
                # is_best = val_loss > best_val_loss
                # best_val_loss = max(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    # 'best_loss': best_val_loss,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'hyperparams': self.hyperparams,
                                    'params': self.params
                                    }
                # self.model.save_checkpoint(model_dictionary, True)


    @staticmethod
    def generate_output(hyperparams, params, epoch, model, testset, disp_uncertainty=True, startPoint=500,
                        endPoint=3500):
        if params['save_fig']:
            # Turn on evaluation mode which disables dropout.
            model.eval()
            hidden = model.init_hidden(1)
            outSeq = []
            upperlim95 = []
            lowerlim95 = []
            with torch.no_grad():
                for i in range(endPoint):
                    if i >= startPoint:
                        out, hidden = model.forward(out, hidden)
                    else:
                        out, hidden = model.forward(testset.data[i].unsqueeze(0).unsqueeze(0), hidden)
                    outSeq.append(out.data.cpu()[0][0].unsqueeze(0))

            outSeq = torch.cat(outSeq, dim=0)  # [seqLength * feature_dim]
            # self.data = self.scaler.transform(self.data)
            target = testset.scaler.inverse_transform(testset.data).unsqueeze(1)
            outSeq = testset.scaler.inverse_transform(outSeq)

            plt.figure(figsize=(15, 5))
            for i in range(target.size(-1)):
                plt.plot(target[:, :, i].numpy(), label='Target' + str(i),
                         color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                plt.plot(range(startPoint), outSeq[:startPoint, i].numpy(),
                         label='1-step predictions for target' + str(i),
                         color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)

                plt.plot(range(startPoint, endPoint), outSeq[startPoint:, i].numpy(),
                         label='Recursive predictions for target' + str(i),
                         color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)

            plt.xlim([startPoint - 500, endPoint])
            plt.xlabel('Index', fontsize=15)
            plt.ylabel('Value', fontsize=15)
            # plt.title('Time-series Prediction on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
            plt.legend()
            plt.tight_layout()
            plt.text(startPoint - 500 + 10, target.min(), 'Epoch: ' + str(epoch), fontsize=15)
            save_dir = Path('../results')
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir.joinpath('fig_epoch' + str(epoch)).with_suffix('.png'))
            plt.close()
            return outSeq

        else:
            pass


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
