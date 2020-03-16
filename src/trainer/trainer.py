__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from history.history import History


class Trainer:
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None):
        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        self.hyperparams = hyperparams
        self.params = params
        self.device = torch.device('cuda:0' if self.params['device'] == 'cuda' else 'cpu')
        self.model = model.to(self.device).double()
        self.dataset = dataset
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(),
                                           lr=self.hyperparams['lr'],
                                           weight_decay=self.hyperparams['weight_decay'])

        # self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        # self.loader_kwargs = {'drop_last': False, 'shuffle': False}

        self.train_loader = DataLoader(self.dataset.trainset,
                                       batch_size=self.hyperparams['train_batch_size'],
                                       drop_last=True)  # todo: output.view(batch_size, -1) needs this!

        self.test_loader = DataLoader(self.dataset.testset,
                                      batch_size=self.hyperparams['test_batch_size'],
                                      drop_last=True)

        # self.test_loader = Dataloader(self.dataset.testset,
        #                               batch_size=self.hyperparams['test_batch_size'],
        #                                         seq_len=self.hyperparams['seq_len'],
        #                               **self.loader_kwargs)

        self.experiment_fpath = Path(f'../experiments/{self.params["experiment_name"]}')
        self.experiment_fpath.mkdir(parents=True, exist_ok=True)

        # Resume or not
        if self.params['resume'] or self.params['pretrained']:
            print("=> loading checkpoint ")
            cpt_path = self.experiment_fpath / 'checkpoints'
            if not cpt_path.exists():
                raise Exception(
                    "You do not have any checkpoint to resume\n if you want to start over. Make sure --resume and --pretrained is False")
            last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]  # todo: change with Path
            self.load_checkpoint(epoch=last_epoch)
            print("=> loaded checkpoint")
        else:
            self.start_epoch = 1
            print("=> Start training from scratch")

        # Init history to log loss or something
        self.history = History()

    def callbacks(self, epoch, train=False):
        self.model.train(False)
        y = self.dataset.trainset.labels

        yhat = self.model(torch.DoubleTensor(self.dataset.trainset.data).to(self.device)).detach().cpu().numpy()


        plt.plot(self.dataset.inverse_transform(y[:, 0, :]), label='y')
        for i in range(yhat.shape[2]):
            plt.plot(self.dataset.inverse_transform(yhat[:, 0, i][:, np.newaxis]), label=f'yhat{i}')
        plt.legend()
        plt.show()
        self.model.train(True)

    def _validate_hyperparams(self, hyperparams):
        raise NotImplementedError()

    def _validate_params(self, params):
        raise NotImplementedError()

    def _on_epoch(self, epoch, train=True):
        # todo: reimplement this function
        batch_size = self.hyperparams['train_batch_size'] if train else self.hyperparams['test_batch_size']
        # Disable gradient calculations if validation or test period is active.
        with_grad_or_not = torch.enable_grad if train else torch.no_grad
        loader = self.train_loader if train else self.test_loader

        # hidden = self.model.init_hidden(self.hyperparams['train_batch_size'])

        self.model.train(train)  # enable or disable dropout
        with with_grad_or_not():

            logs = dict()
            logs['loss'] = list()
            for batch_ix, (data, targets) in enumerate(loader):

                data, targets = data.to(self.device), targets.to(self.device)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                # self.model.reset_states()
                # hidden_ = self.model.repackage_hidden(hidden)
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.

                # Loss
                # todo: add loss calculations. you can look into encoder_decoder_rnntrainer.py for more info.
                output = self.model(data)
                loss = self.criterion(output, targets)

                # # Loss1: Free Running Loss
                # outVal = data[0].unsqueeze(0)
                # outVals = []
                # hids1 = []
                # for i in range(data.size(0)):
                #     outVal, hidden_, hid = self.model.forward(outVal, hidden_, return_hiddens=True)
                #     outVals.append(outVal)
                #     hids1.append(hid)
                # outSeq1 = torch.cat(outVals, dim=0)
                # hids1 = torch.cat(hids1, dim=0)
                # loss1 = self.criterion(outSeq1.view(self.hyperparams['train_batch_size'], -1),
                #                        targets.contiguous().view(self.hyperparams['train_batch_size'], -1))
                #
                # '''Loss2: Teacher forcing loss'''
                # outSeq2, hidden, hids2 = self.model.forward(data, hidden, return_hiddens=True)
                # loss2 = self.criterion(outSeq2.view(self.hyperparams['train_batch_size'], -1),
                #                        targets.contiguous().view(self.hyperparams['train_batch_size'], -1))
                #
                # '''Loss3: Simplified Professor forcing loss'''
                # loss3 = self.criterion(hids1.view(self.hyperparams['train_batch_size'], -1),
                #                        hids2.view(self.hyperparams['train_batch_size'], -1).detach())
                #
                # '''Total loss = Loss1+Loss2+Loss3'''
                # loss = loss1 + loss2 + loss3

                if train:
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    if self.hyperparams.get('clip', None):  # if clip is given
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams['clip'])
                    self.optimizer.step()

                logs['loss'].append(loss.item())

                print('\t'.join((
                    f"{'[  Training]' if train else '[Validation]'}",
                    f"Epoch: {epoch:3d} ",
                    f"[{batch_ix * len(data)}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f} % )]",
                    f"Loss: {np.array(logs['loss']).mean().item():5.4f}",
                    # f"Proba: {self.proba(output)}",
                )))

            self.history.append(phase='train' if train else 'test',
                                log_dict={'epoch': epoch,
                                          'loss': np.mean(logs['loss']),
                                          })

        self.model.train(not train)

        return np.mean(logs['loss'])

    def fit(self):  # todo: move into trainer

        if self.params['pretrained']:
            raise Exception("-You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            # # See what the scores are before training
            # with torch.no_grad():
            #     for loss in self._on_epoch(train=False, epoch=0):
            #         pass

            for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):
                train_loss = self._on_epoch(train=True, epoch=epoch)
                print(f'[E.   Training] Epoch {epoch:3d} || Loss:{train_loss:5.4f} |')
                val_loss = self._on_epoch(train=False, epoch=epoch)
                print(f'[E. Validation] Epoch {epoch:3d} || Loss:{val_loss:5.4f} |')

                if epoch % self.params['save_interval'] == 0:
                    if self.params['save_fig']:
                        self.callbacks(epoch=epoch)
                        self.save_checkpoint(epoch=epoch)

                        self.learning_curve()


        except KeyboardInterrupt:
            print('Exiting from training early')

    def save_checkpoint(self, epoch):  # todo: move into generic model
        # Save the model if the validation loss is the best we've seen so far.
        # is_best = val_loss > best_val_loss
        # best_val_loss = max(val_loss, best_val_loss)

        self.cpt_fpath = self.experiment_fpath / 'checkpoints' / str(epoch)
        self.cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'hyperparams': self.hyperparams,
                    'params': self.params},
                   self.cpt_fpath / 'model-optim.pth')

    def load_checkpoint(self, epoch):  # todo: move into generic model
        # load model
        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == 'cpu':
            map_location = self.device.type

        checkpoint = torch.load(self.experiment_fpath / 'checkpoints' / str(epoch) / 'model-optim.pth',
                                map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        # self.hyperparams = checkpoint['hyperparams']
        # self.params = checkpoint['params']

        if self.device.type == 'cuda':
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def predict(self, data):
        raise NotImplementedError()

    @staticmethod
    def proba(output):  # todo: move into trainer
        return torch.nn.functional.softmax(output.detach(), dim=1).cpu().numpy()

    def predict_log_proba(self):
        raise NotImplementedError()

    def predict_proba(self):
        raise NotImplementedError()

    def learning_curve(self):
        train_df, test_df = self.history.to_dataframe(phase='both')
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(train_df['epoch'], train_df['loss'], label='training loss', color='orange')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(loc='upper right')

        ax[1].plot(test_df['epoch'], test_df['loss'], label='test loss', color='blue')
        ax[1].set_ylabel('Magnitude')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Epoch')

        save_dir = self.experiment_fpath / 'figures'
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'lr.png')
        plt.close()
