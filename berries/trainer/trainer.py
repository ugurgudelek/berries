__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

from ..plot import Plotter


class Trainer:
    def __init__(self,
                 root,
                 model, dataset, metrics,
                 hyperparams, params,
                 logger,
                 optimizer=None, criterion=None):

        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        self.root = root
        self.hyperparams = hyperparams
        self.params = params

        self.device = torch.device('cuda:0' if self.params['device'] == 'cuda' else 'cpu')

        self.model = model.to(self.device).float()
        self.dataset = dataset
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(),
                                           lr=self.hyperparams.get('lr', 0.001),
                                           weight_decay=self.hyperparams.get('weight_decay', 0))

        self.metrics = metrics

        self.experiment_fpath = self.root / 'projects' / params['project_name'] / params['experiment_name']

        self.train_loader = DataLoader(self.dataset.trainset,
                                       batch_size=self.hyperparams['train_batch_size'],
                                       drop_last=False,
                                       shuffle=True,  # todo: output.view(batch_size, -1) needs this!
                                       num_workers=0,
                                       pin_memory=self.on_cuda)  # True if cuda else otherwise
        # about pin_memory: https://stackoverflow.com/a/55564072

        self.test_loader = DataLoader(self.dataset.testset,
                                      batch_size=self.hyperparams['test_batch_size'],
                                      drop_last=False,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=self.on_cuda)  # True if cuda else otherwise

        self.logger = logger

        self._callback_fns = list()

        # Initialize plotter
        self.plotter = Plotter()

        # Resume or not
        if self.params['resume'] or self.params['pretrained']:
            print("Loading checkpoint...")
            cpt_path = self.experiment_fpath / 'checkpoints'
            if not cpt_path.exists():
                raise Exception(
                    """
                    You do not have any checkpoint to resume.
                    If you want to start over, make sure --resume and --pretrained is False.
                    """
                )
            last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]  # todo: change with Path
            self.load_checkpoint(epoch=last_epoch)
            print(f"Checkpoint is loaded from {last_epoch}")
        else:
            self.start_epoch = 1
            print("Starting training from epoch 1")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

    @property
    def on_cuda(self):
        return self.device.type == 'cuda'

    def _validate_hyperparams(self, hyperparams):
        raise NotImplementedError()

    def _validate_params(self, params):
        raise NotImplementedError()

    def _on_epoch(self, epoch, train=True):

        loader = self.train_loader if train else self.test_loader

        self.model.train(train)  # enable or disable dropout or batch norm

        seen_item = 0
        # todo: add aux into data
        for batch_ix, item in enumerate(loader):
            data = item['data']['x']
            targets = item['target']
            aux = item['data'].get('aux', None)

            data, targets = data.to(self.device), targets.to(self.device)
            if aux is not None:
                aux = aux.to(self.device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # self.model.reset_states()
            # hidden_ = self.model.repackage_hidden(hidden)

            # Loss
            # todo: add loss calculations. you can look into encoder_decoder_rnntrainer.py for more info.
            if aux is None:
                output = self.model(data)
            else:
                output = self.model(data, aux)
            loss = self.criterion(output, targets)

            if train:
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if self.hyperparams.get('clip', None):  # if clip is given
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams['clip'])
                self.optimizer.step()

            seen_item += len(data)
            if batch_ix % self.params['stdout_interval'] == 0:
                print('\t'.join((
                    f"{'Train' if train else 'Test'}",
                    f"Epoch: {epoch} [{seen_item}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f}%)]",
                    f"Batch Loss: {loss.item():.6f}",
                )))

            yield loss, output, targets

        self.model.train(not train)

    def fit(self):

        if self.params['pretrained']:
            raise Exception("-You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            # See what the scores are before training
            with torch.no_grad():
                loss_container = list()
                metric_container = defaultdict(list)
                for loss, output, target in self._on_epoch(train=False, epoch=0):
                    loss_container.append(loss.item())
                    metric_container = self._calculate_metrics(yhat=output, y=target, container=metric_container)
                self.logger.log_metric(log_name='validation_loss', x=0, y=np.mean(loss_container))
                self._log_metrics(phase='validation', epoch=0, container=metric_container)

            for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):
                # Training loop
                loss_container = list()
                metric_container = defaultdict(list)
                for loss, output, target in self._on_epoch(train=True, epoch=epoch):
                    loss_container.append(loss.item())
                    metric_container = self._calculate_metrics(yhat=output, y=target, container=metric_container)
                self.logger.log_metric(log_name='training_loss', x=epoch, y=np.mean(loss_container))
                self._log_metrics(phase='training', epoch=epoch, container=metric_container)

                # Validation loop
                loss_container = list()
                metric_container = defaultdict(list)
                with torch.no_grad():
                    for loss, output, target in self._on_epoch(train=False, epoch=epoch):
                        loss_container.append(loss.item())
                        metric_container = self._calculate_metrics(yhat=output, y=target, container=metric_container)
                self.logger.log_metric(log_name='validation_loss', x=epoch, y=np.mean(loss_container))
                self._log_metrics(phase='validation', epoch=epoch, container=metric_container)

                if epoch % self.params['log_interval'] == 0:
                    self.run_callbacks(epoch=epoch)
                    self.save_checkpoint(epoch=epoch)

                    self.logger.save()

                    # todo: add into logger
                    self.plotter.learning_curve(read_dir=self.experiment_fpath,
                                                save_dir=self.experiment_fpath / 'figures')

        except KeyboardInterrupt:
            print('Exiting from training early')

    def _log_metrics(self, phase, epoch, container):
        for metric_key, metric_val in container.items():
            self.logger.log_metric(log_name=f'{phase}_{metric_key}',
                                   x=epoch,
                                   y=np.mean(metric_val))

    def _calculate_metrics(self, yhat, y, container):
        for metric_fn in self.metrics:
            container[metric_fn.__name__].append(metric_fn()(yhat, y))
        return container

    def save_checkpoint(self, epoch):  # todo: move into generic model
        # Save the model if the validation loss is the best we've seen so far.
        # is_best = val_loss > best_val_loss
        # best_val_loss = max(val_loss, best_val_loss)

        self.cpt_fpath = self.experiment_fpath / 'checkpoints' / str(epoch)
        self.cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
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
        # self.history = checkpoint['history']

        if self.device.type == 'cuda':
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def predict(self, x):
        self.model.train(False)
        with torch.no_grad():
            output = self.model(x.to(self.device))

        output = output.detach().cpu().numpy()
        self.model.train(True)
        return output

    def predict_loader(self, dataloader):
        output_list = list()
        target_list = list()
        with torch.no_grad():
            for batch_ix, item in enumerate(dataloader):
                data = item['data']['x']
                targets = item['target']
                aux = item['data'].get('aux', None)

                data, targets = data.to(self.device), targets.to(self.device)
                if aux is not None:
                    aux = aux.to(self.device)

                if aux is None:
                    output = self.model(data)
                else:
                    output = self.model(data, aux)

                output_list.append(output.detach().cpu().numpy())
                target_list.append(targets.detach().cpu().numpy())

        return np.concatenate(output_list), np.concatenate(target_list)

    def attach_callback(self, callback_fn):
        self._callback_fns.append(callback_fn)

    def run_callbacks(self, epoch):
        for callback_fn in self._callback_fns:
            callback_fn(instance=self, epoch=epoch)

    @staticmethod
    def proba(x):
        return torch.nn.functional.softmax(x.detach(), dim=1).cpu().numpy()
