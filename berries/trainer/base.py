__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import os
from collections import defaultdict
from typing import Iterable
import functools

import numpy as np
import pandas as pd
import torch
from berries.logger.logger import LocalLogger
from berries.plot.plotter import Plotter
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


def hook(before=None, after=None):

    def wrap(f):

        @functools.wraps(f)
        def wrapped_f(self, *args, **kwargs):

            if before:
                self.__getattribute__(before)(*args, **kwargs)

            returned_value = f(self, *args, **kwargs)

            if after:
                if returned_value is None:
                    self.__getattribute__(after)()
                elif isinstance(returned_value, tuple):
                    self.__getattribute__(after)(*returned_value)

            return returned_value

        return wrapped_f

    return wrap


class BaseTrainer():

    def __init__(self, model, metrics, hyperparams, params, optimizer,
                 criterion, logger) -> None:
        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        self.hyperparams = hyperparams
        self.params = params
        self.verbose = self.params.get('verbose', 1)

        self.device = torch.device('cuda:0' if self.params['device'] ==
                                   'cuda' else 'cpu')

        self.model = model.to(self.device).float()
        self.optimizer = optimizer or Adam(params=self.model.parameters(),
                                           lr=self.hyperparams.get('lr', 0.001),
                                           weight_decay=self.hyperparams.get(
                                               'weight_decay', 0))
        self.criterion = criterion or MSELoss()
        self.logger = logger or self._contruct_logger(backend=LocalLogger)

        self.metrics = metrics

        self.batch_size = self.hyperparams.get('batch_size', 16)
        self.validation_batch_size = self.hyperparams.get(
            'validation_batch_size', 16)

        # Initialize plotter
        self.plotter = Plotter()

        self._resume_or_not()

        self.stdout_items = dict()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @property
    def experiment_fpath(self):
        return self.params['root'] / 'projects' / self.params[
            'project_name'] / self.params['experiment_name']

    @property
    def on_cuda(self):
        return self.device.type == 'cuda'

    def __repr__(self):
        return """repr"""

    def _validate_hyperparams(self, hyperparams):
        raise NotImplementedError()

    def _validate_params(self, params):
        raise NotImplementedError()

    def _contruct_logger(self, backend):
        return backend(root=self.params['root'],
                       project_name=self.params['project_name'],
                       experiment_name=self.params['experiment_name'],
                       params=self.params,
                       hyperparams=self.hyperparams)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

    def _resume_or_not(self):
        # Resume or not
        if self.params['resume'] or self.params['pretrained']:

            cpt_path = self.experiment_fpath / 'checkpoints'
            # print(f"Loading checkpoint...{cpt_path}")
            if not cpt_path.exists():
                raise Exception(f"""
                    You do not have any checkpoint to resume.
                    If you want to start over, make sure --resume and --pretrained is False.
                    """)
            # todo: change with Path
            last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]
            self._load_checkpoint_from_epoch(epoch=last_epoch)
            print(f"Checkpoint is loaded from {last_epoch}")
        else:
            self.start_epoch = 1
            print("Starting training from epoch 1")

    def _save_checkpoint(self, epoch):  # todo: move into generic model
        # Save the model if the validation loss is the best we've seen so far.
        # is_best = val_loss > best_val_loss
        # best_val_loss = max(val_loss, best_val_loss)

        self.cpt_fpath = self.experiment_fpath / 'checkpoints' / str(epoch)
        self.cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, self.cpt_fpath / 'model-optim.pth')

    def _load_checkpoint_from_epoch(self,
                                    epoch):  # todo: move into generic model
        # load model
        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == 'cpu':
            map_location = self.device.type

        checkpoint = torch.load(self.experiment_fpath / 'checkpoints' /
                                str(epoch) / 'model-optim.pth',
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

    def _log_metrics(self, phase, epoch, container):
        for metric_key, metric_val in container.items():
            self.logger.log_metric(log_name=f'{phase}_{metric_key}',
                                   x=epoch,
                                   y=np.mean(metric_val))

    def handle_batch(self, batch):
        data = batch['data']
        target = batch['target']

        # cast data to a device
        data, target = data.to(self.device), target.to(self.device)

        return data, target

    def forward(self, data):
        # model forward pass
        return self.model(data)

    def compute_loss(self, output, targets):
        # compute loss
        loss = self.criterion(output, targets)
        return loss

    def _set_grad_enabled(self, train):
        # if torch.is_grad_enabled():

        # if train and not self.model.training:
        #     self.model.train()
        #     torch.set_grad_enabled(True)

        # if not train and self.model.training:
        #     self.model.train(False)
        #     torch.set_grad_enabled(False)

        torch.set_grad_enabled(train)
        self.model.train(train)

    def _log_metrics(self, phase, epoch, container):
        for metric_key, metric_val in container.items():
            self.logger.log_metric(log_name=f'{phase}_{metric_key}',
                                   x=epoch,
                                   y=np.mean(metric_val))

    def _calculate_metrics(self, yhat, y, container):
        for metric_fn in self.metrics:
            container[metric_fn.__name__].append(metric_fn()(yhat, y))
        return container

    def _to_loader(self, dataset, training, batch_size=None):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size or
            (self.batch_size if training else self.validation_batch_size),
            shuffle=self.params.get('train_shuffle', training),
            drop_last=self.params.get('drop_last', False),
            num_workers=0,
            pin_memory=self.on_cuda)

    @staticmethod
    def _make_iterable_if_not(item):
        return item if isinstance(item, Iterable) else (item,)

    @hook(before='before_fit_one_batch', after='after_fit_one_batch')
    def _fit_one_batch(self, batch, train):
        """All training steps are implemented here. 
        This function is the core of handling model - actual training loop.

        Args:
            batch (dict): [description]
            train (bool): [description]

        Returns:
            loss    (torch.Tensor): [description]
            output  (torch.Tensor): [description]
            data    (torch.Tensor): [description]
            target  (torch.Tensor): [description]
        """

        self._set_grad_enabled(train)

        data, target = self.handle_batch(batch)
        output = self.forward(data)
        loss = self.compute_loss(output, target)

        if train:
            # do not let pytorch accumulates gradient
            self.optimizer.zero_grad()

            # calculate gradient with backpropagation
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if self.hyperparams.get('clip', None):  # if clip is given
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.hyperparams['clip'])

            # distribute gradients to update weights
            self.optimizer.step()

        return loss, output, data, target

    @hook(before='before_fit_one_epoch', after='after_fit_one_epoch')
    def _fit_one_epoch(self, loaders, phase, epoch):

        if not isinstance(loaders, Iterable):
            raise Exception("loaders must be an iterable")

        metric_container = defaultdict(list)
        seen_item = 0
        for loader in loaders:
            for batch_ix, batch in enumerate(loader):

                loss, output, data, target = self._fit_one_batch(
                    batch, train=True if phase == 'training' else False)
                seen_item += len(target)

                # Store loss
                metric_container['loss'].append(loss.item())

                # Store other metrics
                metric_container = self._calculate_metrics(
                    yhat=output, y=target, container=metric_container)

                self.stdout_items['phase'] = phase
                self.stdout_items['epoch'] = epoch
                self.stdout_items['seen_item'] = seen_item
                self.stdout_items['dataset_len'] = sum(
                    len(loader.dataset) for loader in loaders)
                self.stdout_items['batch_loss'] = loss.item()

                if self.verbose != 0:
                    if batch_ix % self.params['stdout_interval'] == 0:
                        self.print_stdout_items()

        self._log_metrics(phase=phase,
                          epoch=epoch,
                          container={
                              key: np.array(values).mean()
                              for key, values in metric_container.items()
                          })

    @hook(before='before_fit', after='after_fit')
    def fit(self, dataset, validation_dataset):
        if self.params['pretrained']:
            raise Exception("You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.dataset = dataset
            self.validation_set = validation_dataset

            # Support for list of datasets
            train_loaders = [
                self._to_loader(dataset=d, training=True)
                for d in self._make_iterable_if_not(dataset)
            ]
            validation_loaders = [
                self._to_loader(dataset=d, training=False)
                for d in self._make_iterable_if_not(validation_dataset)
            ]

            # run 1 epoch before training to watch untrained model performance
            if not self.params['resume']:
                metric_container = self._fit_one_epoch(
                    loaders=validation_loaders, epoch=0, phase='validation')

            for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):

                for phase in ['training', 'validation']:

                    metric_container = self._fit_one_epoch(
                        loaders=train_loaders
                        if phase == 'training' else validation_loaders,
                        epoch=epoch,
                        phase=phase)

                if epoch % self.params['log_interval'] == 0:
                    self._save_checkpoint(epoch=epoch)
                    self.logger.save()

                self._save_checkpoint(epoch=epoch)

        except KeyboardInterrupt:
            print('Exiting from training early')

    def _transform(self, dataset, batch_size):
        loader = self._to_loader(dataset, training=False, batch_size=batch_size)

        transformed = []
        targets = []
        for batch_ix, batch in enumerate(loader):
            loss, output, data, target = self._fit_one_batch(batch, train=False)
            if output.shape[1] != 1:  # classification
                output = output.argmax(dim=1)
            transformed.append(output)
            targets.append(target)

        transformed = torch.cat(transformed, axis=0)
        targets = torch.cat(targets, axis=0)
        return transformed, targets

    def transform(self, dataset, batch_size=None):
        transformed, targets = self._transform(dataset, batch_size)
        return (transformed.cpu().detach().numpy(),
                targets.cpu().detach().numpy())

    def fit_transform(self, dataset):
        self.fit(dataset=dataset)
        return self.transform(dataset=dataset)

    def score(self, dataset):
        raise NotImplementedError()
        transformed = self._transform(dataset)
        return transformed == dataset.get_targets()

    def to_prediction_dataframe(self, dataset, save=True):
        predictions, targets = self.transform(dataset=dataset)
        prediction_dataframe = pd.DataFrame({
            'prediction': predictions,
            'target': targets
        })
        if save:
            prediction_dataframe.to_csv(self.experiment_fpath /
                                        'predictions.csv',
                                        index=False)
        return prediction_dataframe

    def print_stdout_items(self):

        phase = self.stdout_items['phase']
        epoch = self.stdout_items['epoch']
        seen_item = self.stdout_items['seen_item']
        dataset_len = self.stdout_items['dataset_len']
        batch_loss = self.stdout_items['batch_loss']

        print('\t'.join([
            f"{phase}",
            f"Epoch: {epoch} [{seen_item:04}/{dataset_len:04} ({100. * seen_item / dataset_len:.0f}%)]",
            f"Batch Loss: {batch_loss:.6f}",
        ]))

    # Hook methods

    def before_fit(self, dataset, validation_dataset):
        pass

    def after_fit(self):
        pass

    def before_fit_one_epoch(self, loaders, phase, epoch):
        pass

    def after_fit_one_epoch(self):
        pass

    def before_fit_one_batch(self, batch, train):
        pass

    def after_fit_one_batch(self, loss, output, data, target):
        pass
