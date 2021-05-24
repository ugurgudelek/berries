__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import os
from collections import defaultdict
from typing import Iterable
import functools

import numpy as np
import pandas as pd
import torch
from berries.plot.plotter import Plotter
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import warnings


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

        self.hyperparams = hyperparams
        self.params = params
        self.logger = logger

        self.device = torch.device('cuda:0' if self.params['device'] ==
                                   'cuda' else 'cpu')

        self.model = model.to(self.device).float()
        self.optimizer = optimizer or Adam(params=self.model.parameters(),
                                           lr=self.hyperparams.get('lr', 0.001),
                                           weight_decay=self.hyperparams.get(
                                               'weight_decay', 0))
        self.criterion = criterion or MSELoss()
        if self.criterion.reduction != 'none' and self.params.get(
                'log_history', False):
            warnings.warn("""
                          'reduction' parameter of the criterion is not 'none' and
                          'log_history' parameters of the self.params is True
                          In this configuration history_container has invalid shape for loss attibute.
                          Therefore to be able to run experiment. log_history is set to False.
                          """)
            self.params['log_history'] = False

        self.metrics = metrics

        self.batch_size = self.hyperparams.get('batch_size', 16)
        self.validation_batch_size = self.hyperparams.get(
            'validation_batch_size', 16)

        # Initialize plotter
        self.plotter = Plotter()

        self._resume_or_not()

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

    def _resume_or_not(self):
        # Resume or not
        if self.params['resume'] or self.params['pretrained']:

            cpt_path = self.experiment_fpath / 'checkpoints'
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
            self.epoch = 1
            print("Starting training from epoch 1")

    def _get_last_checkpoint_path(self):
        cpt_path = self.experiment_fpath / 'checkpoints'
        last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]
        path = self.experiment_fpath / f'checkpoints/{last_epoch}/model-optim.pth'
        return path

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
        self.epoch = epoch
        # self.hyperparams = checkpoint['hyperparams']
        # self.params = checkpoint['params']
        # self.history = checkpoint['history']

        if self.device.type == 'cuda':
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

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

    def _calculate_metrics(self, yhat, y, container):
        for metric_fn in self.metrics:
            container[metric_fn.__name__].append(metric_fn()(yhat, y))
        return container

    def _pad_collate(batch):
        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        raise NotImplementedError()
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
        return xx_pad, yy_pad, x_lens, y_lens

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
            if self.criterion.reduction == 'none':
                loss.sum().backward()
            else:
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

        history_container = defaultdict(list)

        seen_item = 0
        for loader_ix, loader in enumerate(loaders):

            for batch_ix, batch in enumerate(loader):

                batch_loss, batch_output, batch_data, batch_target = self._fit_one_batch(
                    batch, train=True if phase == 'training' else False)
                seen_item += len(batch_target)

                # Store
                history_container['id'].append(
                    batch.get(
                        'id',
                        torch.tensor(
                            range(seen_item - len(batch_target), seen_item)))
                )  # generate new id if id attr is not available
                history_container['loss'].append(batch_loss.detach(
                ) if batch_loss.dim() != 0 else batch_loss.detach().unsqueeze(
                    dim=0))
                history_container['output'].append(batch_output.detach())
                history_container['target'].append(batch_target.detach())

                if self.params['stdout']['verbose']:
                    if self.params['stdout']['on_batch'] and (
                            batch_ix +
                            1) % self.params['stdout']['on_batch'] == 0:

                        dataset_len = sum(
                            len(loader.dataset) for loader in loaders)
                        batch_loss = batch_loss.mean().item()

                        print('\t'.join([
                            f"{phase}",
                            f"Epoch: {epoch} [{seen_item:04}/{dataset_len:04} ({100. * seen_item / dataset_len:.0f}%)]",
                            f"Batch Loss: {batch_loss:.6f}",
                        ]))

            history_container = {
                key: torch.cat(value_list)
                for key, value_list in history_container.items()
            }

        if self.params.get('log_history', None):
            self.logger.log_history(
                phase=phase,
                epoch=epoch,
                history={
                    key: val.cpu().numpy()
                    for key, val in history_container.items()
                })

        metric_container = dict()
        # Add loss to metrics
        metric_container['loss'] = history_container['loss'].mean().item()

        # Calculate metrics
        for metric_fn in self.metrics:
            metric_container[metric_fn.__name__.lower()] = metric_fn()(
                yhat=history_container['output'], y=history_container['target'])

        # Log loss and metrics
        for metric_name, metric_value in metric_container.items():
            self.logger.log_metric(metric_name=metric_name,
                                   phase=phase,
                                   epoch=epoch,
                                   metric_value=metric_value)

        return (history_container, metric_container)

    @hook(before='before_fit', after='after_fit')
    def fit(self, dataset, validation_dataset):
        if self.params['pretrained']:
            raise Exception("You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.dataset = dataset
            self.validation_dataset = validation_dataset

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
            if not self.params.get('resume', False):
                self.epoch = 0
                self.phase = 'validation'
                history_container, metric_container = self._fit_one_epoch(
                    loaders=validation_loaders,
                    epoch=self.epoch,
                    phase=self.phase)

            for self.epoch in range(self.start_epoch,
                                    self.hyperparams['epoch'] + 1):

                for self.phase in ['training', 'validation']:

                    history_container, metric_container = self._fit_one_epoch(
                        loaders=train_loaders
                        if self.phase == 'training' else validation_loaders,
                        epoch=self.epoch,
                        phase=self.phase)

                    if self.params['stdout']['verbose']:
                        n = self.params['stdout']['on_epoch']
                        if n and (self.epoch) % n == 0:

                            print('\t'.join([
                                f"{self.phase}", f"Epoch: {self.epoch}",
                                f"Loss: {metric_container['loss']:.6f}", *[
                                    f"{metric_fn.__name__.lower()}: {metric_container[metric_fn.__name__.lower()]:.6f}"
                                    for metric_fn in self.metrics
                                ]
                            ]))

                if self.epoch % self.params['log']['on_epoch'] == 0:
                    self.logger.save()

                if self.params.get('checkpoint', False) and \
                    (self.epoch % self.params['checkpoint']['on_epoch']) == 0:
                    self._save_checkpoint(epoch=self.epoch)

        except KeyboardInterrupt:
            print('Exiting from training early. Bye!')
            self.logger.stop()

    def _transform(self, dataset, batch_size, classification):
        loader = self._to_loader(dataset, training=False, batch_size=batch_size)

        transformed = []
        targets = []
        for batch_ix, batch in enumerate(loader):
            loss, output, data, target = self._fit_one_batch(batch, train=False)
            if classification:
                output = output.argmax(dim=1)
            transformed.append(output)
            targets.append(target)

        transformed = torch.cat(transformed, axis=0)
        targets = torch.cat(targets, axis=0)
        return transformed, targets

    def transform(self, dataset, batch_size=None, classification=True):
        transformed, targets = self._transform(dataset, batch_size,
                                               classification)
        return (transformed.cpu().detach().numpy(),
                targets.cpu().detach().numpy())

    def fit_transform(self, dataset, classification=True):
        self.fit(dataset=dataset)
        return self.transform(dataset=dataset, classification=classification)

    def score(self, dataset):
        raise NotImplementedError()
        transformed = self._transform(dataset)
        return transformed == dataset.get_targets()

    def to_prediction_dataframe(self, dataset, classification=True, save=True):
        predictions, targets = self.transform(dataset=dataset,
                                              classification=classification)
        prediction_dataframe = pd.DataFrame({
            'prediction': predictions.squeeze(),
            'target': targets.squeeze()
        })
        if save:
            prediction_dataframe.to_csv(self.experiment_fpath /
                                        'predictions.csv',
                                        index=False)
        return prediction_dataframe

    # Hook methods

    def before_fit(self, dataset, validation_dataset):
        pass

    def after_fit(self):
        pass

    def before_fit_one_epoch(self, loaders, phase, epoch):
        pass

    def after_fit_one_epoch(self, history_container, metric_container):
        pass

    def before_fit_one_batch(self, batch, train):
        pass

    def after_fit_one_batch(self, loss, output, data, target):
        pass
