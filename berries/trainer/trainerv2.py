__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from pathlib import Path
from berries.metric import metrics
# from ..plot import Plotter
from collections import defaultdict
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
import os
import numpy as np
from torch.utils.data import DataLoader

import functools
import operator

from abc import abstractmethod


class Meta(type):
    def __call__(cls, *args, **kwargs):
        print('start Meta.__call__')
        instance = super().__call__(*args, **kwargs)  # Child class init
        instance.initialize(kwargs.get('trainer_params'))
        print('end Meta.__call__')
        return instance


class BaseModel(nn.Module, metaclass=Meta):
    def __init__(self):
        print("Base.__init__()")
        super().__init__()  # nn.Module init

    def initialize(self, **kwargs) -> None:

        print("Base.initialize()")

        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        model.hyperparams = hyperparams
        model.params = params

        model.device = torch.device('cuda:0' if model.params['device'] ==
                                    'cuda' else 'cpu')

        model = model.to(model.device).float()
        model.criterion = criterion or MSELoss()
        model.optimizer = Adam(
            params=model.parameters(),
            lr=model.hyperparams.get('lr', 0.001),
            weight_decay=model.hyperparams.get('weight_decay', 0))

        model.metrics = metrics
        # model.logger = logger

        model.batch_size = model.hyperparams.get('batch_size', 16)

        # Initialize plotter
        # model.plotter = Plotter()

        model.resume_or_not()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
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

    def resume_or_not(self):
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
            self.load_checkpoint(epoch=last_epoch)
            print(f"Checkpoint is loaded from {last_epoch}")
        else:
            self.start_epoch = 1
            print("Starting training from epoch 1")

    def save_checkpoint(self, epoch):  # todo: move into generic model
        # Save the model if the validation loss is the best we've seen so far.
        # is_best = val_loss > best_val_loss
        # best_val_loss = max(val_loss, best_val_loss)

        self.cpt_fpath = self.experiment_fpath / 'checkpoints' / str(epoch)
        self.cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, self.cpt_fpath / 'model-optim.pth')

    def load_checkpoint(self, epoch):  # todo: move into generic model
        # load model
        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == 'cpu':
            map_location = self.device.type

        checkpoint = torch.load(self.experiment_fpath / 'checkpoints' /
                                str(epoch) / 'model-optim.pth',
                                map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
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

    def _handle_batch(self, batch):
        data = batch['data']
        target = batch['target']

        # cast data to a device
        data, target = data.to(self.device), target.to(self.device)

        return data, target

    def _forward(self, data):
        # model forward pass
        return self(data)

    def _compute_loss(self, output, targets):
        # compute loss
        loss = self.criterion(output, targets)
        return loss

    def _set_grad_enabled(self, train):
        if train and not self.training:
            self.train()
            torch.set_grad_enabled(True)

        if not train and self.training:
            self.train(False)
            torch.set_grad_enabled(False)

    def _fit_one_batch(self, batch, train):

        self._set_grad_enabled(train)

        data, target = self._handle_batch(batch)
        output = self._forward(data)
        loss = self._compute_loss(output, target)

        if train:
            # do not let pytorch accumulates gradient
            self.optimizer.zero_grad()

            # calculate gradient with backpropagation
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if self.hyperparams.get('clip', None):  # if clip is given
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               self.hyperparams['clip'])

            # distribute gradients to update weights
            self.optimizer.step()

        return loss, output, data, target

    def _fit_one_epoch(self, loader, epoch=None, train=True):

        seen_item = 0
        for batch_ix, batch in enumerate(loader):
            loss, output, data, target = self._fit_one_batch(batch, train)

            seen_item += len(data)
            if batch_ix % self.params['stdout_interval'] == 0:
                print('\t'.join((
                    f"{'Train' if train else 'Test'}",
                    f"Epoch: {epoch or 'NA'} [{seen_item}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f}%)]",
                    f"Batch Loss: {loss.item():.6f}",
                )))

            yield loss, output, data, target

    def _log_metrics(self, phase, epoch, container):
        for metric_key, metric_val in container.items():
            self.logger.log_metric(log_name=f'{phase}_{metric_key}',
                                   x=epoch,
                                   y=np.mean(metric_val))

    def _calculate_metrics(self, yhat, y, container):
        for metric_fn in self.metrics:
            container[metric_fn.__name__].append(metric_fn()(yhat, y))
        return container

    def fit(self, dataset):
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=True)

        loss_container = list()
        metric_container = defaultdict(list)
        for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):
            for loss, output, data, target in self._fit_one_epoch(
                    loader=train_loader, epoch=epoch, train=True):
                loss_container.append(loss.item())
                metric_container = self._calculate_metrics(
                    yhat=output, y=target, container=metric_container)
                self.logger.log_metric(log_name='training_loss',
                                       x=epoch,
                                       y=np.mean(loss_container))
                self._log_metrics(phase='training',
                                  epoch=epoch,
                                  container=metric_container)

            if epoch % self.params['log_interval'] == 0:
                # self.run_callbacks(epoch=epoch)
                self.save_checkpoint(epoch=epoch)
                self.logger.save()

        self.is_fitted = True

    def transform(self, dataset):
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=True)

        if self.is_fitted:

            transformed = []
            for loss, output, data, target in self._fit_one_epoch(
                    loader=loader, train=False):
                t = output.cpu().data.numpy()
                transformed.append(t)
            transformed = np.concatenate(transformed, axis=0)
            return transformed
        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset):
        self.fit(dataset=dataset)
        transformed = self.transform(dataset=dataset)
        return transformed


class CNN(BaseModel):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels, input_dim,
                 trainer_params):
        super().__init__()  # Baseclass init
        print("Child.__init__()")
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=20,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    params = {
        'project_name': 'debug',
        'experiment_name': 'mnist-float-v2',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume': False,
        'pretrained': False,
        'log_interval': 1,
        'stdout_interval': 10,
        'root': Path('./'),
    }

    hyperparams = {
        'lr': 0.001,
        'batch_size': 2048,
        'epoch': 2,
    }
    model = CNN(in_channels=1,
                out_channels=10,
                input_dim=(1, 28, 28),
                trainer_params={
                    'metrics': [metrics.Accuracy],
                    'hyperparams': hyperparams,
                    'params': params,
                    'criterion': torch.nn.CrossEntropyLoss(),
                }

                )
