__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import os
from collections import defaultdict

import numpy as np
import torch
from berries.logger.logger import LocalLogger
from berries.plot.plotter import Plotter
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


class BaseTrainer():

    def __init__(self, model, metrics, hyperparams, params, optimizer,
                 criterion, logger) -> None:
        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        self.hyperparams = hyperparams
        self.params = params

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
        if train and not self.model.training:
            self.model.train()
            torch.set_grad_enabled(True)

        if not train and self.model.training:
            self.model.train(False)
            torch.set_grad_enabled(False)

    def _fit_one_batch(self, batch, train):

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

    def _fit_one_epoch(self, loader, epoch=None, train=True):

        seen_item = 0
        for batch_ix, batch in enumerate(loader):
            self.before_each_batch(loader, seen_item, batch_ix, batch, epoch,
                                   train)
            loss, output, data, target = self._fit_one_batch(batch, train)
            seen_item += len(data)
            self.after_each_batch(loader, seen_item, batch_ix, batch, epoch,
                                  train, loss, output, data, target)
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

    def fit(self, dataset, validation_dataset):
        if self.params['pretrained']:
            raise Exception("-You can not use fit with --pretrained=True")
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.dataset = dataset
            self.validation_dataset = validation_dataset

            train_loader = DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0,
                                      pin_memory=self.on_cuda)

            validation_loader = DataLoader(
                dataset=validation_dataset,
                batch_size=self.validation_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=self.on_cuda)

            for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):

                self.before_each_epoch(epoch)

                for phase in ['training', 'validation']:

                    self.before_each_phase(phase)

                    metric_container = defaultdict(list)
                    for loss, output, data, target in self._fit_one_epoch(
                            loader=train_loader
                            if phase == 'training' else validation_loader,
                            epoch=epoch,
                            train=True if phase == 'training' else False):

                        # Store loss
                        metric_container['loss'].append(loss.item())

                        # Store other metrics
                        metric_container = self._calculate_metrics(
                            yhat=output, y=target, container=metric_container)

                    self._log_metrics(
                        phase=phase,
                        epoch=epoch,
                        container={
                            key: np.array(values).mean()
                            for key, values in metric_container.items()
                        })

                    self.after_each_phase(phase, metric_container)

                self.after_each_epoch(epoch)

                if epoch % self.params['log_interval'] == 0:
                    self._save_checkpoint(epoch=epoch)
                    self.logger.save()

                self._save_checkpoint(epoch=epoch)

        except KeyboardInterrupt:
            print('Exiting from training early')

    def _transform(self, dataset, batch_size=None):
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size or self.validation_batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=0,
                            pin_memory=self.on_cuda)

        transformed = []
        for loss, output, data, target in self._fit_one_epoch(loader=loader,
                                                              train=False):
            transformed.append(output)
        transformed = torch.cat(transformed, axis=0)
        return transformed

    def _fit_transform(self, dataset):
        self.fit(dataset=dataset)
        return self._transform(dataset=dataset)

    def transform(self, dataset):
        return self._transform(dataset).cpu().detach().numpy()

    def fit_transform(self, dataset):
        return self._fit_transform(dataset).cpu().detach().numpy()

    def score(self, dataset):
        raise NotImplementedError()
        transformed = self._transform(dataset)
        return transformed == dataset.get_targets()

    def stdout_items(self, batch_ix, train, epoch, seen_item, loader, loss):
        return [
            f"{'Train' if train else ' Test'}",
            f"Epoch: {epoch or 'NA'} [{seen_item:04}/{len(loader.dataset):04} ({100. * batch_ix / len(loader):.0f}%)]",
            f"Batch Loss: {loss.item():.6f}",
        ]

    def before_fit(self, *args, **kwargs):
        pass

    def after_fit(self, *args, **kwargs):
        pass

    def before_each_epoch(self, *args, **kwargs):
        pass

    def after_each_epoch(self, *args, **kwargs):
        pass

    def before_each_phase(self, *args, **kwargs):
        pass

    def after_each_phase(self, *args, **kwargs):
        pass

    def before_each_batch(self, loader, seen_item, batch_ix, batch, epoch,
                          train):
        pass

    def after_each_batch(self, loader, seen_item, batch_ix, batch, epoch, train,
                         loss, output, data, target):
        if batch_ix % self.params['stdout_interval'] == 0:
            print('\t'.join(
                self.stdout_items(batch_ix, train, epoch, seen_item, loader,
                                  loss)))
