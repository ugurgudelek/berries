__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import os
from pathlib import Path
from collections import defaultdict
from typing import Iterable
import functools
import copy

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


class NodeTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def __init__(self, data):
        pass

    def append(self, data):
        _data = data.data.detach().to(self.device)
        self.data = torch.cat([self.data, _data])


class Container:
    def __init__(self, keys):
        self.keys = keys
        self._container = {key: NodeTensor([]) for key in keys}

    def __getattr__(self, key):
        return self._container[key]

    def __setitem__(self, key, data):
        self._container[key] = NodeTensor([data])

    def __getitem__(self, key):
        return self._container[key]

    def items(self):
        return self._container.items()

    def __str__(self) -> str:
        return str(self._container)


class BaseTrainer:
    def __init__(self, model, metrics, params, optimizer, scheduler, criterion, logger, device) -> None:

        self.params = params
        self.logger = logger
        if self.params["id"] is None and self.logger is not None:
            self.params["id"] = self.logger.run.name

        self.device = device

        self.model = model.to(self.device).float()
        self.optimizer = optimizer or Adam(
            params=self.model.parameters(),
            lr=self.params.get("lr", 0.001),
            weight_decay=self.params.get("weight_decay", 0),
        )

        self.scheduler = scheduler
        self.criterion = criterion or MSELoss(reduction="none")

        self.metrics = metrics
        for metric_fn in self.metrics:
            metric_fn.to(self.device)

        self.batch_size = self.params.get("batch_size", 16)
        self.validation_batch_size = self.params.get("validation_batch_size", 16)

        self._resume_or_not()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    @property
    def experiment_fpath(self):
        return Path(self.params["root"]) / "projects" / self.params["project"] / self.params["id"]

    @property
    def on_cuda(self):
        return self.device.type == "cuda"

    def __repr__(self):
        return f"BaseTrainer(model={self.model},\
                 metrics={self.metrics},\
                 params={self.params},\
                 optimizer={self.optimizer},\
                 criterion={self.criterion},\
                 logger={self.logger},\
                 scheduler={self.scheduler})"

    def _resume_or_not(self):
        # Resume or not
        if self.params["resume"]:

            cpt_path = self.experiment_fpath / "checkpoints"
            if not cpt_path.exists():
                raise Exception(
                    f"""
                    You do not have any checkpoint to resume.
                    If you want to start over, make sure --resume and --pretrained is False.
                    """
                )
            # todo: change with Path
            last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]
            self._load_checkpoint_from_epoch(epoch=last_epoch)
            print(f"Checkpoint is loaded from {last_epoch}")
        elif self.params["pretrained"]:
            self._load_checkpoint_from_path(path=self.params["pretrained_path"])
        else:
            self.start_epoch = 1
            self.epoch = 1

            self.best_checkpoint_metric = -np.inf if self.params["checkpoint_trigger"] == "increase" else np.inf

    def _get_last_checkpoint_path(self):
        cpt_path = self.experiment_fpath / "checkpoints"
        last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]
        path = cpt_path / f"{last_epoch}/model-optim.pth"
        return path

    def _get_best_checkpoint_path(self):
        return self.experiment_fpath / "checkpoints/best/model-optim.pth"

    def _save_checkpoint(self, path_posix=None):  # todo: move into generic model

        path_posix = path_posix or str(self.epoch)
        cpt_fpath = self.experiment_fpath / "checkpoints" / path_posix
        cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save(
            {
                "epoch": self.epoch,
                "best_checkpoint_metric": self.best_checkpoint_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            cpt_fpath / "model-optim.pth",
        )

    def _load_checkpoint_from_path(self, path):
        # load model
        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == "cpu":
            map_location = self.device.type

        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.epoch = checkpoint["epoch"]

        try:  # to be able to backward compatible
            self.best_checkpoint_metric = checkpoint["best_checkpoint_metric"]
        except:
            pass
        # self.params = checkpoint['params']
        # self.params = checkpoint['params']
        # self.history = checkpoint['history']

        if self.device.type == "cuda":
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def _load_checkpoint_from_epoch(self, epoch):  # todo: move into generic model

        model_path = self.experiment_fpath / "checkpoints" / str(epoch) / "model-optim.pth"
        self._load_checkpoint_from_path(path=model_path)

    def handle_batch(self, batch):
        data = batch["data"]
        target = batch["target"]

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
            batch_size=batch_size or (self.batch_size if training else self.validation_batch_size),
            shuffle=self.params.get("train_shuffle", training),
            drop_last=self.params.get("drop_last", False),
            num_workers=0,
            pin_memory=self.on_cuda,
        )

    @staticmethod
    def _make_iterable_if_not(item):
        return item if isinstance(item, Iterable) else (item,)

    @hook(before="before_fit_one_batch", after="after_fit_one_batch")
    def _fit_one_batch(self, batch, train):
        """All training steps are implemented here.
        This function is the core of handling model - actual training loop.
        """

        self._set_grad_enabled(train)

        data, target = self.handle_batch(batch)

        # do not let pytorch accumulates gradient
        self.optimizer.zero_grad()

        # track history if only in train
        with torch.set_grad_enabled(train):
            output = self.forward(data)
            loss = self.compute_loss(output, target)

            if train:

                # calculate gradient with backpropagation
                if self.criterion.reduction == "none":
                    loss.mean().backward()
                else:
                    loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if self.params.get("clip", None):  # if clip is given
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params["clip"])

                # distribute gradients to update weights
                self.optimizer.step()

        return loss, output, data, target

    @hook(before="before_fit_one_epoch", after="after_fit_one_epoch")
    def _fit_one_epoch(self, loaders, phase, epoch):

        if not isinstance(loaders, Iterable):
            raise Exception("loaders must be an iterable")

        # history = Container(keys=['_id', 'loss', 'output', 'target'])

        metric_container = Container(
            keys=[
                "loss",
                *[metric.__class__.__name__.lower() for metric in self.metrics],
            ]
        )

        self.model.train(phase == "training")

        self.num_seen_sample = 0
        for loader_ix, loader in enumerate(loaders):

            for batch_ix, batch in enumerate(loader):

                (
                    batch_loss,
                    batch_output,
                    batch_data,
                    batch_target,
                ) = self._fit_one_batch(batch, train=True if phase == "training" else False)
                self.num_seen_sample += len(batch_target)

                # Store
                # generate new id if id attr is not available
                # _id = batch.get('id', torch.tensor(range(self.num_seen_sample - len(batch_target), self.num_seen_sample))) #yapf:disable
                # if reduction != 'none' batch_loss has zero-dim, so expand dims if this is the case
                _batch_loss = batch_loss if batch_loss.dim() != 0 else batch_loss.unsqueeze(dim=0)  # yapf:disable

                self.logger.log(
                    {
                        "loss": _batch_loss.detach().mean().item(),
                        "batch_ix": batch_ix,
                        "epoch": epoch,
                        "loader_ix": loader_ix,
                        "phase": phase,
                    }
                )

                # Update metrics
                metric_container["loss"].append(_batch_loss.detach())
                for metric_fn in self.metrics:
                    name = metric_fn.__class__.__name__.lower()
                    metric = metric_fn(batch_output.detach(), batch_target.detach())
                    metric_container[name].append(metric.unsqueeze(dim=0))

                # # Calculate metrics
                # for metric_fn in self.metrics:
                #     name = metric_fn.__class__.__name__.lower()
                #     metric = metric_fn(batch_output.detach(), batch_target.detach())
                #     self.logger.log(
                #         {f"{phase}/{name}": metric, "loader_ix": loader_ix, "batch_ix": batch_ix}, commit=False
                #     )
                # self.logger.log(
                #     {f"{phase}/loss": _batch_loss.mean(), "loader_ix": loader_ix, "batch_ix": batch_ix}, commit=True
                # )

                # history._id.append(_id)
                # history.loss.append(_batch_loss)
                # history.output.append(batch_output)
                # history.target.append(batch_target)

                if self.params.get("stdout_verbose", True):
                    if "stdout_on_batch" in self.params:
                        if (batch_ix + 1) % self.params["stdout_on_batch"] == 0:  # yapf:disable
                            dataset_len = sum(len(loader.dataset) for loader in loaders)  # yapf:disable
                            batch_loss = batch_loss.mean().item()

                            print(
                                "\t".join(
                                    [
                                        f"{phase}",
                                        f"Epoch: {epoch} [{self.num_seen_sample:04}/{dataset_len:04} ({100. * self.num_seen_sample / dataset_len:.0f}%)]",
                                        f"Batch Loss: {batch_loss:.6f}",
                                    ]
                                )
                            )

        # Compute metrics
        metric_container["loss"] = metric_container["loss"].mean()
        for metric_fn in self.metrics:
            metric_container[metric_fn.__class__.__name__.lower()] = metric_fn.compute()

        # Log metrics
        self.logger.log({f"{phase}/loss": metric_container["loss"], "epoch": epoch}, commit=False)
        self.logger.log({f"{phase}/{key}": val.item() for key, val in metric_container.items()} | {"epoch": epoch}, commit=True)

        if phase == "training":
            if self.scheduler is not None:
                self.scheduler.step()

        # Print stdout for every 'on_epoch'
        if self.params.get("stdout_verbose", True):
            if self.epoch % self.params["stdout_on_epoch"] == 0:
                print(
                    "\t".join(
                        [
                            f"{self.phase}",
                            f"Epoch: {self.epoch}",
                            f"Loss: {self.logger.run.summary[f'{phase}/loss']:.6f}",
                            *[f"{metric_fn.__class__.__name__.lower()}: {self.logger.run.summary[f'{phase}/{metric_fn.__class__.__name__.lower()}']:.6f}" for metric_fn in self.metrics],
                        ]
                    )
                )  # yapf:disable

        # Log history if requested
        # if (
        #     "history" in self.params["log"] and self.params["log"]["history"]
        # ):  # yapf:disable
        #     self.logger.log_history(
        #         phase=self.phase,
        #         epoch=self.epoch,
        #         history={key: tensor.numpy() for key, tensor in history.items()},
        #     )

        return metric_container

    @hook(before="before_fit", after="after_fit")
    def fit(self, training_dataset, validation_dataset):
        if self.params["pretrained"]:
            raise Exception("You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            self.training_dataset = training_dataset
            self.validation_dataset = validation_dataset

            # Support for list of datasets
            train_loaders = [self._to_loader(dataset=d, training=True) for d in self._make_iterable_if_not(training_dataset)]
            validation_loaders = [self._to_loader(dataset=d, training=False) for d in self._make_iterable_if_not(validation_dataset)]

            # run 1 epoch before training to watch untrained model performance
            if self.params.get("validate_epoch_0", False) and not self.params.get("resume", False):

                self.epoch = 0
                self.phase = "validation"
                metric_container = self._fit_one_epoch(loaders=validation_loaders, epoch=self.epoch, phase=self.phase)

            for self.epoch in range(self.start_epoch, self.params["epoch"] + 1):

                for self.phase in ["training", "validation"]:

                    metric_container = self._fit_one_epoch(
                        loaders=train_loaders if self.phase == "training" else validation_loaders,
                        epoch=self.epoch,
                        phase=self.phase,
                    )

                # Save checkpoint if the model is best
                if "checkpoint_metric" in self.params:
                    checkpoint_metric = metric_container[self.params["checkpoint_metric"]].item()  # yapf:disable

                    trigger = self.params.get("checkpoint_trigger", "increase") == "increase"
                    if trigger == (checkpoint_metric > self.best_checkpoint_metric):  # yapf:disable
                        print(f"Best:{checkpoint_metric} | Last:{self.best_checkpoint_metric}")
                        self.best_checkpoint_metric = checkpoint_metric
                        self._save_checkpoint(path_posix="best")
                        self.logger.log_model(path=self._get_best_checkpoint_path(), name=f"model_best")

                if "checkpoint_on_epoch" in self.params:
                    if (self.epoch % self.params["checkpoint_on_epoch"]) == 0:
                        self._save_checkpoint()

                # Save logs for every 'on_epoch'
                # if self.epoch % self.params["log"]["on_epoch"] == 0:
                #     self.logger.save()

        except KeyboardInterrupt:
            print("Exiting from training early. Bye!")
            self.logger.stop()

    @torch.no_grad()
    def transform(self, dataset, batch_size=None):
        loader = self._to_loader(dataset, training=False, batch_size=batch_size)

        transformed = []
        targets = []
        for batch_ix, batch in enumerate(loader):
            loss, output, data, target = self._fit_one_batch(batch, train=False)
            transformed.append(output.detach())
            targets.append(target.detach())

        transformed = torch.cat(transformed, axis=0)
        targets = torch.cat(targets, axis=0)
        return transformed, targets

    def fit_transform(self, dataset, classification=True):
        self.fit(dataset=dataset)
        return self.transform(dataset=dataset, classification=classification)

    def _transform_iter(self, dataset, batch_size, classification):
        loader = self._to_loader(dataset, training=False, batch_size=batch_size)
        for batch_ix, batch in enumerate(loader):
            loss, output, data, target = self._fit_one_batch(batch, train=False)
            if classification:
                output = output.argmax(dim=1)
            yield output.detach(), target.detach()

    def transform_iter(self, dataset, batch_size=None, classification=True):
        for transformed, targets in self._transform_iter(dataset, batch_size, classification):
            yield transformed.cpu().numpy(), targets.cpu().numpy()

    def score(self, dataset, batch_size=None, classification=True):
        transformed, targets = self._transform(dataset, batch_size, classification)

        metric_container = Container(keys=[*[metric.__class__.__name__.lower() for metric in self.metrics]])

        # Calculate metrics
        for metric_fn in self.metrics:
            metric_container[metric_fn.__class__.__name__.lower()] = metric_fn(transformed, targets)  # yapf:disable

        return metric_container, (
            transformed.cpu().detach().numpy(),
            targets.cpu().detach().numpy(),
        )

    def to_prediction_dataframe(self, dataset, classification=True, save=None):
        predictions, targets = self.transform(dataset=dataset, classification=classification)
        prediction_dataframe = pd.DataFrame({"prediction": predictions.squeeze(), "target": targets.squeeze()})

        if save == True:
            prediction_dataframe.to_csv(self.experiment_fpath / "predictions.csv", index=False)
        elif isinstance(save, Path) or isinstance(save, str):
            prediction_dataframe.to_csv(save, index=False)
        return prediction_dataframe

    # Hook methods

    def before_fit(self, training_dataset, validation_dataset):
        pass

    def after_fit(self):
        pass

    def before_fit_one_epoch(self, loaders, phase, epoch):
        pass

    def after_fit_one_epoch(self, metric_container):
        pass

    def before_fit_one_batch(self, batch, train):
        pass

    def after_fit_one_batch(self, loss, output, data, target):
        pass
