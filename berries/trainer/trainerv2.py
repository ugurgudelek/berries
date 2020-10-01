import torch
from torch.nn import MSELoss
from torch.optim import Adam
import os
import numpy as np
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, metrics, hyperparams, params,
                 logger,
                 optimizer=None, criterion=None, device=None) -> None:

        self.hyperparams = hyperparams
        self.params = params

        self.device = torch.device(
            'cuda:0' if self.params['device'] == 'cuda' else 'cpu')

        self.model = model.to(self.device).float()
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(),
                                           lr=self.hyperparams.get(
                                               'lr', 0.001),
                                           weight_decay=self.hyperparams.get('weight_decay', 0))

        self.metrics = metrics
        self.logger = logger

        self.resume_or_not()

    @property
    def experiment_fpath(self):
        return self.params['root'] / 'projects' / self.params['project_name'] / self.params['experiment_name']

    @property
    def on_cuda(self):
        return self.device.type == 'cuda'

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
                raise Exception(
                    f"""
                    You do not have any checkpoint to resume.
                    If you want to start over, make sure --resume and --pretrained is False.
                    """
                )
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

    def _log_metrics(self, phase, epoch, container):
        for metric_key, metric_val in container.items():
            self.logger.log_metric(log_name=f'{phase}_{metric_key}',
                                   x=epoch,
                                   y=np.mean(metric_val))

    def fit(self, dataset):
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=True)

    def fit_transform(self, dataset):
        raise NotImplementedError()

    def transform(self, dataset):
        raise NotImplementedError()
