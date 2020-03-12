__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.nn import MSELoss
from torch.optim import Adam

import os
from pathlib import Path

import matplotlib.pyplot as plt

from history.history import History

class Trainer:
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None):
        # self._validate_hyperparams(hyperparams)
        # self._validate_params(params)

        self.hyperparams = hyperparams
        self.params = params
        self.device = torch.device('cuda:0' if self.params['device'] == 'cuda' else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion or MSELoss()
        self.optimizer = optimizer or Adam(params=model.parameters(),
                                           lr=self.hyperparams['lr'],
                                           weight_decay=self.hyperparams['weight_decay'])
        # self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        # self.loader_kwargs = {'drop_last': False, 'shuffle': False}

        self.experiment_fpath = Path(f'../experiments/{self.params["experiment_name"]}')
        self.experiment_fpath.mkdir(parents=True, exist_ok=True)

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

        self.history = History()

    def callbacks(self, epoch, train=False):
        raise NotImplementedError()

    def _validate_hyperparams(self, hyperparams):
        raise NotImplementedError()

    def _validate_params(self, params):
        raise NotImplementedError()

    def _on_epoch(self, epoch, train=True):
        raise NotImplementedError()

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
                print(f'Train| end of epoch {epoch:3d} | | {train_loss:5.4f} |')
                val_loss = self._on_epoch(train=False, epoch=epoch)
                print(f'Val | end of epoch {epoch:3d} | | {val_loss:5.4f} |')


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

        self.cpt_fpath = self.experiment_fpath/'checkpoints'/str(epoch)
        self.cpt_fpath.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'hyperparams': self.hyperparams,
                    'params': self.params},
                   self.cpt_fpath/'model-optim.pth')

    def load_checkpoint(self, epoch):  # todo: move into generic model
        # load model
        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == 'cpu':
            map_location = self.device.type

        checkpoint = torch.load(self.experiment_fpath/'checkpoints'/str(epoch)/'model-optim.pth',
                                map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']+1
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
    def proba(output):   # todo: move into trainer
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
        plt.savefig(save_dir/'lr.png')
        plt.close()

