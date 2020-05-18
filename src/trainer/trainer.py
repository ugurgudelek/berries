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
from tqdm import tqdm

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
                                       drop_last=False,
                                       shuffle=True)  # todo: output.view(batch_size, -1) needs this!

        self.test_loader = DataLoader(self.dataset.testset,
                                      batch_size=self.hyperparams['test_batch_size'],
                                      drop_last=False,
                                      shuffle=False)

        # self.test_loader = Dataloader(self.dataset.testset,
        #                               batch_size=self.hyperparams['test_batch_size'],
        #                                         seq_len=self.hyperparams['seq_len'],
        #                               **self.loader_kwargs)

        self.experiment_fpath = Path(f'../experiments/{self.params["experiment_name"]}')
        self.experiment_fpath.mkdir(parents=True, exist_ok=True)

        # Init history to log loss or something
        self.history = History()

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



    def _validate_hyperparams(self, hyperparams):
        raise NotImplementedError()

    def _validate_params(self, params):
        raise NotImplementedError()

    def _on_epoch(self, epoch, train=True):

        loader = self.train_loader if train else self.test_loader

        self.model.train(train)  # enable or disable dropout

        logs = dict()
        logs['loss'] = list()
        seen_item = 0
        for batch_ix, (data, targets, aux) in enumerate(loader):

            data, targets = data.to(self.device), targets.to(self.device)
            aux = aux.to(self.device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # self.model.reset_states()
            # hidden_ = self.model.repackage_hidden(hidden)


            # Loss
            # todo: add loss calculations. you can look into encoder_decoder_rnntrainer.py for more info.
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
            if batch_ix % self.params['log_interval'] == 0:
                print('\t'.join((
                    f"{'Train' if train else 'Test'}",
                    f"Epoch: {epoch} [{seen_item}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f}%)]",
                    f"Batch Loss: {loss.item():.6f}",
                )))


            logs['loss'].append(loss.item())

            yield loss

            # self.plot_small_data(x=data, y=targets)

        self.history.append(phase='train' if train else 'test',
                            log_dict={'epoch': epoch,
                                      'loss': np.mean(logs['loss']),
                                      })

        self.model.train(not train)

    def fit(self):

        if self.params['pretrained']:
            raise Exception("-You can not use fit with --pretrained=True")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            # # # See what the scores are before training
            with torch.no_grad():
                for loss in self._on_epoch(train=False, epoch=0):
                    pass

            for epoch in range(self.start_epoch, self.hyperparams['epoch'] + 1):
                # Training loop
                for loss in self._on_epoch(train=True, epoch=epoch):
                    pass
                # Validation loop
                with torch.no_grad():
                    for loss in self._on_epoch(train=False, epoch=epoch):
                        pass

                if epoch % self.params['save_interval'] == 0:
                    if self.params['save_fig']:
                        # self.callbacks(epoch=epoch)
                        self.save_checkpoint(epoch=epoch)
                        self.learning_curve()


                        for i in range(len(self.dataset.train_datasets)):
                            self.plot_prediction(dataloader=DataLoader(self.dataset.train_datasets[i],
                                                          batch_size=self.hyperparams['test_batch_size'],
                                                          drop_last=False,
                                                          shuffle=False),
                                                 img=self.dataset.train_datasets[i].data.T,

                                                 name=f'prediction-{epoch}-[train{i}]')

                        self.plot_prediction(dataloader=DataLoader(self.dataset.testset,
                                                                   batch_size=self.hyperparams['test_batch_size'],
                                                                   drop_last=False,
                                                                   shuffle=False),
                                             img=self.dataset.testset.data.T,

                                             name=f'prediction-{epoch}-[test]')


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
                    'params': self.params,
                    'history': self.history},
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
        self.history = checkpoint['history']

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
        outputs = list()
        targets = list()
        with torch.no_grad():
            for batch_ix, (data, target, aux) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                aux = aux.to(self.device)
                output = self.model(data, aux)

                outputs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

        return np.concatenate(outputs), np.concatenate(targets)

    def plot_small_data(self, x, y):

            prediction = self.predict(x)
            # prediction's shape: [batch, seq, feature]

            # many output part
            # x = x[-1, :, :].cpu().numpy()
            # y = y[-1, :, :].cpu().numpy()
            # prediction = prediction[-1, :, :]
            # one output part
            x = x[:, 0, :].cpu().numpy()
            y = y[:, :].cpu().numpy()

            t_xy = self.dataset.inverse_transform(np.hstack((x, y)))
            prediction = self.dataset.inverse_transform(np.hstack((x, prediction)))[:, 1].reshape(-1, 1)

            t_xyp = np.hstack((t_xy, prediction))

            # return t_xyp[:, 0], t_xyp[:, 1], t_xyp[:, 2],  # [x, y, prediction]

            x,y,p = t_xyp[:, 0], t_xyp[:, 1], t_xyp[:, 2]
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=False)
            axs[0, 0].plot(x, label='x')
            axs[0, 0].legend()

            axs[1, 0].plot(y, label='y')
            axs[1, 0].plot(p, label='prediction')
            axs[1, 0].legend()



            plt.show()




    def callbacks(self, epoch, train=False):



        plot_container = None
        loader = DataLoader(self.dataset.trainset,
                                       batch_size=self.hyperparams['train_batch_size'],
                                       drop_last=True,
                                       shuffle=False)
        with tqdm(total=len(loader)) as pbar:
            for x, y in loader:
                prediction = self.predict(x)
                # prediction's shape: [batch, seq, feature]

                x = x[:, -1, :].cpu().numpy()
                y = y[:, -1, :].cpu().numpy()
                prediction = prediction[:, -1, :]

                t_xy = self.dataset.inverse_transform(np.hstack((x, y)))
                prediction = self.dataset.inverse_transform(np.hstack((x, prediction)))[:, 1].reshape(-1, 1)

                t_xyp = np.hstack((t_xy, prediction))
                if plot_container is None:
                    plot_container = t_xyp
                else:
                    plot_container = np.vstack((plot_container, t_xyp))
                pbar.update(x.shape[0])

        x, y, p = plot_container[:, 0], plot_container[:, 1], plot_container[:, 2]

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, squeeze=False)

        axs[0, 0].plot(x, label='x')
        axs[0, 0].legend()

        axs[1, 0].plot(y, label='y')
        axs[1, 0].legend()

        axs[2, 0].plot(p, label='prediction')
        axs[2, 0].legend()

        plt.show()


    @staticmethod
    def proba(output):
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

    def plot_prediction(self, dataloader, img, name):
        prediction, targets = self.predict_loader(dataloader=dataloader)
        indices = list(range(len(prediction)))

        fig, axes = plt.subplots(nrows=2)

        axes[0].scatter(indices, prediction, label='prediction', s=1, c='r')
        axes[0].plot(indices, targets, label='true')
        axes[0].legend()

        axes[0].set_xlim(indices[0], indices[-1])

        axes[1].imshow(img[:, indices])
        plt.suptitle('Test')

        save_dir = self.experiment_fpath / 'figures'
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'{name}.png')
        plt.close()
