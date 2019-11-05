__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pathlib import Path

from history.history import History

from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision

import os


# class TimeSeriesDataloader():
#     def __init__(self, dataset, batch_size, seq_len, **kwargs):
#         self.dataset = dataset
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.time_skip = len(self.dataset) // self.batch_size
#
#     def __iter__(self):
#         for ix in range(len(self.dataset)):
#
#             col_ix = ix % self.batch_size
#             batch_ix = ix // self.batch_size
#
#             real_ix = col_ix * self.time_skip + batch_ix * self.seq_len
#             yield self.dataset[real_ix]

# class TimeSeriesDatasetSampler(torch.utils.data.sampler.Sampler):
#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices (list, optional): a list of indices
#         num_samples (int, optional): number of samples to draw
#
#     Examples:
#         train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         sampler=ImbalancedDatasetSampler(train_dataset),
#         batch_size=args.batch_size,
#         **kwargs
#     )
#     """
#
#     def __init__(self, num_samples, time_skip):
#         self.num_samples = num_samples
#         self.time_skip = time_skip
#
#     def __iter__(self):
#         for i in range(0, self.num_samples, self.time_skip):
#             yield i
#
#     def __len__(self):
#         return self.num_samples


class RNNTrainer:
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None):
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
            cpt_path = self.experiment_fpath/'checkpoints'
            if not cpt_path.exists():
                 raise Exception("You do not have any checkpoint to resume\n if you want to start over. Make sure --resume and --pretrained is False")
            last_epoch = sorted(list(map(int, os.listdir(cpt_path))))[-1]
            self.load_checkpoint(epoch=last_epoch)
            print("=> loaded checkpoint")
        else:
            self.start_epoch = 1
            print("=> Start training from scratch")

        # Dataloader are not working with timeseries signal data.
        # TODO: check for further info.
        # self.train_loader = TimeSeriesDataloader(self.dataset.trainset,
        #                                batch_size=self.hyperparams['train_batch_size'],
        #                                          seq_len=self.hyperparams['seq_len'],
        #                                # sampler=TimeSeriesDatasetSampler(13056, 204),
        #                                **self.loader_kwargs)
        # self.test_loader = TimeSeriesDataloader(self.dataset.testset,
        #                               batch_size=self.hyperparams['test_batch_size'],
        #                                         seq_len=self.hyperparams['seq_len'],
        #                               **self.loader_kwargs)

        self.history = History()


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



    def save_checkpoint(self, epoch):
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

    def load_checkpoint(self, epoch):
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

    @staticmethod
    def get_batch(source, i, seq_len):
        seq_len = min(seq_len, len(source) - 1 - i)
        data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
        target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
        return data, target

    def _on_epoch(self, epoch, train=True):
        # Train and test differences
        with_grad_or_not = torch.enable_grad if train else torch.no_grad
        # loader = self.train_loader if train else self.test_loader
        dataset = self.dataset.trainset.batched_data if train else self.dataset.testset.batched_data
        self.model.train(train)  # enable or disable dropout

        hidden = self.model.init_hidden(self.hyperparams['train_batch_size'])

        with with_grad_or_not():

            logs = dict()
            logs['loss'] = list()
            for batch_ix, i in enumerate(range(0, len(dataset), self.hyperparams['seq_len'])):

                data, targets = self.get_batch(dataset, i, seq_len=self.hyperparams['seq_len'])
                data, targets = data.to(self.device), targets.to(self.device)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = self.model.repackage_hidden(hidden)
                hidden_ = self.model.repackage_hidden(hidden)
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.

                # Loss1: Free Running Loss
                outVal = data[0].unsqueeze(0)
                outVals = []
                hids1 = []
                for i in range(data.size(0)):
                    outVal, hidden_, hid = self.model.forward(outVal, hidden_, return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals, dim=0)
                hids1 = torch.cat(hids1, dim=0)
                loss1 = self.criterion(outSeq1.view(self.hyperparams['train_batch_size'], -1),
                                       targets.contiguous().view(self.hyperparams['train_batch_size'], -1))

                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model.forward(data, hidden, return_hiddens=True)
                loss2 = self.criterion(outSeq2.view(self.hyperparams['train_batch_size'], -1),
                                       targets.contiguous().view(self.hyperparams['train_batch_size'], -1))

                '''Loss3: Simplified Professor forcing loss'''
                loss3 = self.criterion(hids1.view(self.hyperparams['train_batch_size'], -1),
                                       hids2.view(self.hyperparams['train_batch_size'], -1).detach())

                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1 + loss2 + loss3

                if train:
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams['clip'])
                    self.optimizer.step()

                logs['loss'].append(loss.item())

                if batch_ix % self.params['log_interval'] == 0 and batch_ix > 0:
                    print('\t'.join((
                        f"{'Train' if train else 'Test'}",
                        f"Epoch: {epoch} ", #[{batch_ix * len(data)}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f} % )]
                f"Loss: {np.array(logs['loss']).mean().item():.6f}",
                # f"Proba: {self.proba(output)}",
                )))

            self.history.append(phase='train' if train else 'test',
                                log_dict={'epoch': epoch,
                                          'loss': np.mean(logs['loss']),
                                          })

            self.model.train(not train)

        return np.mean(logs['loss'])


    def fit(self):

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
                        self.generate_output(model=self.model, testset=self.dataset.testset,
                                             epoch=epoch, experiment_fpath=self.experiment_fpath, device=self.device,
                                             start_point=self.params['start_point'],
                                             recursive_start_point=self.params['recursive_start_point'],
                                             end_point=self.params['end_point'])
                        self.save_checkpoint(epoch=epoch)

                        self.learning_curve()


        except KeyboardInterrupt:
            print('Exiting from training early')

    def plot(self, epoch, train=False):
        self.generate_output(model=self.model, testset=self.dataset.trainset if train else self.dataset.testset,
                             epoch=epoch, experiment_fpath=self.experiment_fpath, device=self.device,
                             start_point=self.params['start_point'],
                             recursive_start_point=self.params['recursive_start_point'],
                             end_point=self.params['end_point'])

    @staticmethod
    def generate_output(model, testset, epoch,  experiment_fpath,device,
                        start_point=3000,
                        recursive_start_point=3500,
                        end_point=4000):

            # Turn on evaluation mode which disables dropout.
            model.eval()
            hidden = model.init_hidden(1)
            out_sequences = []
            with torch.no_grad():
                for i in range(end_point):
                    if i >= recursive_start_point:
                        out, hidden = model.forward(out, hidden)
                    else:
                        out, hidden = model.forward(testset.data[i].unsqueeze(0).unsqueeze(0).to(device), hidden)
                    out_sequences.append(out.data.cpu()[0][0].unsqueeze(0))



            out_sequences = torch.cat(out_sequences, dim=0)  # [seqLength * feature_dim]
            target = testset.scaler.inverse_transform(testset.data)[:out_sequences.shape[0]]
            out_sequences = testset.scaler.inverse_transform(out_sequences)


            seq_loss = torch.abs(out_sequences - target)
            seq_loss = (seq_loss - seq_loss.min(dim=0)[0])/(seq_loss.max(dim=0)[0] - seq_loss.min(dim=0)[0])*0.01

            # seq_loss_sum = pd.Series(seq_loss.numpy()[:,0]).rolling(window=22, center=False).mean().shift(-11).fillna(0).values.reshape(-1,1)

            plt.figure(figsize=(15, 5))
            for i in range(target.size(-1)):
                # plot target data
                plt.plot(target[:, i].numpy(), label='Target' + str(i),
                         color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                # plot 1-step predictions
                plt.plot(range(start_point, recursive_start_point), out_sequences[start_point:recursive_start_point, i].numpy(),
                         label='1-step predictions for target' + str(i),
                         color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)

                # # plot multi-step predictions
                # plt.plot(range(recursive_start_point, end_point), out_sequences[recursive_start_point:end_point, i].numpy(),
                #          label='Recursive predictions for target' + str(i),
                #          color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                # plot seq loss
                # plt.plot(seq_loss[:, i].numpy(), label='seq-loss', color='red', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                # plot seq loss
                # plt.plot(seq_loss_sum[:, i], label='seq-loss-ma', color='yellow', marker='.', linestyle='--',
                #          markersize=1.5, linewidth=1)

            plt.xlim([start_point, end_point])
            plt.xlabel('Index', fontsize=15)
            plt.ylabel('Value', fontsize=15)
            plt.title('Time-series Prediction on ' + experiment_fpath.name, fontsize=18, fontweight='bold')
            plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            plt.text(start_point + 10, target.min(), 'Epoch: ' + str(epoch), fontsize=15)
            save_dir = experiment_fpath/'figures'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir.joinpath('fig_epoch' + str(epoch)).with_suffix('.png'))
            plt.close()
            return out_sequences


    @staticmethod
    def proba(output):
        return torch.nn.functional.softmax(output.detach(), dim=1).cpu().numpy()


    def predict(self, data=None):
        if data is None:
            data = self.dataset.testset.data

        self.model.hidden = None
        pred = self.model(data)
        return pred.detach().numpy()


    def predict_log_proba(self):
        pass


    def predict_proba(self):
        pass

    # def score(self, data=None, targets=None, kind='accuracy'):
    #     if data is None:
    #         data = self.dataset.testset.data
    #     if targets is None:
    #         targets = self.dataset.testset.targets
    #
    #     output = self.model(data)
    #     if kind == 'accuracy':
    #         return self.accuracy(output, targets)
    #     raise RuntimeError(f"{kind} is not recognized in available kinds.")
