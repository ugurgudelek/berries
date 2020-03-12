# -*- coding: utf-8 -*-
# @Time   : 3/12/2020 12:03 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : rnntrainer.py

import torch

import numpy as np
import matplotlib.pyplot as plt

from trainer.trainer import Trainer


class RNNTrainer(Trainer):  # add base generic trainer class
    def __init__(self, model, dataset, hyperparams, params, optimizer=None, criterion=None):
        super().__init__(model, dataset, hyperparams, params, optimizer=optimizer, criterion=criterion)

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


    @staticmethod
    def get_batch(source, i, seq_len):
        """
        Custom batch method.
        Dataloader are not working with timeseries signal data.
        # todo: check for further info.
        """
        seq_len = min(seq_len, len(source) - 1 - i)
        data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
        target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
        return data, target

    def _on_epoch(self, epoch, train=True):
        # todo: reimplement this function

        # Disable gradient calculations if validation or test period is active.
        with_grad_or_not = torch.enable_grad if train else torch.no_grad
        # loader = self.train_loader if train else self.test_loader
        dataset = self.dataset.trainset.batched_data if train else self.dataset.testset.batched_data

        hidden = self.model.init_hidden(self.hyperparams['train_batch_size'])

        self.model.train(train)  # enable or disable dropout
        with with_grad_or_not():

            logs = dict()
            logs['loss'] = list()
            for batch_ix, i in enumerate(range(0, len(dataset), self.hyperparams['seq_len'])):

                data, targets = self.get_batch(dataset, i, seq_len=self.hyperparams['seq_len'])
                data, targets = data.to(self.device), targets.to(self.device)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                self.model.reset_states()
                # hidden_ = self.model.repackage_hidden(hidden)
                self.optimizer.zero_grad()  # Pytorch accumulates gradients.

                # Loss
                # todo: add loss calculations. you can look into encoder_decoder_rnntrainer.py for more info.
                output = self.model(data)
                loss = self.criterion(output, targets)


                # # Loss1: Free Running Loss
                # outVal = data[0].unsqueeze(0)
                # outVals = []
                # hids1 = []
                # for i in range(data.size(0)):
                #     outVal, hidden_, hid = self.model.forward(outVal, hidden_, return_hiddens=True)
                #     outVals.append(outVal)
                #     hids1.append(hid)
                # outSeq1 = torch.cat(outVals, dim=0)
                # hids1 = torch.cat(hids1, dim=0)
                # loss1 = self.criterion(outSeq1.view(self.hyperparams['train_batch_size'], -1),
                #                        targets.contiguous().view(self.hyperparams['train_batch_size'], -1))
                #
                # '''Loss2: Teacher forcing loss'''
                # outSeq2, hidden, hids2 = self.model.forward(data, hidden, return_hiddens=True)
                # loss2 = self.criterion(outSeq2.view(self.hyperparams['train_batch_size'], -1),
                #                        targets.contiguous().view(self.hyperparams['train_batch_size'], -1))
                #
                # '''Loss3: Simplified Professor forcing loss'''
                # loss3 = self.criterion(hids1.view(self.hyperparams['train_batch_size'], -1),
                #                        hids2.view(self.hyperparams['train_batch_size'], -1).detach())
                #
                # '''Total loss = Loss1+Loss2+Loss3'''
                # loss = loss1 + loss2 + loss3

                if train:
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams['clip'])
                    self.optimizer.step()

                logs['loss'].append(loss.item())

                if batch_ix % self.params['log_interval'] == 0 and batch_ix > 0:
                    print('\t'.join((
                        f"{'Train' if train else 'Test'}",
                        f"Epoch: {epoch} ",
                        # [{batch_ix * len(data)}/{len(loader.dataset)} ({100. * batch_ix / len(loader):.0f} % )]
                        f"Loss: {np.array(logs['loss']).mean().item():.6f}",
                        # f"Proba: {self.proba(output)}",
                    )))

            self.history.append(phase='train' if train else 'test',
                                log_dict={'epoch': epoch,
                                          'loss': np.mean(logs['loss']),
                                          })

        self.model.train(not train)

        return np.mean(logs['loss'])


    def callbacks(self, epoch, train=False):
        self.plot(epoch=epoch, train=train)

    def plot(self, epoch, train=False):  # todo: look again
        self.generate_output(model=self.model, testset=self.dataset.trainset if train else self.dataset.testset,
                             epoch=epoch, experiment_fpath=self.experiment_fpath, device=self.device,
                             start_point=self.params['start_point'],
                             recursive_start_point=self.params['recursive_start_point'],
                             end_point=self.params['end_point'])

    @staticmethod
    def generate_output(model, testset, epoch, experiment_fpath, device,
                        start_point=3000,
                        recursive_start_point=3500,
                        end_point=4000):  # todo: look again

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
        seq_loss = (seq_loss - seq_loss.min(dim=0)[0]) / (seq_loss.max(dim=0)[0] - seq_loss.min(dim=0)[0]) * 0.01

        # seq_loss_sum = pd.Series(seq_loss.numpy()[:,0]).rolling(window=22, center=False).mean().shift(-11).fillna(0).values.reshape(-1,1)

        plt.figure(figsize=(15, 5))
        for i in range(target.size(-1)):
            # plot target data
            plt.plot(target[:, i].numpy(), label='Target' + str(i),
                     color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            # plot 1-step predictions
            plt.plot(range(start_point, recursive_start_point),
                     out_sequences[start_point:recursive_start_point, i].numpy(),
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
        save_dir = experiment_fpath / 'figures'
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir.joinpath('fig_epoch' + str(epoch)).with_suffix('.png'))
        plt.close()
        return out_sequences
