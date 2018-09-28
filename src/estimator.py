import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import pandas as pd
import collections

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import os
import time

from functools import partial
from collections import defaultdict


class Estimator:
    """
    """


    def __init__(self, model, device, exp_dir='../experiment'):

        self.model = model

        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), 0.005)

        self.writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'summary'))


        # ============== FIT - PREDICT - VALIDATE METHODS ====================
        self._train_on_batch = partial(self._on_batch, train=True)
        self._validate_on_batch = partial(self._on_batch, train=False)


        # Predict given xs at once
        self.predict = partial(self._on_batch, ys=None, train=False)

        # Fit model for given dataloader xs and ys
        self.fit = partial(self._on_dataloader, train=True)

        # Validate given xs and ys at once - same as _validate_on_batch
        # todo: make validation work on dataloader.
        self.validate = self._validate_on_batch

    def _on_dataloader(self, dataloader, train):
        """Never call this function directly!"""

        losses = np.array([])
        for step, (xs, ys, info) in enumerate(dataloader):
            xs = Variable(xs.float(), requires_grad=False)
            ys = Variable(ys.float(), requires_grad=False)

            if train:
                output,loss = self._train_on_batch(xs, ys)
            else:
                output,loss = self._validate_on_batch(xs, ys)

            losses = np.append(losses, loss.item())

        return losses.mean()


    def _on_batch(self, xs, ys, train):
        """Never call this function directly!"""
        # if this call for validation or test, change model mode to evaluation
        # this is necessary because dropout and batch normalization should behave differently on evaluation mode
        if not train:
            self.model.eval()

        self.optimizer.zero_grad()  # pytorch accumulates gradients.

        xs = xs.to(self.device)
        # forward
        output = self.model(xs)
        loss = None

        if ys is not None:
            ys = ys.to(self.device)

            # loss
            loss = self.criterion(output, ys)

            if train:

                # backward
                loss.backward(retain_graph=True)
                #optimize
                self.optimizer.step()

        # detach first hiddens of previous iteration
        if self.model.name == 'LSTM':
            self.model.detach()

        # if this call for validation or test, model mode has changed to eval before,
        # now we need to revert this behaviour
        if not train:
            self.model.train()

        return output, loss



    # def run_epoch(self, epoch, t):
    #
    #     # Train
    #     tlosses = np.array([])
    #     taccs  =np.array([])
    #     for step, (tX, ty, info) in enumerate(self.train_dataloader):
    #
    #         if step % 100 == 0:
    #             t.set_description('EPOCH : {} || STEP : {}'.format(epoch, step))
    #
    #         tX, ty = Variable(tX.float(), requires_grad=False), Variable(ty.float(), requires_grad=False)
    #
    #         if self.use_cuda:
    #             tX, ty = tX.cuda(), ty.cuda()
    #
    #         toutput, tloss,  tacc = self.train_on_batch(tX, ty)
    #
    #         toutput, tloss = toutput.cpu(), tloss.cpu()
    #
    #
    #         tlosses = np.append(tlosses, tloss.item())
    #         taccs = np.append(taccs, tacc)
    #
    #     epoch_training_loss = tlosses.mean()
    #     epoch_training_acc = taccs.mean()
    #
    #
    #     # Validate
    #     voutputs, vlosses = np.array([]), np.array([])
    #     vaccs = np.array([])
    #     for i, (vX, vy, info) in enumerate(self.valid_dataloader):
    #         vX, vy = Variable(vX.float(), requires_grad=False), Variable(vy.float(), requires_grad=False)
    #
    #         if self.use_cuda:
    #             vX, vy = vX.cuda(), vy.cuda()
    #         voutput, vloss, vacc = self.validate_on_batch(vX, vy)
    #
    #         voutput, vloss = voutput.cpu(), vloss.cpu()
    #         voutputs = np.concatenate((voutputs, voutput.data.numpy()), axis=0) if voutputs.size else voutput.data.numpy()
    #         vlosses = np.append(vlosses, vloss.item())
    #         vaccs = np.append(vaccs, vacc)
    #     epoch_validation_loss = vlosses.mean()
    #     epoch_validation_acc = vaccs.mean()
    #     return epoch_training_loss , epoch_validation_loss, epoch_training_acc, epoch_validation_acc

    # def train_on_batch(self, Xs, ys, train=True):
    #     self.optimizer.zero_grad()  # pytorch accumulates gradients.
    #
    #     # forward + backward + optimize
    #     #self.model.hidden = self.model.init_hidden()  # detach history of initial hidden
    #
    #
    #     output = self.model(Xs)
    #     loss = self.criterion(output, ys)
    #     # print(loss.cpu().data.numpy(), np.sum(output.cpu().data.numpy()))
    #
    #     out_argmax = np.argmax(output.cpu().data.numpy(), axis=1)
    #     ys_argmax = np.argmax(ys.cpu().data.numpy(), axis=1)
    #     acc = np.sum(out_argmax == ys_argmax) / out_argmax.__len__()
    #
    #
    #     if train:
    #         loss.backward(retain_graph=True)
    #         self.optimizer.step()
    #
    #     # detach to not backpropagate whole lstm network
    #     # todo: actually below lines should be like self.model.hidden[0].detach_() aka inplace version. but not it working okey..
    #     self.model.hidden[0].detach()
    #     self.model.hidden[1].detach()
    #
    #     return output, loss, acc

    # def validate_on_batch(self, Xs, ys):
    #     self.model.eval()
    #
    #     output, loss, vacc = self.train_on_batch(Xs, ys, train=False)
    #
    #     self.model.train()
    #
    #     return output, loss, vacc

    # def predict(self, Xs):
    #     self.model.eval()
    #
    #     # self.model.hidden = self.model.init_hidden(batch_size=1)
    #     pX = Variable(torch.FloatTensor(Xs), requires_grad=False).unsqueeze(0)
    #     if self.use_cuda:
    #         pX = pX.cuda()
    #     output = self.model(pX)
    #
    #     self.model.train()
    #
    #     return output.cpu().data.numpy()


    # def predict_all_validation(self):
    #
    #     # Validate
    #     voutputs, vlosses = np.array([]), np.array([])
    #     vXs, vys = np.array([]), np.array([])
    #
    #     names,dates = [],[]
    #     for i, (vX, vy, extra_info) in enumerate(self.valid_dataloader):
    #         vX, vy = Variable(vX.float(), requires_grad=False), Variable(vy.float(), requires_grad=False)
    #         if self.use_cuda:
    #             vX, vy = vX.cuda(), vy.cuda()
    #         voutput, vloss, vacc = self.validate_on_batch(vX, vy)
    #
    #
    #         voutput, vloss = voutput.cpu(), vloss.cpu()
    #         voutputs = np.concatenate((voutputs, voutput.data.numpy()),
    #                                   axis=0) if voutputs.size else voutput.data.numpy()
    #         vlosses = np.append(vlosses, vloss.item())
    #
    #         vXs = np.concatenate((vXs, vX.cpu().data.numpy()),
    #                                   axis=0) if vXs.size else vX.cpu().data.numpy()
    #
    #         vys = np.concatenate((vys, vy.cpu().data.numpy()),
    #                                   axis=0) if vys.size else vy.cpu().data.numpy()
    #
    #         dates += extra_info['date']
    #         names += extra_info['name']
    #
    #     epoch_validation_loss = vlosses.mean()
    #
    #     return vXs, vys,  voutputs, vlosses, (dates,names)


if __name__ == "__main__":

    pass
