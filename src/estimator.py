import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import pandas as pd
import collections

from model import LSTM
from dataset import IndicatorDataset

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import os


class Estimator:
    """
    """

    # todo: buraları sadece bu probleme uygun basit bir hale getir. Şu an çok generic.

    def __init__(self, dataset, model, use_cuda=True, exp_dir='../experiment', batch_size=10):

        self.model = model

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()


        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), 0.005)

        self.dataset = dataset

        self.train_dataloader = DataLoader(dataset.train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=True, collate_fn=Estimator.custom_collate_fn)
        self.valid_dataloader = DataLoader(dataset.valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=True, collate_fn=Estimator.custom_collate_fn)


        self.writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'summary'))

    # def __new__(cls, *args, **kwargs):
    #     cls.__init__(cls, *args, **kwargs)
    #     return cls

    @staticmethod
    def custom_collate_fn(batch):

        if isinstance(batch[0], np.ndarray):
            return torch.stack([torch.from_numpy(b) for b in batch], 0)

        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [Estimator.custom_collate_fn(samples) for samples in transposed]

        elif isinstance(batch[0], dict):
            return pd.DataFrame(list(batch)).to_dict('list')
        else:

            raise Exception('Update custom_collate_fn!!')

    def run_epoch(self, epoch, t):

        # Train
        tlosses = np.array([])
        taccs  =np.array([])
        for step, (tX, ty, info) in enumerate(self.train_dataloader):

            if step % 100 == 0:
                t.set_description('EPOCH : {} || STEP : {}'.format(epoch, step))

            tX, ty = Variable(tX.float(), requires_grad=False), Variable(ty.float(), requires_grad=False)



            if self.use_cuda:
                tX, ty = tX.cuda(), ty.cuda()

            toutput, tloss,  tacc = self.train_on_batch(tX, ty)

            toutput, tloss = toutput.cpu(), tloss.cpu()


            tlosses = np.append(tlosses, tloss.item())
            taccs = np.append(taccs, tacc)

        epoch_training_loss = tlosses.mean()
        epoch_training_acc = taccs.mean()


        # Validate
        voutputs, vlosses = np.array([]), np.array([])
        vaccs = np.array([])
        for i, (vX, vy, info) in enumerate(self.valid_dataloader):
            vX, vy = Variable(vX.float(), requires_grad=False), Variable(vy.float(), requires_grad=False)

            if self.use_cuda:
                vX, vy = vX.cuda(), vy.cuda()
            voutput, vloss, vacc = self.validate_on_batch(vX, vy)

            voutput, vloss = voutput.cpu(), vloss.cpu()
            voutputs = np.concatenate((voutputs, voutput.data.numpy()), axis=0) if voutputs.size else voutput.data.numpy()
            vlosses = np.append(vlosses, vloss.item())
            vaccs = np.append(vaccs, vacc)
        epoch_validation_loss = vlosses.mean()
        epoch_validation_acc = vaccs.mean()
        return epoch_training_loss , epoch_validation_loss, epoch_training_acc, epoch_validation_acc

    def train_on_batch(self, Xs, ys, train=True):
        self.optimizer.zero_grad()  # pytorch accumulates gradients.

        # forward + backward + optimize
        #self.model.hidden = self.model.init_hidden()  # detach history of initial hidden


        output = self.model(Xs)
        loss = self.criterion(output, ys)
        # print(loss.cpu().data.numpy(), np.sum(output.cpu().data.numpy()))

        out_argmax = np.argmax(output.cpu().data.numpy(), axis=1)
        ys_argmax = np.argmax(ys.cpu().data.numpy(), axis=1)
        acc = np.sum(out_argmax == ys_argmax) / out_argmax.__len__()


        if train:
            loss.backward()
            self.optimizer.step()

        # detach to not backpropagate whole lstm network
        self.model.hidden[0].detach_()
        self.model.hidden[1].detach_()

        return output, loss, acc

    def validate_on_batch(self, Xs, ys):
        self.model.eval()

        output, loss, vacc = self.train_on_batch(Xs, ys, train=False)

        self.model.train()

        return output, loss, vacc

    def predict(self, Xs):
        self.model.eval()

        # self.model.hidden = self.model.init_hidden(batch_size=1)
        pX = Variable(FloatTensor(Xs), requires_grad=False).unsqueeze(0)
        if self.use_cuda:
            pX = pX.cuda()
        output = self.model(pX)

        self.model.train()

        return output.cpu().data.numpy()


    def predict_all_validation(self):

        # Validate
        voutputs, vlosses = np.array([]), np.array([])
        vXs, vys = np.array([]), np.array([])

        names,dates = np.array([]), np.array([])
        for i, (vX, vy, extra_info) in enumerate(self.valid_dataloader):
            vX, vy = Variable(vX.float(), requires_grad=False), Variable(vy.float(), requires_grad=False)
            if self.use_cuda:
                vX, vy = vX.cuda(), vy.cuda()
            voutput, vloss = self.validate_on_batch(vX, vy)


            voutput, vloss = voutput.cpu(), vloss.cpu()
            voutputs = np.concatenate((voutputs, voutput.data.numpy()),
                                      axis=0) if voutputs.size else voutput.data.numpy()
            vlosses = np.append(vlosses, vloss.item())

            vXs = np.concatenate((vXs, vX.cpu().data.numpy()),
                                      axis=0) if vXs.size else vX.cpu().data.numpy()

            vys = np.concatenate((vys, vy.cpu().data.numpy()),
                                      axis=0) if vys.size else vy.cpu().data.numpy()

            dates = np.append(dates, extra_info['date'])
            names = np.append(names, extra_info['name'])

        epoch_validation_loss = vlosses.mean()

        return vXs, vys,  voutputs, vlosses, (dates,names)


if __name__ == "__main__":

    pass