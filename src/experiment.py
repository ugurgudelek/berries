"""
Ugur Gudelek
run
ugurgudelek
08-Mar-18
finance-cnn
"""
from dataset import IndicatorDataset

from config import Config

import torch

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

import torch.nn.functional as F

import time

from history import History
from checkpoint import Checkpoint
from estimator import Estimator
from visualize import Visualizer

from dataset import *

from tqdm import tqdm, trange

tqdm.monitor_interval = 0

import io
import imageio


# todo: add logger
# todo: add reporter
# todo: plot and save graph

# import torch
# from torch.autograd import Variable
# from torch import nn
# from torch.utils.data import DataLoader
# from torch import optim
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os
# import dill  # dill extends python’s pickle module for serializing and de-serializing python objects
# import shutil  # high level os functionality
#
# import gc
#
# from collections import defaultdict

# if we seed random func, they will generate same output everytime.


class Experiment:
    """

    """

    def __init__(self, config, dataset, estimator, history, visualizer, epoch):
        self.config = config
        self.dataset = dataset
        self.estimator = estimator
        self.history = history
        self.visualizer = visualizer
        self.epoch = epoch

        if config.RANDOM_SEED is not None:
            self.make_deterministic(seed=config.RANDOM_SEED, use_cuda=config.USE_CUDA)

    def make_deterministic(self, seed, use_cuda):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        print('randomness test: numpy: {} || torch: {}'.format(np.random.random(1), torch.rand(1)))


    @classmethod
    def start_over(cls, config):

        dataset_cls = get_dataset_cls_from_name(name=config.DATASET_NAME)
        dataset = dataset_cls(**config.DATASET_ARGS)

        estimator = Estimator(dataset=dataset,
                              model_args=config.MODEL_ARGS,
                              dataloader_args=config.DATALOADER_ARGS,
                              criterion_args=config.CRITERION_ARGS,
                              optimizer_args=config.OPTIMIZER_ARGS,
                              use_cuda=config.USE_CUDA,
                              writer_path=os.path.join(config.EXPERIMENT_DIR, 'tensorboard'))

        history = History(config.EPOCH_SIZE, config.STORAGE_NAMES)
        visualizer = Visualizer()

        epoch = 0

        return Experiment(config, dataset, estimator, history, visualizer, epoch)

    @classmethod
    def resume(cls, experiment_path, config):


        experiment = Experiment.start_over(config)

        ckpt_path = Checkpoint.get_latest_checkpoint(experiment_path)

        ckpt = Checkpoint.load(ckpt_path)
        # if config.USE_CUDA:
        #     ckpt.model = ckpt.model.cuda()
        # else:
        #     ckpt.model = ckpt.model.cpu()

        experiment.estimator.model.load_state_dict(ckpt.model_state_dict)


        experiment.estimator.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        experiment.epoch = ckpt.epoch + 1
        # experiment.history = ckpt.history

        # todo: improve with more proper way
        # experiment.visualizer.container['tloss'] = experiment.history.container[epoch]['train']['loss']
        # experiment.visualizer.container['vloss'] = experiment.history.container[epoch]['valid']['loss']

        return experiment

    def prediction_to_csv(self, save_path):
        vXs, vys, vpreds, vlosses, (dates,names) = self.estimator.predict_all_validation()

        valid_dataset = self.estimator.dataset.valid_dataset.dataset

        result_df = pd.concat((pd.DataFrame(vpreds, columns=['psell', 'pbuy', 'phold']),
                               pd.DataFrame(vys, columns=['rsell', 'rbuy', 'rhold']),
                                pd.Series(dates, name='date'),
                                pd.Series(names, name='name')), axis=1)

        result_df = pd.merge(result_df, valid_dataset[['date', 'name', 'raw_adjusted_close']], on=['date', 'name'])
        result_df = result_df.drop_duplicates(subset=['date', 'name'])

        result_df.to_csv(os.path.join(save_path, 'prediction_results.csv'), index=False)

    def do(self):

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # plt.show(block=False)
        with trange(self.epoch, self.config.EPOCH_SIZE) as t:
            for self.epoch in t:
                # t.set_description('EPOCH %i' % self.epoch)
                # t.set_postfix(loss=2.5, gen=1.1, str='h',
                #               lst=[1, 2])

                # print('epoch : {}'.format(self.epoch))

                # Estimate - Train & Validate
                (toutputs, tloss, voutputs, vloss) = self.estimator.run_epoch(self.epoch, t)

                # Checkpoint
                Checkpoint(model_state_dict=self.estimator.model.state_dict(), optimizer_state_dict=self.estimator.optimizer.state_dict(),
                           epoch=self.epoch, history=self.history,
                           experiment_dir=self.config.EXPERIMENT_DIR).save()

                # Train
                # Sample Predict
                ix, (pX, py, extra_info) = self.estimator.dataset.train_dataset.get_sample()
                prediction = self.estimator.predict(pX)

                # Visualize
                tim = self.visualizer.prediction_to_image(actual=py, prediction=prediction[0, :],
                                                          im_title='TLoss:{:0.5f} || VLoss:{:0.5f} || Epoch:{} ||ix:{}'.format(
                                                              tloss, vloss, self.epoch, ix))

                # Validation
                # Sample Predict
                ix, (pX, py, extra_info) = self.estimator.dataset.valid_dataset.get_sample()
                prediction = self.estimator.predict(pX)

                # Visualize
                vim = self.visualizer.prediction_to_image(actual=py, prediction=prediction[0, :],
                                                          im_title='TLoss:{:0.5f} || VLoss:{:0.5f} || Epoch:{} ||ix:{}'.format(
                                                              tloss, vloss, self.epoch, ix))

                # Report to Tensorboard
                self.estimator.writer.add_image('prediction image', tim, self.epoch)
                self.estimator.writer.add_image('valid prediction image', vim, self.epoch)
                self.estimator.writer.add_scalar('training_loss', tloss, self.epoch)
                self.estimator.writer.add_scalar('validation_loss', vloss, self.epoch)

                self.estimator.writer.add_text('Text', 'text logged at step: {}'.format(self.epoch), self.epoch)

                self.estimator.writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100),
                                                   self.epoch)

                # self.visualizer.append_data('tloss', tloss)
                # self.visualizer.append_data('vloss', vloss)

                # self.visualizer.visualize(self.epoch)
                # self.visualizer.report()

                # todo: do we need this really? not sure.
                # Save
                # self.history.append(epoch=epoch, phase='train', name='loss', value=tloss)
                # self.history.append(epoch=epoch, phase='valid', name='loss', value=vloss)


def resume_test():
    exp_name = 'resume_exp'
    stock_name = 'xlf'

    print('Experiment starting for {} ...'.format(stock_name))
    config = Config()
    config.STOCK_NAMES = [stock_name]
    config.EXPERIMENT_DIR = '../experiment/finance_cnn/{}/{}'.format(exp_name, stock_name)
    config.set_dataset_args()


    # experiment = Experiment.start_over(config=config)
    # experiment.do()
    experiment = Experiment.resume(config.EXPERIMENT_DIR, config=config)

    # experiment1 = Experiment.resume(config.EXPERIMENT_DIR, config=config)
    # experiment2 = Experiment.resume(config.EXPERIMENT_DIR, config=config)
    # experiment3 = Experiment.resume(config.EXPERIMENT_DIR, config=config)
    # experiment4 = Experiment.resume(config.EXPERIMENT_DIR, config=config)
    #
    # def pred(exp):
    #     # X, y, extra_info = next(exp.estimator.valid_dataloader.__iter__())
    #     # X, y, extra_info = next(exp.estimator.valid_dataloader.__iter__())
    #
    #     for i, (X,y,extra_info) in enumerate(exp.estimator.valid_dataloader):
    #         print(extra_info['date'])
    #         if i == 3:
    #             break
    #     print("=====")
    #     # print('Date : ', extra_info['date'])
    #     # print('Name : ', extra_info['name'])
    #     # print('X :', X.data.numpy().reshape(28,28))
    #
    #     X, y = torch.autograd.Variable(X.float(), requires_grad=False), torch.autograd.Variable(y.float(), requires_grad=False)
    #     voutput, vloss = exp.estimator.validate_on_batch(X, y)
    #     print(extra_info['date'],voutput, vloss)
    #
    # pred(experiment1)
    # pred(experiment2)
    # pred(experiment3)
    # pred(experiment4)
    # print()
    experiment.prediction_to_csv(save_path=config.EXPERIMENT_DIR)

def exp_pipeline():
    stock_names = ['dia', 'ewa', 'ewc', 'ewg', 'ewh', 'ewj', 'eww', 'spy', 'xlb',
                   'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly']

    exp_name = 'stock_exp'
    for stock_name in stock_names:

        print('Experiment starting for {} ...'.format(stock_name))
        config = Config()
        config.STOCK_NAMES = [stock_name]
        config.EXPERIMENT_DIR = '../experiment/finance_cnn/{}/{}'.format(exp_name, stock_name)
        config.set_dataset_args()

        experiment = Experiment.start_over(config)
        experiment.do()
        experiment.prediction_to_csv(save_path=config.EXPERIMENT_DIR)

def stress_test():
    exp_name = 'stress_exp'
    stock_name = ['dia']

    for i in range(0,28):

        config = Config()
        config.STOCK_NAMES = [stock_name]
        config.EXPERIMENT_DIR = '../experiment/finance_cnn/{}/{}'.format(exp_name, i)
        config.set_dataset_args()


        experiment = Experiment.start_over(config)

        experiment.estimator.stress_dataset(i)

        experiment.do()

        experiment.prediction_to_csv(save_path=config.EXPERIMENT_DIR)

exp_pipeline()
stress_test()
# resume_test()
