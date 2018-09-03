from estimator import Estimator
from model import LSTM, GenericModel
from dataset import IndicatorDataset
import torch
from torch import nn

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import trange
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

import os
import collections

from torch.autograd import Variable
from torch import FloatTensor



# done: plot grafiklerini daha anlamlı hale getir.
# todo: dataset classını toparla.
# todo: classification ve regression için modelleri ayırmayı düşünebilirsin.
# todo: add weight image 2d cnn & 1d dense & think how can we do this RNN
# in_progress: add buy&sell strategy class
# in_progress: smoot output of the lstm to work with buysell class
# todo: add buy&sell metric table creation
# todo: think auc&roc or another metric graphs
# done: add model to the tensorboard or onnx - onnx DONE, tensorboard later

def custom_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(b) for b in batch], 0)

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]

    elif isinstance(batch[0], dict):
        return pd.DataFrame(list(batch)).to_dict('list')
    else:

        raise Exception('Update custom_collate_fn!!')

class Config:
    """
    """

    def __init__(self):
        """
        """
        self.RANDOM_SEED = 42
        self.MODEL_NAME = 'LSTM'
        self.EPOCH_SIZE = 100

        self.SEQ_LEN = 256
        self.INPUT_SIZE = 15
        # self.OUTPUT_SIZE = 3  # down, steady, up
        self.OUTPUT_SIZE = 1
        self.NUM_LAYERS = 4
        self.HIDDEN_SIZE = 40

        # self.LABEL_WINDOW = 7
        self.LABEL_TYPE = 'regression'

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 100
        self.VALID_BATCH_SIZE = 100
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'IndicatorDataset'
        self.INPUT_PATH = '../input/spy.csv'

        self.EXPERIMENT_DIR = '../experiment/spy_' + str(int(time.time()))

        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')

        if self.USE_CUDA:
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'

        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))

    def __str__(self):
        string = ''
        for attr_key, attr_val in self.__dict__.items():
            string += attr_key
            string += '='
            string += str(attr_val)
            string += '\n'
        return string

    def save(self):
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        path = os.path.join(self.EXPERIMENT_DIR, 'config.ini')

        with open(path, 'w') as file:
            file.write(self.__str__())


if __name__ == "__main__":

    config = Config()
    config.save()

    dataset = IndicatorDataset(dataset_name=config.DATASET_NAME,
                               input_path=config.INPUT_PATH,
                               train_valid_ratio=config.TRAIN_VALID_RATIO,
                               save_dataset=True,
                               seq_len=config.SEQ_LEN,
                               label_type=config.LABEL_TYPE)

    train_dataloader = DataLoader(dataset.train_dataset,
                                  batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=False,
                                  drop_last=True, collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(dataset.valid_dataset,
                                  batch_size=config.VALID_BATCH_SIZE,
                                  shuffle=False,
                                  drop_last=True, collate_fn=custom_collate_fn)

    model = LSTM(input_size=config.INPUT_SIZE,
                 seq_length=config.SEQ_LEN,
                 num_layers=config.NUM_LAYERS,
                 out_size=config.OUTPUT_SIZE,
                 hidden_size=config.HIDDEN_SIZE,
                 batch_size=config.TRAIN_BATCH_SIZE,
                 device=config.DEVICE).to(config.DEVICE)
    model.to_onnx(directory=config.EXPERIMENT_DIR)
    model.to_txt(directory=config.EXPERIMENT_DIR)

    estimator = Estimator(model=model,
                          device=config.DEVICE,
                          exp_dir=config.EXPERIMENT_DIR)

    # layers = model.get_layers()
    #
    # wbn = model.weight_bias_name()
    #
    # weights, biases, names = list(zip(*[(weight,bias,name) for weight,bias,name in wbn]))
    #
    # model.visualize_weights().show()

    train_xs, train_ys = dataset.train_dataset.get_all_data(transforms=[FloatTensor, Variable])
    valid_xs, valid_ys = dataset.valid_dataset.get_all_data(transforms=[FloatTensor, Variable])

    epoch = 0
    with trange(epoch, config.EPOCH_SIZE) as t:
        for epoch in t:
            # Fit the model
            training_loss = estimator.fit(dataloader=train_dataloader)

            # Predict validation set
            valid_prediction, valid_loss = estimator.validate(xs=valid_xs, ys=valid_ys)
            valid_prediction = valid_prediction.to('cpu').data.numpy()
            valid_loss = valid_loss.item()

            # Log loss
            estimator.writer.add_scalar('training_loss', training_loss, epoch)
            estimator.writer.add_scalar('validation_loss', valid_loss, epoch)


    def turning_points(data, on, window=15):
        data = data.copy()
        data['maxs'] = data[on].rolling(window, center=True, min_periods=window).apply(
            IndicatorDataset.is_center_max)
        data['mins'] = data[on].rolling(window, center=True, min_periods=window).apply(
            IndicatorDataset.is_center_min)

        data['label'] = 'mid'
        data.loc[data['maxs'] == 1, 'label'] = 'top'
        data.loc[data['mins'] == 1, 'label'] = 'bot'

        data = data.drop(['maxs', 'mins'], axis=1)

        return data['label']

    def plot_top_bot_turning_point(p):
        p['y_label'] = turning_points(p, 'y')
        p['yhat_label'] = turning_points(p, 'yhat')

        x = range(len(p))

        # (r,g,b,a)
        true_colormap = {'mid':(0.2, 0.4, 0.6, 0), 'top':(1, 0, 0, 0.7), 'bot':(0, 1, 0, 0.7)}
        pred_colormap = {'mid': (0.2, 0.4, 0.6, 0), 'top': (0, 0, 1, 0.7), 'bot': (0, 1, 1, 0.7)}

        plt.scatter(x=x, y=p['y'], c=[true_colormap[label] for label in p['y_label']], label='y')
        plt.scatter(x=x, y=p['yhat'], c=[pred_colormap[label] for label in p['yhat_label']], label='yhat')
        plt.plot(x, p['y'], '--b', label='close', alpha=1)
        plt.plot(x, p['yhat'], lw=1, label='prediction', c='g', alpha=0.5)
        plt.legend()

    def label_wrt_distance(self, stocks, window=7):
    # tobecontinued.........
    # todo: change prediction to zigzag
        # point turning points then process for distance
        # after this line, stocks has 'label' column which has top-mid-bot values.
        stocks = label_top_bot_mid(stocks=stocks, window=window)
        stocks = filter_consequtive_same_label(stocks=stocks)
        stocks = crop_firstnonbot_and_lastnontop(stocks=stocks)

        def distance(idxs, turning_points):
            """
            Assumes turning_points start with increasing segment.
            :param idxs:
            :param turning_points:
            :return:
            """
            segments = np.array(list(zip(turning_points[:-1], turning_points[1:])))

            def calc_dist(x, lower, upper):
                return (x-lower)/(upper-lower)

            state = True
            dist = np.zeros_like(idxs, dtype=np.float)
            for (lower,upper) in segments:
                for i in range(lower,upper):
                    if state:
                        dist[i] = calc_dist(i, lower, upper)
                    else:
                        dist[i] = 1 - calc_dist(i, lower, upper)

                state = not state

            return dist

        def inner_func(stock_data):
            mid_idxs = stock_data.loc[stock_data['label'] == 'mid'].index.values
            top_idxs = stock_data.loc[stock_data['label'] == 'top'].index.values
            bot_idxs = stock_data.loc[stock_data['label'] == 'bot'].index.values

            turning_points = np.sort(np.concatenate((bot_idxs, top_idxs)))
            stock_data['label'] = distance(stock_data.index.values, turning_points)

            # at this point "label" values are between 0-1 range.
            # let me add -0.5 bias
            stock_data['label'] = stock_data['label'] - 0.5

            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def filter_consequtive_same_label(self, stocks):

        def inner_func(stock_data):
            state = None
            for i,row in stock_data.iterrows():
                if row['label'] == 'mid':
                    continue
                if state is None:
                    state = row['label']
                    continue
                if state == row['label']:
                    stock_data.loc[i,'label'] = 'mid'
                else:
                    state = row['label']
            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def crop_firstnonbot_and_lastnontop(self, stocks):

        def inner_func(stock_data):
            first_bot_idx = stock_data.loc[stock_data['label'] == 'bot'].index.values[0]
            last_top_idx = stock_data.loc[stock_data['label'] == 'top'].index.values[-1]

            return stock_data.loc[first_bot_idx:last_top_idx, :]

        return stocks.groupby('name').apply(inner_func).dropna().reset_index(drop=True)


    # Training Plots

    train_prediction, train_loss = estimator.validate(xs=train_xs, ys=train_ys)
    train_prediction = train_prediction.to('cpu').data.numpy()

    train_prediction_df = pd.DataFrame(dict(y=train_ys.data.numpy().flatten(), yhat=train_prediction.flatten()))

    plot_top_bot_turning_point(train_prediction_df.iloc[:300])

    # Validation Plots

    valid_prediction, valid_loss = estimator.validate(xs=valid_xs, ys=valid_ys)
    valid_prediction = valid_prediction.to('cpu').data.numpy()

    valid_prediction_df = pd.DataFrame(dict(y=valid_ys.data.to('cpu').numpy().flatten(), yhat=valid_prediction.flatten()))
    valid_prediction_df.plot()
    # plt.show()
    #
    # if config.LABEL_TYPE == 'classification':
    #     pXs, pys, poutputs, plosses, (pdates, pnames) = estimator.predict_all_validation()
    #     prediction_df = pd.DataFrame(
    #         dict(y0=pys[:, 0], y1=pys[:, 1], y2=pys[:, 2], yhat0=poutputs[:, 0], yhat1=poutputs[:, 1],
    #              yhat2=poutputs[:, 2]))
    #
    #
    #     def onehot2label(row):
    #         row = row.values
    #         return pd.Series(dict(y=np.argmax(row[:3]), yhat=np.argmax(row[3:])))
    #
    #
    #     labeled_pred_df = prediction_df.apply(onehot2label, axis=1)
    #
    #     f1 = f1_score(y_true=labeled_pred_df['y'], y_pred=labeled_pred_df['yhat'], average=None)
    #     print('f1 score:\n{score}'.format(score=f1))
    #
    #     confusion = confusion_matrix(y_true=labeled_pred_df['y'], y_pred=labeled_pred_df['yhat'])
    #     print('confusion matrix:\n{confusion}'.format(confusion=confusion))
