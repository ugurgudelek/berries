from estimator import Estimator
from model import LSTM, GenericModel
from dataset import IndicatorDataset
import torch
from torch import nn

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

import os
import collections

from torch.autograd import Variable
from torch import FloatTensor

# todo: try: calculate pct_change and label outliers

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

        self.SEQ_LEN = 128
        self.INPUT_SIZE = 9
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
        self.INPUT_PATH = '../dataset/finance/xlf.csv'

        self.EXPERIMENT_DIR = '../experiment/xlf_test_' + str(int(time.time()))

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
    with tqdm(total=100) as pbar:
        pbar.update(10)
    # with trange(epoch, config.EPOCH_SIZE) as t:
    #     for epoch in t:
    #         # Fit the model
    #         training_loss = estimator.fit(dataloader=train_dataloader)
    #
    #         # Predict validation set
    #         valid_prediction, valid_loss = estimator.validate(xs=valid_xs, ys=valid_ys)
    #         valid_prediction = valid_prediction.to('cpu').data.numpy()
    #         valid_loss = valid_loss.item()
    #
    #         # Log loss
    #         estimator.writer.add_scalar('training_loss', training_loss, epoch)
    #         estimator.writer.add_scalar('validation_loss', valid_loss, epoch)

    # Training Plots

    train_prediction, train_loss = estimator.validate(xs=train_xs, ys=train_ys)
    train_prediction = train_prediction.to('cpu').data.numpy()

    train_prediction_df = pd.DataFrame(dict(y=train_ys.data.numpy().flatten(), yhat=train_prediction.flatten()))

    # plot_top_bot_turning_point(train_prediction_df.iloc[:300])

    # Validation Plots

    valid_prediction, valid_loss = estimator.validate(xs=valid_xs, ys=valid_ys)
    valid_prediction = valid_prediction.to('cpu').data.numpy()

    valid_prediction_df = pd.DataFrame(dict(y=valid_ys.data.to('cpu').numpy().flatten(), yhat=valid_prediction.flatten()))
    valid_prediction_df.plot()
    plt.show()
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
