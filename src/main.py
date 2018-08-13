from estimator import Estimator
from model import LSTM
from dataset import IndicatorDataset
import torch

from torch.utils.data import DataLoader

from tqdm import trange
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix



class Config:
    """
    """
    # todo: config dosyasını kaldırıp main içerisinde oluşturmayı tekrar düşün.
    # todo: şimdilk sadece problem bazlı çalışalım.

    def __init__(self):
        """
        """
        self.RANDOM_SEED = 42
        self.MODEL_NAME = 'LSTM'
        self.EPOCH_SIZE = 20

        self.SEQ_LEN = 128
        self.INPUT_SIZE = 15
        self.OUTPUT_SIZE = 3  # down, steady, up

        self.LABEL_WINDOW = 7

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 10
        self.VALID_BATCH_SIZE = 10
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'IndicatorDataset'
        self.INPUT_PATH = '../input/spy_spline.csv'

        self.EXPERIMENT_DIR = '../experiment/spy_spline_2'

        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')

        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))


if __name__ == "__main__":

    config = Config()

    dataset = IndicatorDataset(dataset_name=config.DATASET_NAME,
                               input_path=config.INPUT_PATH,
                               train_valid_ratio=config.TRAIN_VALID_RATIO,
                               save_dataset=True,
                               seq_len=config.SEQ_LEN)
    model = LSTM(input_size=config.INPUT_SIZE,
                 seq_length=config.SEQ_LEN,
                 num_layers=1,
                 out_size=config.OUTPUT_SIZE,
                 batch_size=config.TRAIN_BATCH_SIZE,
                 use_cuda=config.USE_CUDA)

    estimator = Estimator(dataset=dataset,
                          model=model,
                          use_cuda=config.USE_CUDA,
                          exp_dir=config.EXPERIMENT_DIR,
                          train_batch_size=config.TRAIN_BATCH_SIZE,
                          valid_batch_size=config.VALID_BATCH_SIZE)

    epoch = 0
    with trange(epoch, config.EPOCH_SIZE) as t:
        for epoch in t:
            tloss, vloss, tacc, vacc = estimator.run_epoch(epoch, t)
            print(tloss, vloss, tacc, vacc)

            estimator.writer.add_scalar('training_loss', tloss, epoch)
            estimator.writer.add_scalar('validation_loss', vloss, epoch)
            estimator.writer.add_scalar('training_acc', tacc, epoch)
            estimator.writer.add_scalar('validation_acc', vacc, epoch)
    print()

    ix, (sample_x, sample_y, ext_info) = dataset.train_dataset.get_sample()
    # prediction = estimator.predict(sample_x)
    pXs, pys, poutputs, plosses, (pdates, pnames) = estimator.predict_all_validation()
    prediction_df = pd.DataFrame(dict(y0=pys[:, 0], y1=pys[:, 1], y2=pys[:, 2], yhat0=poutputs[:, 0], yhat1=poutputs[:, 1], yhat2=poutputs[:, 2]))

    def onehot2label(row):
        row = row.values
        return pd.Series(dict(y=np.argmax(row[:3]), yhat=np.argmax(row[3:])))

    labeled_pred_df = prediction_df.apply(onehot2label, axis=1)

    f1 = f1_score(y_true=labeled_pred_df['y'], y_pred=labeled_pred_df['yhat'], average=None)
    print('f1 score:\n{score}'.format(score=f1))

    confusion = confusion_matrix(y_true=labeled_pred_df['y'], y_pred=labeled_pred_df['yhat'])
    print('confusion matrix:\n{confusion}'.format(confusion=confusion))



    # plt.plot(sample_x[0][0])
    # plt.show()
    #
    # raw_sample = dataset.get_data_seq(name=ext_info['name'].iloc[0],
    #                                   first_date=ext_info['date'].iloc[0],
    #                                   last_date=ext_info['date'].iloc[-1])
    # plt.plot(raw_sample['date'], raw_sample['close'], '-r')
    # plt.show()
    # print()
