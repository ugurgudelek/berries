from estimator import Estimator
from model import LSTM
from dataset import IndicatorDataset
import torch

from torch.utils.data import DataLoader

from tqdm import trange


# todo: model tahminlerini sklearn tarzı bir hale getirip, gökberkin istediği tabloları ve plotları oluştur.

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
        self.TRAIN_BATCH_SIZE = 2
        self.VALID_BATCH_SIZE = 2
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'IndicatorDataset'
        self.INPUT_PATH = '../input/spy.csv'

        self.EXPERIMENT_DIR = '../experiment/spy_only'

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
                          batch_size=config.TRAIN_BATCH_SIZE)



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