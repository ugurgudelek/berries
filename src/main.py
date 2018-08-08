from estimator import Estimator
from model import LSTM
from dataset import IndicatorDataset
import torch


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
        self.INPUT_SIZE = 6
        self.OUTPUT_SIZE = 3  # down, steady, up

        self.LABEL_WINDOW = 7

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 64
        self.VALID_BATCH_SIZE = 1
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
                               save_dataset=True)
    model = LSTM(input_size=config.INPUT_SIZE,
                 seq_length=config.SEQ_LEN,
                 num_layers=1,
                 out_size=config.OUTPUT_SIZE,
                 batch_size=config.TRAIN_BATCH_SIZE,
                 use_cuda=config.USE_CUDA)

    # estimator = Estimator(dataset=dataset,
    #                       model=model,
    #                       use_cuda=config.USE_CUDA,
    #                       summary_writer_path='../summary',
    #                       exp_dir=config.EXPERIMENT_DIR,
    #                       batch_size=config.TRAIN_BATCH_SIZE)