"""
Ugur Gudelek
config
ugurgudelek
08-Mar-18
finance-cnn
"""

# todo: make config an external txt file
import torch


class Config:
    """

    """

    def __init__(self):
        """

        """
        self.RANDOM_SEED = 42
        self.MODEL_NAME = 'CNN'
        self.EPOCH_SIZE = 20
        self.INPUT_SIZE = 28
        self.OUTPUT_SIZE = 3  # down, steady, up

        self.STOCK_NAMES = ['dia']
        self.SAVE_DATASET = True

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 64
        self.VALID_BATCH_SIZE = 64
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'IndicatorDataset'
        self.INPUT_PATH = '../dataset/finance/stocks/stocks.csv'

        self.EXPERIMENT_DIR = '../experiment/finance_cnn_exp1_dia_only'

        self.DATASET_ARGS = {'dataset_name': self.DATASET_NAME,
                             'train_valid_ratio': self.TRAIN_VALID_RATIO,
                             'input_path': self.INPUT_PATH,
                             'stock_names': self.STOCK_NAMES,
                             'save_dataset':self.SAVE_DATASET}

        self.MODEL_ARGS = {'model_name':self.MODEL_NAME,
                           'input_size': self.INPUT_SIZE,
                           'out_size': self.OUTPUT_SIZE,
                           'batch_size': self.TRAIN_BATCH_SIZE
                           }

        self.CRITERION_ARGS = {'criterion_name': 'MSE'}
        self.OPTIMIZER_ARGS = {'optimizer_name': 'Adam',
                               'lr': 0.0005}

        self.DATALOADER_ARGS = {'train_batch_size': self.TRAIN_BATCH_SIZE,
                                'train_shuffle': self.TRAIN_SHUFFLE,
                                'valid_batch_size': self.VALID_BATCH_SIZE,
                                'valid_shuffle': self.VALID_SHUFFLE
                                }


        self.STORAGE_NAMES = ['y_hat', 'loss', 'y']

        self.RESUME = False

        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')
        # self.USE_CUDA = False
        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))

    def set_dataset_args(self):
        self.DATASET_ARGS = {'dataset_name': self.DATASET_NAME,
                         'train_valid_ratio': self.TRAIN_VALID_RATIO,
                         'input_path': self.INPUT_PATH,
                         'stock_names': self.STOCK_NAMES,
                         'save_dataset': self.SAVE_DATASET}