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
        self.MODEL_NAME = 'CNN'
        self.EPOCH_SIZE = 100
        self.INPUT_SIZE = 28
        self.OUTPUT_SIZE = 3  # down, steady, up

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 64
        self.VALID_BATCH_SIZE = 64
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False

        self.DATASET_NAME = 'IndicatorDataset'
        self.INPUT_PATH = '../dataset/finance/stocks/stocks.csv'

        self.EXPERIMENT_DIR = '../experiment/finance_cnn'
        self.RANDOM_SEED = 7


        self.DATASET_ARGS = {'dataset_name': self.DATASET_NAME,
                             'train_valid_ratio': self.TRAIN_VALID_RATIO,
                             'input_path': self.INPUT_PATH}

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

        # self.USE_CUDA = torch.cuda.is_available()
        self.USE_CUDA = False
        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))
        # self.USE_CUDA = False
# class Config:
#     """
#
#     """
#
#     def __init__(self):
#         self.stocks_dir = '../input/raw_data'
#         self.stock_names = ['dia','ewa','ewc','ewg',
#                             'ewh','ewj','eww','spy',
#                             'xlb','xle','xlf','xli',
#                             'xlk','xlp','xlu','xlv','xly']
#
#         # self.stock_names = ['spy']
#         self.label_after = 20
#
#         self.input_size = 28
#         self.seq_length = 28
#         self.num_layers = 1
#         self.out_size = 1
#
#         self.train_batch_size = 1000
#         self.valid_batch_size = 1000
#
#         self.train_shuffle = True
#         self.valid_shuffle = False
#
#         self.epoch_size = 20
#         self.storage_names = ['y_hat', 'loss', 'y']
