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
        self.EPOCH_SIZE = 401
        self.SEQ_LENGTH = 96
        self.NUM_LAYERS = 1
        self.INPUT_SIZE = 6
        self.OUTPUT_SIZE = 96

        self.TRAIN_VALID_RATIO = 0.90
        self.TRAIN_BATCH_SIZE = 256
        self.VALID_BATCH_SIZE = 256
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False


        # self.INPUT_PATH = '../dataset/energy/load_wo_feb29.csv'
        self.INPUT_PATH = '../dataset/energy/pvgeneration.csv'
        # self.INPUT_PATH = '../dataset/energy/sample_load.csv'
        self.EXPERIMENT_DIR = '../experiment/pvgeneration_w_hours'
        self.RANDOM_SEED = 7

        # self.TRAIN_DAY = 2700  # 2700 days * 96 quarter out of 2922 days
        # self.TRAIN_DAY = 2555 # loaddataste
        self.TRAIN_DAY = 2102  # pvdataset
        self.VALID_DAY = 365

        self.STORAGE_NAMES = ['y_hat', 'loss', 'y']

        self.RESUME = False

        self.USE_CUDA = torch.cuda.is_available()
        # self.USE_CUDA = False
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


