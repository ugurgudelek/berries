import os
import torch
import time

class GenericConfig:
    def __init__(self):
        self.EXPERIMENT_DIR = None
        # Device params
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
        if self.EXPERIMENT_DIR is None:
            raise Exception("EXPERIMENT_DIR should have a value")

        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        path = os.path.join(self.EXPERIMENT_DIR, 'config.ini')

        with open(path, 'w') as file:
            file.write(self.__str__())

class ConfigLSTM(GenericConfig):
    def __init__(self):
        GenericConfig.__init__(self)

        self.DATASET_NAME = 'ToyDataset'
        # Experiment params
        self.EPOCH_SIZE = 1000
        self.EXPERIMENT_DIR = '../experiment/{}/{}'.format(self.DATASET_NAME, str(int(time.time())))

        # Dataloader params
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False
        self.TRAIN_BATCH_SIZE = 2
        self.VALID_BATCH_SIZE = 2

class ConfigCNN(GenericConfig):
    """
    """

    def __init__(self):
        """
        """
        GenericConfig.__init__(self)

        # Aux params
        self.RANDOM_SEED = 42

        # Model params
        self.MODEL_NAME = 'CNN'
        self.INPUT_SIZE = 1
        self.OUTPUT_SIZE = 10

        # Dataloader params
        self.TRAIN_SHUFFLE = True
        self.VALID_SHUFFLE = False
        self.TRAIN_BATCH_SIZE = 200
        self.VALID_BATCH_SIZE = 200

        # Dataset params
        self.TRAIN_VALID_RATIO = 0.90
        self.DATASET_NAME = 'MNISTDataset'

        # Experiment params
        self.EPOCH_SIZE = 100
        self.EXPERIMENT_DIR = '../experiment/{}/{}'.format(self.DATASET_NAME ,str(int(time.time())))



