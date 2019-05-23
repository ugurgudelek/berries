import torch
import time
import os

class GenericConfig:
    def __init__(self):
        self.EXPERIMENT_DIR = None
        # Device params
        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            if torch.cuda.get_device_name(0) == 'GeForce GT 650M':
                self.USE_CUDA = False
                print('USE_CUDA is set to False because this GPU is too old.')

        self.DEVICE = torch.device('cuda:0' if self.USE_CUDA else 'cpu')

        print('CUDA AVAILABLE:{}'.format(self.USE_CUDA))

    def __str__(self):
        return '\n'.join([f"{attr_key}={attr_val}"
                          for attr_key, attr_val in self.__dict__.items()])

    def save(self):
        if self.EXPERIMENT_DIR is None:
            raise NotImplementedError("EXPERIMENT_DIR should have a value")

        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        path = os.path.join(self.EXPERIMENT_DIR, 'config.ini')

        with open(path, 'w') as file:
            file.write(self.__str__())


class ConfigLSTM(GenericConfig):

    def __init__(self, dataset_name):
        GenericConfig.__init__(self)

        self.DATASET_NAME = dataset_name

        # Experiment params
        self.EPOCH_SIZE = 1000

        # models resume method will change this new EXPERIMENT_DIR
        self.EXPERIMENT_DIR = f'../experiment/{self.DATASET_NAME}/{int(time.time())}'

        # Dataloader params
        self.TRAIN_BATCH_SIZE = 1  # stateful lstm is not working with batch_size > 1. Why? Maybe answer is in inner functions of pytorch.
        self.VALID_BATCH_SIZE = self.TRAIN_BATCH_SIZE

        # Model params
        self.INPUT_SIZE = 1
        self.NUM_LAYERS = 1

        self.PREDICT_N = 40
        self.FUTURE_PREDICT_INIT_N = 2
        self.STATEFUL = True
        self.TRAIN_TEST_RATIO = 0.5
        self.LR = 0.001
        self.VERBOSE = 1

        self.INIT_X = 0.

        if self.DATASET_NAME == 'LongMemoryDebugDataset':
            # Starting to converge around epoch=68 on regression, epoch=100 on classification
            self.CLASSIFICATION = True
            self.FINITE = True
            self.HIDDEN_SIZE = 1
            self.OUTPUT_SIZE = 2
            self.WINDOW_SIZE = 10
            self.SUBSEQ_SIZE = 1
            self.HIDDEN_RESET_PERIOD = self.WINDOW_SIZE // self.SUBSEQ_SIZE
            if self.PREDICT_N % self.WINDOW_SIZE != 0:
                raise Exception("self.PREDICT_N % self.WINDOW_SIZE should be zero")

        elif self.DATASET_NAME == 'SineWaveDataset':
            self.CLASSIFICATION = False
            self.FINITE = False
            self.HIDDEN_SIZE = 2
            self.OUTPUT_SIZE = 1
            self.STEPS_PER_CYCLE = 60
            self.NUMBER_OF_CYCLES = 100
            self.RANDOM_FACTOR = 0.
            self.SUBSEQ_SIZE = 1
            self.HIDDEN_RESET_PERIOD = -1
            if self.PREDICT_N % self.STEPS_PER_CYCLE != 0:
                raise Exception("self.PREDICT_N % self.STEPS_PER_CYCLE should be zero")

        elif self.DATASET_NAME == 'SPY':
            self.CLASSIFICATION = False
            self.FINITE = False
            self.HIDDEN_SIZE = 20
            self.OUTPUT_SIZE = 1
            self.SUBSEQ_SIZE = 1
            self.HIDDEN_RESET_PERIOD = -1

        elif self.DATASET_NAME == 'WeatherDataset':
            self.CLASSIFICATION = False
            self.FINITE = False
            self.HIDDEN_SIZE = 20
            self.OUTPUT_SIZE = 1
            self.SUBSEQ_SIZE = 1
            self.HIDDEN_RESET_PERIOD = -1

        elif self.DATASET_NAME == 'LoadDataset':
            self.CLASSIFICATION = False
            self.FINITE = False
            self.HIDDEN_SIZE = 20
            self.OUTPUT_SIZE = 1
            self.SUBSEQ_SIZE = 1
            self.HIDDEN_RESET_PERIOD = -1

        else:
            raise Exception('You should pass proper dataset name.')

        if self.CLASSIFICATION and self.OUTPUT_SIZE < 2:
            raise Exception("OUTPUT_SIZE should be > 1")

        if self.TRAIN_BATCH_SIZE > self.HIDDEN_SIZE and self.TRAIN_BATCH_SIZE != 1:
            raise Exception("Check TRAIN_BATCH_SIZE and HIDDEN_SIZE. They are not valid.")








