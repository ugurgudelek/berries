"""This function read merged binary h5py file from input folder.
Stores dataset related functions.
We can now call cnn training here.
#todo: do not call cnn here.
"""
import pandas as pd
import numpy as np
import cnn
class Dataset:
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.X = None
        self.y = None

    def load(self, path):
        self.dataframe = pd.read_hdf(path_or_buf=path, key='df')

        #normalize flabel
        #todo: make this before dataset creation
        self.dataframe['flabel'] = self.label_normalization(self.dataframe['flabel'])

    def train_test_split(self, date):
        # get 784px flatten image
        self.X_train = self.dataframe.loc[(self.dataframe['date'] <= date)].iloc[:,:784]
        self.y_train = self.dataframe.loc[(self.dataframe['date'] <= date)]['flabel']
        # get 784px flatten image
        self.X_test = self.dataframe.loc[(self.dataframe['date'] > date)].iloc[:,:784]
        self.y_test = self.dataframe.loc[(self.dataframe['date'] > date)]['flabel']

    def transformXy(self):
        # transform to numpy array
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)


    def label_normalization(self,y):
        return (y - y.mean()) / y.std()

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load(path="../input/dataholder/dataholder.h5py")
    dataset.train_test_split('2014-12-31')
    dataset.transformXy()

    params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 10, "epochs": 100}
    cnnengine = cnn.CNNEngine(params=params, model_save_path='../model', run_number='7')
    cnnengine.train_direct(dataset.X_train, dataset.y_train)
    cnnengine.save_model()