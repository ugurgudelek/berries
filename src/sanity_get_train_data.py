import os
import pandas as pd
import numpy as np
import clustering
import classes
import metrics as mt
import time

from helper import *

def get_train_data(stock_names, read_path="../input/images_with_labels", labels_are_last=1, save_path="../input/last_saved_data"):
    
    if os.path.isfile(save_path+"/last_saved.pickle"):
        return pd.read_pickle(save_path+"/last_saved.pickle")
    
    all_train_images = []
    all_train_labels = []
    all_train_names = []
    all_train_dates=[]

    # todo: burada şu strig colon işini çöz
    for stock in stock_names:
        data_df = pd.read_csv(read_path + "/{}.csv".format(stock), header=None)
        names = data_df.iloc[:, 0] # first element
        dates = data_df.iloc[:, 1] # second element
        images = data_df.iloc[:, 2:-labels_are_last] # remaining elements
        labels = data_df.iloc[:, -labels_are_last:] # last elements

        print("all images are merging with {} ...".format(stock))

        # determine where to split
        train_image_count = images.shape[0]

        # split train and test
        # for 16 year of data : nearly 14 year train-last 2 year test
        train_images = images.iloc[0:train_image_count]
        train_labels = labels.iloc[0:train_image_count]
        train_names = names.iloc[0:train_image_count]
        train_dates = dates.iloc[0:train_image_count]

        # todo: need to make data class because above not seems good. -ugurgudelek

        if len(all_train_images) == 0:
            all_train_images = np.array(train_images)
            all_train_labels = train_labels.values
            all_train_names = np.array(train_names)
            all_train_dates = np.array(train_dates)
        else:
            all_train_images = np.append(all_train_images, train_images, axis=0)
            all_train_labels = np.append(all_train_labels, train_labels.values, axis=0)
            all_train_names = np.append(all_train_names,train_names, axis=0)
            all_train_dates = np.append(all_train_dates,train_dates, axis=0)

        print("current train shape is {} and {} label ".format(pd.DataFrame(all_train_images).shape,
                                                               all_train_labels.shape[1]))
    
    print("Sorting train data by date and name...")
    sorted_train_data = pd.DataFrame()
    sorted_train_data['date'] = all_train_dates
    sorted_train_data['name'] = all_train_names
    sorted_train_data['image'] = [i for i in all_train_images]
    sorted_train_data['label'] = all_train_labels
    sorted_train_data = sorted_train_data.sort_values(by = ['date', 'name'])
    
    data = {'train_images': pd.DataFrame(np.asarray([i for i in sorted_train_data['image']])),
            'train_labels': pd.DataFrame(sorted_train_data['label'].values),
            'train_names': pd.DataFrame(sorted_train_data['name'].values),
            'train_dates': pd.DataFrame(sorted_train_data['date'].values)
    }    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pd.to_pickle(data, save_path+"/last_saved.pickle")

    return data
