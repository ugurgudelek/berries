import numpy as np
import pandas as pd
import csv

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import datetime
import os

def construct_cnn(params):
    # CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(params["input_w"], params["input_h"], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params["num_classes"]))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['mse', 'mae'])
    return model



def fit(model, data, params):
    train_images = data['train_images'].as_matrix()
    train_labels = data['train_labels'].as_matrix()
    train_images = train_images.reshape(train_images.shape[0], params["input_w"], params["input_h"], 1)

    print("model will be trained with {}".format(train_images.shape))
    # fit the model to the training data
    print("Fitting model to the training data...")
    print("")
    model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
              validation_data=None)

    return model

def start_cnn_session(data, params, model_save_name, model_path="../model", result_path = "../result", model_read_name = ""):
    """Trains and evaluates CNN on the given train and test data, respectively."""

    # get date and clock info for model saving..
    now = str(datetime.datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')

    if model_read_name != "":
        
        model = load_model(model_path + "/" + model_read_name)
        
    else:
    
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # construct cnn
        print("CNN constructing...")
        model = construct_cnn(params=params)
        
        # fit data
        print("CNN fit session started...")
        model = fit(model, data, params)
        
        # save model before test
        model.save(model_path + "/" + model_save_name + "_before_" + now)    
