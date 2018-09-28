import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import datetime
import os

from helper import quantize




def custom_test_on_batch(model, image, label, q_ratio=0.38):
    prediction = model.predict(image)
    mse = (label - prediction) ** 2
    p_q = quantize(prediction,q_ratio)
    r_q = quantize(label,q_ratio)
    if p_q == r_q:
        return prediction, mse, 1
    else:
        return prediction, mse, 0


def construct_mlp(params):
    model = Sequential()
    model.add(Dense(392, activation='relu', input_shape=(params["input_w"] * params["input_h"], )))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss=keras.losses.mean_squared_error, 
                optimizer=keras.optimizers.Adadelta(), 
                metrics=['mse', 'mae'])
    
    
    return model



def fit(model, data, params):
    train_images = data['train_images'].as_matrix()
    train_labels = data['train_labels'].as_matrix()
    train_images = train_images.reshape(train_images.shape[0], params["input_w"] , params["input_h"])
	train_images = train_images[]

    print("model will be trained with {}".format(train_images.shape))
    # fit the model to the training data
    print("Fitting model to the training data...")
    print("")
    model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
              validation_data=None)

    return model


def test(model, data, params, q_ratio=0.38):
    test_images = data['test_images'].as_matrix()
    test_labels = data['test_labels'].as_matrix()
    test_names = data['test_names'].as_matrix()
    test_dates = data['test_dates'].as_matrix()
    test_images = test_images.reshape(test_images.shape[0], params["input_w"] * params["input_h"])
    
    precisions = []
    accuracies = []
    losses = []
    
    predictions = []
    names = []
    dates = []
    actuals = []

    # train_data_size = train_images.shape[0]
    # test_data_size = test_images.shape[0]
    # cur_pointer = train_data_size + 1
    print("Calculating accuracy day by day...", end='\n\n')
    for i, (image, label, name, date) in enumerate(zip(test_images, test_labels, test_names, test_dates)):

        image = image.reshape((1, params["input_w"] * params["input_h"]))
        label = label.reshape((1, params["num_classes"]))
        # test for next image

        prediction, _, acc_cur = custom_test_on_batch(model, image, label, q_ratio=q_ratio)
        # loss_cur,acc_cur = model.test_on_batch(image,label)
        
        predictions.append(prediction[0][0])
        names.append(name[0])
        dates.append(date[0])
        actuals.append(label[0][0])
            
        accuracies.append(acc_cur)

        # train with only 1 more image
        model.train_on_batch(image, label)

        # show values every 100 cycle
        if i % 100 == 0 and i != 0:
            print("{} to {} mean : ".format(i - 100, i), np.mean(accuracies))

    print()
    print(np.mean(accuracies))

    print()
    
    pred_df = pd.DataFrame({'Name' : np.asarray(names), 'Date' : np.asarray(dates), 'Prediction' : np.asarray(predictions), 'Actual' : np.asarray(actuals)})
    return pred_df
    #history = {'prediction': predictions, 'loss': losses, 'acc': accuracies }
    #return model, history


def start_mlp_session(data, params, model_save_name, model_path="../model", result_path = "../result", model_read_name = ""):
    """Trains and evaluates MLP on the given train and test data, respectively."""

    # get date and clock info for model saving..
    now = str(datetime.datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')

    if model_read_name != "":
        
        model = load_model(model_path + "/" + model_read_name)
        
    else:
    
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # construct mlp
        print("MLP constructing...")
        model = construct_mlp(params=params)
        
        # fit data
        print("MLP fit session started...")
        model = fit(model, data, params)
        
        # save model before test
        model.save(model_path + "/" + model_save_name + "_before_" + now)

    # test
    print("MLP test session started...")
    pred_df = test(model, data, params)
    pred_df.to_pickle(result_path + "/predictions_mlp_" + model_save_name + "_" + now)
    
    # predictions, names, dates = test(model, data, params)
    # # make predictions, names, dates and actual labels numpy array
    # predictions = [item for sublist in predictions for item in sublist]
    # predictions = np.asarray(predictions)
    # names = [item for sublist in names for item in sublist]
    # names = np.asarray(names).reshape((-1,1))
    # dates = [item for sublist in dates for item in sublist]
    # dates = np.asarray(dates).reshape((-1,1))
    # actual = data['test_labels'].values

    # # save predictions after test
    # np.savetxt(result_path + "/predictions_" + model_save_name + "_" + now + ".csv", np.concatenate((names, dates, predictions, actual), axis=1), delimiter=',', fmt = "%s %s %s %s")
