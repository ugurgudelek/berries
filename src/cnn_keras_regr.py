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


def test(model, data, params, q_ratio=0.38):
    test_images = data['test_images']
    test_labels = data['test_labels']
    test_names = data['test_names']
    test_dates = data['test_dates']

    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)
    
    precisions = []
    accuracies = []
    losses = []
    
    predictions = []
    names = []
    dates = []
    actuals = []
    mses = []

    cur_date = d_df.date.iloc[0]
    train_again_images = []
    train_again_labels = []
    print("Calculating accuracy day by day...", end='\n\n')
    for i, (image, label, name, date) in enumerate(zip(test_images, test_labels, test_names, test_dates)):

        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))
        # test for next image

        if cur_date != date: # update model
            for train_image,train_label in zip(train_again_images,train_again_labels):
                # train with only 1 more image
                model.train_on_batch(train_image, train_label)
            train_again_images = []
            train_again_labels = []
            cur_date = date

        prediction, mse, acc_cur = custom_test_on_batch(model, image, label, q_ratio=q_ratio)
        # loss_cur,acc_cur = model.test_on_batch(image,label)
        train_again_images.append(image)
        train_again_labels.append(label)
        
        predictions.append(prediction[0][0])
        names.append(name)
        dates.append(date)
        actuals.append(label[0][0])
            
        accuracies.append(acc_cur)
        mses.append(mse)

        # show values every 100 cycle
        if i % 100 == 0 and i != 0:
            print("{} to {} mean : ".format(i - 100, i), np.mean(accuracies))

    print()
    print(np.mean(accuracies))
    print(np.mean(mses))

    print()
    
    pred_df = pd.DataFrame({'Name' : np.asarray(names), 'Date' : np.asarray(dates), 'Prediction' : np.asarray(predictions), 'Actual' : np.asarray(actuals)})
    return pred_df
    #history = {'prediction': predictions, 'loss': losses, 'acc': accuracies }
    #return model, history


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

    # test
    print("CNN test session started...")
    # q_ratio = 0.38 # means 3 class :
    q_ratio = 0 # means 2 class
    pred_df = test(model, data, params, q_ratio=q_ratio)
    pred_df.to_pickle(result_path + "/predictions_" + model_save_name + "_qratio_"+str(q_ratio)+"_" + now)
    
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
