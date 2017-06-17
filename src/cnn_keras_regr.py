import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import datetime
import os


def quantize(x, ratio=0.38):
    if x > ratio:
        return 1
    elif x < -ratio:
        return -1
    else:
        return 0


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

    print("model will be trained with {} and be tested with {} sample".format(train_images.shape))
    # fit the model to the training data
    print("Fitting model to the training data...")
    print("")
    model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
              validation_data=None)

    return model


def test(model, data, params, q_ratio=0.38):
    test_images = data['test_images'].as_matrix()
    test_labels = data['test_labels'].as_matrix()
    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)

    recalls = []
    precisions = []
    fprs = []
    tprs = []
    accuracies = []
    losses = []
    predictions=[]

    # train_data_size = train_images.shape[0]
    # test_data_size = test_images.shape[0]
    # cur_pointer = train_data_size + 1
    print("Calculating accuracy day by day...", end='\n\n')
    for i, (image, label) in enumerate(zip(test_images, test_labels)):

        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))
        # test for next image

        prediction, loss_cur, acc_cur = custom_test_on_batch(model, image, label, q_ratio=q_ratio)
        # loss_cur,acc_cur = model.test_on_batch(image,label)

        predictions.append(prediction)
        accuracies.append(acc_cur)
        losses.append(loss_cur)


        # train with only 1 more image
        model.train_on_batch(image, label)



        # show values every 100 cycle
        if i % 100 == 0 and i != 0:
            print("{} to {} mean : ".format(i - 100, i), np.mean(accuracies))

    print()
    print(np.mean(accuracies))

    print()
    history = {'prediction': predictions, 'loss': losses, 'acc': accuracies }
    return model, history,


def start_cnn_session(data, params, model_name, save_path="../model"):
    """Trains and evaluates CNN on the given train and test data, respectively."""

    # get date and clock info for model saving..
    now = str(datetime.datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # construct cnn
    print("CNN constructing...")
    model = construct_cnn(params=params)

    # fit data
    print("CNN fit session started...")
    model = fit(model, data, params)

    # save model before test
    model.save(save_path + "/" + model_name + "_before_" + now)

    # test
    print("CNN test session started...")
    model = test(model, data, params)

    # save model after test
    model.save(save_path + "/" + model_name + "_after_" + now)
