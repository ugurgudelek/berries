import numpy as np
import pandas as pd
import csv
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import datetime
import os

from helper import quantize

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
    model.add(Dense(params["num_classes"], activation="softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['binary_accuracy'])
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
    test_images = data['test_images'].as_matrix()
    test_labels = data['test_labels'].as_matrix()
    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)
    
    predictions=[]

    # train_data_size = train_images.shape[0]
    # test_data_size = test_images.shape[0]
    # cur_pointer = train_data_size + 1
    print("Calculating accuracy day by day...", end='\n\n')
    for i, (image, label) in enumerate(zip(test_images, test_labels)):

        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))

        # test for next image
        prediction = model.predict(image)

        predictions.append(prediction)

        # train with only 1 more image
        model.train_on_batch(image, label)

        # inform every 100 cycle
        if i % 100 == 0 and i != 0:
            print("Test {}-th image".format(i))
            
    return predictions


def start_cnn_session(data, params, model_name, model_save_path="../model", result_save_path="../result", full_path=""):
    """Trains and evaluates CNN on the given train and test data, respectively."""

    # get date and clock info for model saving..
    now = str(datetime.datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')

    if full_path != "":

        model = load_model(full_path)
        
    else:
        
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
        # construct cnn
        print("CNN constructing...")
        model = construct_cnn(params=params)
        
        # fit data
        print("CNN fit session started...")
        model = fit(model, data, params)
        
        # save model before test
        model.save(model_save_path + "/" + model_name + "_before_" + now)

    # test
    print("CNN test session started...")
    predictions = test(model, data, params)
    # predictions is list of lists, flatten it
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.asarray(predictions)
    # make list of numpy array out of data['test_labels']
    actual = data['test_labels'].values

    # calculate and print accuracy
    print("Accuracy: {}%".format(np.sum(np.argmax(predictions,1) == np.argmax(actual,1)) / predictions.shape[0] * 100)) # simple accuracy (in %)

    # save predictions after test
    np.savetxt(result_save_path + "/predictions_" + model_name + "_" + now, np.concatenate((predictions, actual), axis=1), delimiter=',')
