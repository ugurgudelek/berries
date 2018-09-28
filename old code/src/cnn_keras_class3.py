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
    test_images = data['test_images']
    test_labels = data['test_labels']
    test_names = data['test_names']
    test_dates = data['test_dates']

    test_images = test_images.as_matrix().reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)

    predictions = []    
    names = []
    dates = []
    actuals = []
    
    num_correct_preds  = 0
    num_preds = 0

    cur_date = test_dates.iloc[0][0]
    train_again_images = []
    train_again_labels = []
    print("Calculating accuracy day by day...", end='\n\n')
    for i, (image, label, name, date) in enumerate(zip(test_images, test_labels.as_matrix(), test_names[0], test_dates[0])):

        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))

        if cur_date != date: # update model
            for train_image,train_label in zip(train_again_images,train_again_labels):
                # train with only 1 more image
                model.train_on_batch(train_image, train_label)
            train_again_images = []
            train_again_labels = []
            cur_date = date


        # test for next image
        prediction = model.predict(image)

        # update the number of correct predictions
        if np.argmax(prediction, 1) == np.argmax(label, 1):
            num_correct_preds += 1
        num_preds += 1
        
        predictions.append(np.array([prediction[0][0], prediction[0][1], prediction[0][2]]))
        names.append(name)
        dates.append(date)
        actuals.append(np.array([label[0][0], label[0][1], label[0][2]]))

        # train with only 1 more image
        model.train_on_batch(image, label)

        # inform every 100 cycle
        if i % 100 == 0 and i != 0:
            print("{}-th image, mean accuracy {} %".format(i, num_correct_preds / num_preds * 100))
            
    pred_df = pd.DataFrame({'Name' : np.asarray(names), 'Date' : np.asarray(dates), 'Pred0' : np.asarray(predictions)[:,0], 'Pred1' : np.asarray(predictions)[:,1], 'Pred2' : np.asarray(predictions)[:,2], 'Act0' : np.asarray(actuals)[:,0], 'Act1' : np.asarray(actuals)[:,1], 'Act2' : np.asarray(actuals)[:,2]})
            
    return pred_df


def start_cnn_session(data, params, model_save_name, model_path="../model", result_path="../result", model_read_name=""):
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
    pred_df = test(model, data, params)
    pred_df.to_pickle(result_path + "/predictions_" + model_save_name + "_" + now)
    
    # predictions is list of lists, flatten it
    # predictions = [item for sublist in predictions for item in sublist]
    # predictions = np.asarray(predictions)
    # # make list of numpy array out of data['test_labels']
    # actual = data['test_labels'].values

    # # calculate and print accuracy
    # print("Accuracy: {} %".format(np.sum(np.argmax(predictions,1) == np.argmax(actual,1)) / predictions.shape[0] * 100)) # simple accuracy (in %)

    # # save predictions after test
    # np.savetxt(result_path + "/predictions_" + model_save_name + "_" + now, np.concatenate((predictions, actual), axis=1), delimiter=',')
