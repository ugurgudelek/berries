"CNN related codes here..."
import datetime
import os
import numpy as np
from utils import quantize
from utils import Bucket
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error
from keras.optimizers import Adadelta




class CNNEngine:
    def __init__(self, params, model_save_path, run_number,verbose=True):
        self.params = params
        self.model_save_path = model_save_path
        self.verbose = verbose
        self.model_name = '{}_model.h5py'.format(run_number)
        self.model = self.construct_cnn()

        self.X_bucket = Bucket(size=self.params['batch_size'])
        self.y_bucket = Bucket(size=self.params['batch_size'])

        self.X = []
        self.y = []



    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.model.save(filepath=self.model_save_path+'/'+self.model_name)

    def save_Xy(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        with open(self.model_save_path+'/model.X', 'wb') as x_file:
            pickle.dump(x_file, self.model_save_path+'/model.X')
        with open(self.model_save_path+'/model.y', 'wb') as y_file:
            pickle.dump(y_file, self.model_save_path+'/model.y')

    def load_model(self):
        self.model = load_model(filepath=self.model_save_path+'/'+self.model_name)

    def feed(self, row):

        image = row['image']
        label = row['label']
        date = row['date']
        self.X.append(image)
        self.y.append(label)

        self.X_bucket.put(image)
        self.y_bucket.put(label)

        if self.X_bucket.full():
            images = self.X_bucket.get_all_bucket()
            images = np.array(images)
            images = images.reshape(images.shape[0], self.params["input_w"], self.params["input_h"], 1)
            labels = np.array(self.y_bucket.get_all_bucket())

            history = self.model.fit(x=images, y=labels, validation_data=None, epochs=1, verbose=False)

            if self.verbose:
                print('date: {}  ||  MSE: {}'.format(date, history.history['mean_squared_error']))

            # flush all bucket because we need to be prepared for next batch
            self.X_bucket.flush()
            self.y_bucket.flush()

    def retrain(self):
        if self.verbose:
            print("Starting retraining...")

        X = np.array(self.X)
        X = X.reshape(X.shape[0], self.params["input_w"], self.params["input_h"], 1)
        y = np.array(self.y)
        self.model.fit(x=X, y=y, verbose=self.verbose, validation_data=None, batch_size=self.params['batch_size'],
                       epochs=self.params['epochs'] - 1)

    def construct_cnn(self):
        # CNN model
        self.model = Sequential()
        self.model.add(
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.params["input_w"], self.params["input_h"], 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.params["num_classes"]))
        self.model.compile(loss=mean_squared_error,
                           optimizer=Adadelta(),
                           metrics=['mse', 'mae'])

        return self.model


    def train(self, X,y):
        # todo: implement
        X = np.array(X)
        X.reshape(X.shape[0], self.params["input_w"], self.params["input_h"], 1)
        y = np.array(y)


        self.model.fit(X, y, batch_size=self.params["batch_size"], epochs=self.params["epochs"], verbose=1,
                  validation_data=None)
        pass

    def test(self, data, q_ratio=0.38):
        raise Exception("Not implemented yet!")
        test_images = data['test_images']
        test_labels = data['test_labels']
        test_names = data['test_names']
        test_dates = data['test_dates']

        test_images = test_images.as_matrix().reshape(test_images.shape[0], self.params["input_w"],
                                                      self.params["input_h"], 1)

        precisions = []
        accuracies = []
        losses = []

        predictions = []
        names = []
        dates = []
        actuals = []
        mses = []

        cur_date = test_dates.iloc[0][0]
        train_again_images = []
        train_again_labels = []
        print("Calculating accuracy day by day...", end='\n\n')
        for i, (image, label, name, date) in enumerate(zip(test_images, test_labels[0], test_names[0], test_dates[0])):

            image = image.reshape((1, self.params["input_w"], self.params["input_h"], 1))
            label = label.reshape((1, self.params["num_classes"]))
            # test for next image

            if cur_date != date:  # update model
                for train_image, train_label in zip(train_again_images, train_again_labels):
                    # train with only 1 more image
                    self.model.train_on_batch(train_image, train_label)
                train_again_images = []
                train_again_labels = []
                cur_date = date

            prediction, mse, acc_cur = self.custom_test_on_batch(image, label, q_ratio=q_ratio)
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

        pred_df = pd.DataFrame(
            {'Name': np.asarray(names), 'Date': np.asarray(dates), 'Prediction': np.asarray(predictions),
             'Actual': np.asarray(actuals)})
        return pred_df
        # history = {'prediction': predictions, 'loss': losses, 'acc': accuracies }
        # return model, history

    def custom_test_on_batch(self, image, label, q_ratio=0.38):
        raise Exception("Not implemented yet!")
        prediction = self.model.predict(image)
        mse = (label - prediction) ** 2
        p_q = quantize(prediction, q_ratio)
        r_q = quantize(label, q_ratio)
        if p_q == r_q:
            return prediction, mse, 1
        else:
            return prediction, mse, 0
