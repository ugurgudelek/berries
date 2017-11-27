"""Main engine class"""
import datetime
import pandas as pd
import numpy as np

from data import Data

from tqdm import tqdm


class Engine:
    def __init__(self, financeIO, metric_engine, label_engine, image_engine, cnn_engine, stock_names,
                 make_stationary=True, apply_tanh=True, verbose=False):
        self.financeIO = financeIO
        self.metric_engine = metric_engine
        self.label_engine = label_engine
        self.image_engine = image_engine
        self.cnn_engine = cnn_engine

        self.stock_names = stock_names

        self.make_stationary = make_stationary
        self.old_closes = dict()  # to make stationary

        self.apply_tanh = apply_tanh

        self.verbose = verbose

    def feed(self, current_date):
        """This is the core method.
        for each stock in self.stock_names:
        1. Calculates Metrics
        2. Retrieve Label, if metric is proper(mean does not have any None).
        3. Form feature data with metrics + close + volume
        4. Constructs Image.
        5. Trains the model, if the image is proper."""
        if self.verbose:
            print("Date: {}".format(current_date))

        data = Data()
        data.date = current_date

        for stock_name in self.stock_names:
            if self.verbose:
                print("Name: {}".format(stock_name))
            data.stock_name = stock_name

            current_day_data = self.financeIO.get_one_day_data(stock_name=stock_name, date=current_date)
            data.day_data = current_day_data

            if current_day_data is not None:  # if current day is business day
                row = {'stock_name': stock_name,
                       'date': current_date,
                       'close': current_day_data['close'].values[0],
                       'high': current_day_data['high'].values[0],
                       'low': current_day_data['low'].values[0],
                       'volume': current_day_data['volume'].values[0]}
                current_metric_data = self.metric_engine.feed(row=row)  # try to calculate metrics
                data.metric_data = current_metric_data
                if current_metric_data is not None:

                    # get label for next business day
                    current_label = self.label_engine.get_label_for(stock_name=stock_name, date=current_date,
                                                                    old_close=current_day_data['close'].values[0])

                    data.label = current_label

                    # create new image
                    current_feature_data = current_metric_data

                    if self.make_stationary:
                        old_close = self.old_closes.get(stock_name, 0.0)
                        current_close = current_day_data['close'].values[0]
                        stationary_close = (current_close - old_close) / current_close

                        if self.apply_tanh:
                            current_feature_data['close'] = np.tanh(stationary_close)
                        else:
                            current_feature_data['close'] = stationary_close
                    else:
                        current_feature_data['close'] = current_day_data['close'].values[0]

                    current_feature_data['volume'] = current_day_data['volume'].values[0]

                    row = {'date': current_date, 'stock_name': stock_name, 'data': current_feature_data}

                    data.feature_data = current_feature_data

                    current_image = self.image_engine.feed(row=row)
                    data.image = current_image

                    if current_image is not None:
                        # all None checks has passed and we have proper image now
                        # so we can train our model now.
                        # but we need train this model again later for other epochs
                        row = {'date': current_date,'stock_name': stock_name, 'image': current_image, 'label': current_label}
                        self.cnn_engine.feed(row=row)
                        

                self.old_closes[stock_name] = current_day_data['close'].values[0]  # update old close





    def run(self, start_date, end_date):

        current_date = start_date

        with tqdm((end_date - start_date).days) as progress_bar:  # create progress bar
            while current_date <= end_date:  # loop for each date
                self.feed(current_date=current_date)

                # now we can increment to next day
                current_date += datetime.timedelta(days=1)

                # update progress bar
                progress_bar.update(n=1)

        # now lets train our model for other epochs
        self.cnn_engine.retrain()

        # save the trained model
        self.cnn_engine.save_model()

        # save X and y file
        self.cnn_engine.save_Xy()

        # save all containers in engines
        # self.financeIO.save_containers()
        # self.metric_engine.save_containers()
        # self.label_engine.save_containers()
        # self.image_engine.save_containers()
        # self.cnn_engine.save_containers()
