"""Main engine class"""
import datetime
import pandas as pd
import numpy as np


class Engine:
    def __init__(self, financeIO, metric_engine, label_engine, image_engine, cnn_engine, stock_names, make_stationary=True, verbose=False):
        self.financeIO = financeIO
        self.metric_engine = metric_engine
        self.label_engine = label_engine
        self.image_engine = image_engine
        self.cnn_engine = cnn_engine

        self.stock_names = stock_names

        self.make_stationary = make_stationary
        self.old_closes = dict()  # to make stationary

        self.verbose = verbose

    def run(self, start_date, end_date):
        current_date = start_date

        while current_date <= end_date:  # loop for each date
            if self.verbose:
                print("Date: {}".format(current_date))
            for stock_name in self.stock_names:
                current_day_data = self.financeIO.get_one_day_data(stock_name=stock_name, date=current_date)
                if current_day_data is not None:  # means current day is business day
                    row = {'date': current_date, 'data': current_day_data['close'].values[0]}
                    current_metric_data = self.metric_engine.feed(row=row)  # try to calculate metrics

                    if not any(pd.isnull(current_metric_data)):  # if metric_data has not any None

                        # get label for next business day
                        current_label = self.label_engine.get_label_for(stock_name=stock_name, date=current_date,
                                                                        old_close=self.old_closes.get(stock_name, 0.0))

                        if current_label is None:  # it should not be the case but meh...
                            raise Exception("Oh shit!! label is None")

                        # create new image
                        current_feature_data = current_metric_data

                        if self.make_stationary:
                            old_close = self.old_closes.get(stock_name, 0.0)
                            current_close = current_day_data['close'].values[0]
                            stationary_close = (current_close - old_close) / current_close
                            self.old_closes[stock_name] = current_close  # update old close
                            tanh_close = np.tanh(stationary_close)
                            current_feature_data['close'] = tanh_close
                        else:
                            current_feature_data['close'] = current_day_data['close'].values[0]

                        current_feature_data['volume'] = current_day_data['volume'].values[0]

                        row = {'date': current_date, 'stock_name': stock_name, 'data': current_feature_data}

                        current_image = self.image_engine.feed(row=row)
                        if current_image is not None:
                            # all None checks has passed and we have proper image now
                            # so we can train our model now.
                            # but we need train this model again later for other epochs
                            row = {'date': current_date, 'image': current_image, 'label': current_label}
                            self.cnn_engine.feed(row=row)

                self.old_closes[stock_name] = current_day_data['close'].values[0]

            # now we can increment to next day
            current_date += datetime.timedelta(days=1)

        # now lets train our model for other epochs
        self.cnn_engine.retrain()
