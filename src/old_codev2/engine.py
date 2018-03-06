"""Main engine class"""
import datetime
import pandas as pd
import numpy as np
import os
from data import Data,DataHolder

from tqdm import tqdm
from utils import timeit


class Engine:
    def __init__(self, financeIO, metric_engine, label_engine, image_engine, cnn_engine, stock_names,
                 instance_path,run_number,
                 make_stationary=True, apply_tanh=True, verbose=False, save_each_year=True):
        self.save_each_year = save_each_year
        self.instance_path = instance_path
        if not os.path.exists(self.instance_path):
            os.makedirs(self.instance_path)
        self.run_number = run_number
        if self.maybe_forget_to_increment_run_number():
            raise Exception("Do not forget to increment run_number!")

        self.dataholder = DataHolder()
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


    def maybe_forget_to_increment_run_number(self):
        instance_list = os.listdir(self.instance_path)
        if len(instance_list) == 0:
            return False
        last_filename = sorted(instance_list)[-1]
        last_run_number = int(last_filename[0])
        return last_run_number >= self.run_number

    # @timeit('engine')
    def _feed_one_stock(self, stock_name, current_date):
        """
        1. Calculates Metrics
        2. Retrieve Label, if metric is proper(mean does not have any None).
        3. Form feature data with metrics + close + volume
        4. Constructs Image.
        5. Trains the model, if the image is proper.

        :param (str) stock_name:
        :return: None
        """
        if self.verbose:
            print("Name: {}".format(stock_name))

        # create data instance
        data = Data()

        current_daydata = self.financeIO.get_one_day_data(stock_name=stock_name, date=current_date)

        if current_daydata is not None:  # if current day is business day
            # create class attribute for day data
            for key, item in current_daydata.iloc[0].to_dict().items():
                setattr(data, key, item)
            # save it for easy use
            data.daydata = current_daydata.iloc[0]

            current_metricdata = self.metric_engine.feed(row=data)  # try to calculate metrics

            if current_metricdata is not None:

                # create feature series
                # we can use this to save also
                data.features = pd.Series(current_metricdata.to_dict())

                # get label for next business day
                # (stationary_label, raw_label)
                (flabel, label) = self.label_engine.get_label_for(stock_name=stock_name, date=current_date,
                                                                  old_close=data.close)

                data.flabel = flabel
                data.label = label
                # save it for easy use
                data.labeldata = pd.Series({'label': data.label, 'flabel': data.flabel})

                if self.make_stationary:
                    old_close = self.old_closes.get(stock_name, 0.0)
                    current_close = data.close
                    stationary_close = (current_close - old_close) / current_close

                    if self.apply_tanh:
                        data.features['fclose'] = np.tanh(stationary_close)
                    else:
                        data.features['fclose'] = stationary_close
                else:
                    data.features['fclose'] = data.close

                data.features['fvolume'] = data.volume

                # try to create image
                current_image = self.image_engine.feed(row=data)

                if current_image is not None:
                    # all None checks has passed and we have proper image now
                    data.image = pd.Series(current_image)  # we can use this to save
                    # so we can train our model.
                    # but here we've trained only 1 epoch, so model need to be trained again later for other epochs

                    # maybe we do not want to train the model here.
                    if self.cnn_engine is not None:
                        self.cnn_engine.feed(row=data)

                    # save the proper data
                    # 1. daydata
                    # 2. labeldata
                    # 3. features
                    # 4. image
                    data.series = pd.concat([data.daydata, data.labeldata, data.features, data.image], axis=0)
                    self.dataholder.append(data.series)

            self.old_closes[stock_name] = data.close  # update old close

    # @timeit('engine')
    def feed(self, current_date):
        """This is the core method.
        for each stock in self.stock_names:
        """

        for stock_name in self.stock_names:
            self._feed_one_stock(stock_name=stock_name, current_date=current_date)


    def run(self, start_date, end_date):

        current_date = start_date
        current_year = current_date.year

        with tqdm((end_date - start_date).days) as progress_bar:  # create progress bar
            while current_date <= end_date:  # loop for each date
                if self.save_each_year:
                    if current_year != current_date.year: # when we move to next year
                        # save 1 year data
                        self.dataholder.save("../input/dataholder_{}.csv".format(current_year))
                        self.dataholder.reset_storage() # reset storage

                        current_year = current_date.year


                self.feed(current_date=current_date)

                # update progress bar
                progress_bar.update(n=1)
                progress_bar.set_description('Date: %s' % current_date)

                # now we can increment to next day
                current_date += datetime.timedelta(days=1)



        # maybe we do not want to train the model here.
        if self.cnn_engine is not None:
            # now lets train our model for other epochs
            self.cnn_engine.retrain()

            # save the trained model
            self.cnn_engine.save_model()

            # save X and y file
            # self.cnn_engine.save_Xy()

        # save all instances in engines
        self.save_instance()

        if not self.save_each_year:
            # save dataholder
            self.dataholder.save("../input/dataholder.csv")
        else:
            # save last year
            self.dataholder.save("../input/dataholder_{}.csv".format(current_year))
    def save_instance(self):


        self.financeIO.save_instance(filepath=self.instance_path, run_number=self.run_number)
        self.metric_engine.save_instance(filepath=self.instance_path, run_number=self.run_number)
        self.label_engine.save_instance(filepath=self.instance_path, run_number=self.run_number)
        self.image_engine.save_instance(filepath=self.instance_path, run_number=self.run_number)
        
    def load_instance(self):
        self.financeIO.load_instance(filepath=self.instance_path, run_number=self.run_number)
        self.metric_engine.load_instance(filepath=self.instance_path, run_number=self.run_number)
        self.label_engine.load_instance(filepath=self.instance_path, run_number=self.run_number)
        self.image_engine.load_instance(filepath=self.instance_path, run_number=self.run_number)

        #todo: fix TypeError: can't pickle _thread.lock objects
        # self.cnn_engine.save_instance(filepath=self.instance_path, run_number=self.run_number)

    def feed_chunk(self, start_date, end_date):
        data = self.financeIO.query_all(start_date,end_date)
        self.metric_engine.feed_chunk(data=data)



