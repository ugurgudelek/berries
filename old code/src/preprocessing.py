import os
import pandas as pd
import numpy as np
import clustering
import classes
import metrics as mt
import time

from helper import *

def get_last_saved_data(read_path="../input/last_saved_data"):
    return pd.read_pickle(read_path+"/last_saved.pickle")


def apply_normalization_to_raw_data(stock):
    # insert percent change column into main dataframe
    percentage = stock.adjusted_close.pct_change(periods=1)
    percentage.iloc[0] = 0.0
    percentage_100 = percentage * 100
    stock['pct_change_tanh'] = percentage_100.apply(np.tanh)
    return stock


# 2.
def normalize_and_calculate_metrics(stock_names, raw_data_path="../input/raw_data",
                                   path_to_save="../input/stock_with_metrics"):
    """Downloads data, calculates metrics and save results to separate .csv files for each ETF."""

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for stock_name in stock_names:
        # skip if exist
        if stock_name + ".csv" in os.listdir(path_to_save):
            print(path_to_save + "/{}.csv is already exist. Delete file to recompile".format(stock_name))
            continue

        # verbose message
        print("{}.csv will be created with metric and pct_change ".format(stock_name))

        # read stock csv
        stock = pd.read_csv(raw_data_path + "/{}.csv".format(stock_name))

        # create data arr to hold all metric info
        metric_data, metric_function_names = calculate_metrics(stock)

        # assign nan value beginning of the data
        metric_data = assign_null_into_data(arr=metric_data, length=len(stock['adjusted_close']))

        # append data and metrics column-wise
        stock = stack_data_and_metrics(stock, metric_data, metric_function_names)

        # normalize price values before applying labels
        # because we need to get rid of diversity among stocks
        stock = apply_normalization_to_raw_data(stock)

        # save
        stock.to_csv(path_to_save + "/{}.csv".format(stock_name), index=None)


# 3.
def calculate_labels(stock_names, period=28, stock_with_metric_path="../input/stock_with_metrics",
                                   stock_with_labels_path="../input/stock_with_labels"):
    """Calculates labels for both regression and 2-class classification"""
    if not os.path.exists(stock_with_labels_path):
        os.makedirs(stock_with_labels_path)

    for stock_name in stock_names:
        # skip if exist
        if stock_name + ".csv" in os.listdir(stock_with_labels_path):
            print(stock_with_labels_path + "/{}.csv is already exist. Delete file to recompile".format(stock_name))
            continue

        # read stock_metric csv
        stock = pd.read_csv(stock_with_metric_path + "/{}.csv".format(stock_name))



        target_day_class = classes.day_by_day_classes(stock['adjusted_close'].values, period)
        stock['label_day_is_less'] = target_day_class[:, 0]
        stock['label_day_is_more'] = target_day_class[:, 1]

        # tanh(percentage_change) used because of that normalization is must
        target_day_regr = classes.day_by_day_reg(stock['pct_change_tanh'], period)
        stock['label_day_tanh_regr'] = target_day_regr

        stock.to_csv(stock_with_labels_path + "/{}.csv".format(stock_name), index=None)

def calculate_labels_3class(stock_names, thr_1 = -.38, thr_2 = 0.38, period=28, stock_with_metric_path="../input/stock_with_metrics",
                                   stock_with_labels_path="../input/stock_with_labels"):
    if not os.path.exists(stock_with_labels_path):
        os.makedirs(stock_with_labels_path)

    for stock_name in stock_names:
        # skip if exist
        if stock_name + ".csv" in os.listdir(stock_with_labels_path):
            print(stock_with_labels_path + "/{}.csv is already exist. Delete file to recompile".format(stock_name))
            continue

        # read stock_metric csv
        stock = pd.read_csv(stock_with_metric_path + "/{}.csv".format(stock_name))

        # tanh(percentage_change) used because of that normalization is must
        target_day_regr = classes.day_by_day_reg(stock['pct_change_tanh'], period)
        stock['label_day_tanh_less'] = (target_day_regr < thr_1).values.astype(float)
        stock['label_day_tanh_inrange'] = (np.logical_and(target_day_regr > thr_1, target_day_regr < thr_2)).values.astype(float)
        stock['label_day_tanh_more'] = (target_day_regr > thr_2).values.astype(float)

        stock.to_csv(stock_with_labels_path + "/{}.csv".format(stock_name), index=None)


# 4.
def cluster_features(p_stock_names, drop_this_cols, hierarcy_no_plot=True,
                     stock_with_metric_path="../input/stock_with_metrics", save_path="../input/"):
    """Calls the clustering function after calculation of the metrics for all the ETFs.
    This function should be called after the metrics are calculated and saved to .csv files
    but before the images are constructed."""
    # skip if exist
    if save_path + "/clustered_names.csv" in os.listdir(save_path):
        print("{} is already exist. Delete file to recompile".format(save_path + "/clustered_names.csv"))
        return pd.read_csv(save_path + "/clustered_names.csv", header=False, index=False)

    # read the first csv
    raw_data = pd.read_csv(stock_with_metric_path + "/{}.csv".format(p_stock_names[0]))

    # drop irrelevant features
    data = raw_data.drop(drop_this_cols, axis=1)

    # get predictor names for dropping processes
    predictor_names = [name for name in data.columns.values.tolist() if "label" not in name]
    data = data.dropna(subset=predictor_names)  # drop nan values for proper set

    # all data will be appended to this dataframe
    all_data = data

    for stock in p_stock_names[1:len(p_stock_names)]:
        raw_data = pd.read_csv(stock_with_metric_path + "/{}.csv".format(stock))

        # drop irrelevant features
        data = raw_data.drop(drop_this_cols, axis=1)

        # get predictor names for dropping processes
        predictor_names = [name for name in data.columns.values.tolist() if "label" not in name]
        data = data.dropna(subset=predictor_names)  # drop nan values for proper set

        all_data = all_data.append(data)

    # now, cluster features of the whole data
    sorted_predictor_names = clustering.hierarchical_clustering(all_data[predictor_names], no_plot=hierarcy_no_plot)

    # save the names of the clustered features to file
    pd.Series(sorted_predictor_names).to_csv(save_path + "/clustered_names.csv", header=False, index=False)

    # return the names of the clustered features
    return sorted_predictor_names


# 6.
def create_images_from_data(stock_names, sorted_cluster_names, label_names, split_period=28,
                            stock_with_labels_path="../input/stock_with_labels",
                            save_path="../input/images_with_labels"):
    """
    Reads metric data, clusters features, prepares and returns images together with labels.
    Images are flattened before returned.

    :param :type str which_stock: takes stock names

    :param :type int split_period=28 determines chuck size wrt date
    :param :type list label_names select label to use later for prediction
    :param :type bool cluster true: call clustering or false: use csv file

    :returns (list)images, (list)labels, (tuple)(image_row_size,image_col_size)
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for stock_name in stock_names:

        # skip if exist
        if stock_name + ".csv" in os.listdir(save_path):
            print(save_path + "/{}.csv is already exist. Delete file to recompile".format(stock_name))
            continue

        print("images are creating for {} ... ".format(stock_name), end='')
        start_time = time.time()
        data = pd.read_csv(stock_with_labels_path + "/{}.csv".format(stock_name))
        predictor_names = sorted_cluster_names

        # drop irrelevant features
        data = data[['date'] + predictor_names + label_names]
        # when i do this, later i can reach data with stock name and date


        # drop nan values for proper set
        data = data.dropna()

        image_col_size = data[predictor_names].shape[1]
        image_row_size = split_period

        if image_row_size != image_col_size:
            raise Exception("image matrix must be square!")

        images = []
        labels = []
        dates = []
        names = []
        # split image chunks
        for i in range(split_period - 1, data.shape[0]):
            lower = i - split_period + 1
            upper = lower + split_period

            image = data[predictor_names].iloc[lower:upper]

            # normalization for image.
            image = (image - image.mean()) / image.std()

            image_flat = image.values.flatten()  # image_flat'shape : image_row_size * image_col_size
            label = data[label_names].iloc[upper - 1].values
            date = data['date'].iloc[upper-1]


            images.append(image_flat)
            labels.append(label)
            dates.append(date)
            names.append(stock_name)

        # save data into one big dataframe
        # period * period + label_size for each row
        # image_count = row size

        data_df = pd.concat([pd.Series(names),
                             pd.Series(dates),
                             pd.DataFrame(images,dtype='float32'),
                             pd.DataFrame(labels,dtype='float32')],ignore_index=True,axis=1)


        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data_df.to_csv(save_path + "/{}.csv".format(stock_name), index=False, header=None)

        print("Elapsed time : {}".format(time.time() - start_time))

        # return images, labels, (image_row_size, image_col_size)


# 7.
def get_merged_images_and_labels_data(stock_names, read_path="../input/images_with_labels", labels_are_last=1,
                                      train_test_ratio=0.9, save_path="../input/last_saved_data"):
    
    if os.path.isfile(save_path+"/last_saved.pickle"):
        return pd.read_pickle(save_path+"/last_saved.pickle")
    
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []
    all_train_names = []
    all_test_names = []
    all_train_dates=[]
    all_test_dates=[]

    # todo: burada şu strig colon işini çöz
    for stock in stock_names:
        data_df = pd.read_csv(read_path + "/{}.csv".format(stock), header=None)
        names = data_df.iloc[:, 0] # first element
        dates = data_df.iloc[:, 1] # second element
        images = data_df.iloc[:, 2:-labels_are_last] # remaining elements
        labels = data_df.iloc[:, -labels_are_last:] # last elements

        print("all images are merging with {} ...".format(stock))

        # determine where to split
        image_count = images.shape[0]
        train_image_count = int(image_count * train_test_ratio)
        test_image_count = image_count - train_image_count

        # split train and test
        # for 16 year of data : nearly 14 year train-last 2 year test
        train_images = images.iloc[0:train_image_count]
        test_images = images.iloc[train_image_count:]
        train_labels = labels.iloc[0:train_image_count]
        test_labels = labels.iloc[train_image_count:]

        train_names = names.iloc[0:train_image_count]
        train_dates = dates.iloc[0:train_image_count]
        test_names = names.iloc[train_image_count:]
        test_dates = dates.iloc[train_image_count:]

        # todo: need to make data class because above not seems good. -ugurgudelek

        if len(all_train_images) == 0:
            all_train_images = np.array(train_images)
            all_train_labels = train_labels.values
            all_test_images = np.array(test_images)
            all_test_labels = test_labels.values

            all_train_names = np.array(train_names)
            all_test_names = np.array(test_names)
            all_train_dates = np.array(train_dates)
            all_test_dates = np.array(test_dates)
        else:
            all_train_images = np.append(all_train_images, train_images, axis=0)
            all_train_labels = np.append(all_train_labels, train_labels.values, axis=0)
            all_test_images = np.append(all_test_images, test_images, axis=0)
            all_test_labels = np.append(all_test_labels, test_labels.values, axis=0)

            all_train_names = np.append(all_train_names,train_names, axis=0)
            all_test_names =  np.append(all_test_names, test_names, axis=0)
            all_train_dates = np.append(all_train_dates,train_dates, axis=0)
            all_test_dates =  np.append(all_test_dates, test_dates, axis=0)

        print("current train shape is {} and {} label ".format(pd.DataFrame(all_train_images).shape,
                                                               all_train_labels.shape[1]))
        print("current test shape is {} and {} label ".format(pd.DataFrame(all_test_images).shape,
                                                              all_test_labels.shape[1]))
    
    print("Sorting train data by date and name...")
    sorted_train_data = pd.DataFrame()
    sorted_train_data['date'] = all_train_dates
    sorted_train_data['name'] = all_train_names
    sorted_train_data['image'] = [i for i in all_train_images]
    sorted_train_data['label'] = all_train_labels
    sorted_train_data = sorted_train_data.sort_values(by = ['date', 'name'])
    
    
    print("Sorting test data by date and name...")
    sorted_test_data = pd.DataFrame()
    sorted_test_data['date'] = all_test_dates
    sorted_test_data['name'] = all_test_names
    sorted_test_data['image'] = [i for i in all_test_images]
    sorted_test_data['label'] = all_test_labels
    sorted_test_data = sorted_test_data.sort_values(by = ['date', 'name'])

    data = {'train_images': pd.DataFrame(np.asarray([i for i in sorted_train_data['image']])),
            'test_images': pd.DataFrame(np.asarray([i for i in sorted_test_data['image']])),
            'train_labels': pd.DataFrame(sorted_train_data['label'].values),
            'test_labels': pd.DataFrame(sorted_test_data['label'].values),
            'train_names': pd.DataFrame(sorted_train_data['name'].values),
            'train_dates': pd.DataFrame(sorted_train_data['date'].values),
            'test_names': pd.DataFrame(sorted_test_data['name'].values),
            'test_dates': pd.DataFrame(sorted_test_data['date'].values)
    }    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pd.to_pickle(data, save_path+"/last_saved.pickle")

    return data

# merging images and labels for classification
def get_merged_images_and_labels_data_cls(stock_names, read_path="../input/images_with_labels", last_image_col = -3, labels_ind = [-2, -1],
                                      train_test_ratio=0.9, save_path="../input/last_saved_data"):

    if os.path.isfile(save_path+"/last_saved.pickle"):
        return pd.read_pickle(save_path+"/last_saved.pickle")
    
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []
    all_train_names = []
    all_test_names = []
    all_train_dates=[]
    all_test_dates=[]

    # todo: burada şu strig colon işini çöz
    for stock in stock_names:
        data_df = pd.read_csv(read_path + "/{}.csv".format(stock), header=None)
        names = data_df.iloc[:, 0] # first element
        dates = data_df.iloc[:, 1] # second element
        images = data_df.iloc[:, 2:(last_image_col + 1)] # remaining elements
        labels = pd.concat([data_df.iloc[:,labels_ind]], axis = 1)            

        print("all images are merging with {} ...".format(stock))

        # determine where to split
        image_count = images.shape[0]
        train_image_count = int(image_count * train_test_ratio)
        test_image_count = image_count - train_image_count

        # split train and test
        # for 16 year of data : nearly 14 year train-last 2 year test
        train_images = images.iloc[0:train_image_count]
        test_images = images.iloc[train_image_count:]
        train_labels = labels.iloc[0:train_image_count]
        test_labels = labels.iloc[train_image_count:]

        train_names = names.iloc[0:train_image_count]
        train_dates = dates.iloc[0:train_image_count]
        test_names = names.iloc[train_image_count:]
        test_dates = dates.iloc[train_image_count:]

        # todo: need to make data class because above not seems good. -ugurgudelek

        if len(all_train_images) == 0:
            all_train_images = np.array(train_images)
            all_train_labels = train_labels.values
            all_test_images = np.array(test_images)
            all_test_labels = test_labels.values

            all_train_names = np.array(train_names)
            all_test_names = np.array(test_names)
            all_train_dates = np.array(train_dates)
            all_test_dates = np.array(test_dates)
        else:
            all_train_images = np.append(all_train_images, train_images, axis=0)
            all_train_labels = np.append(all_train_labels, train_labels.values, axis=0)
            all_test_images = np.append(all_test_images, test_images, axis=0)
            all_test_labels = np.append(all_test_labels, test_labels.values, axis=0)

            all_train_names = np.append(all_train_names,train_names, axis=0)
            all_test_names =  np.append(all_test_names, test_names, axis=0)
            all_train_dates = np.append(all_train_dates,train_dates, axis=0)
            all_test_dates =  np.append(all_test_dates, test_dates, axis=0)

        print("current train shape is {} and {} label ".format(pd.DataFrame(all_train_images).shape,
                                                               all_train_labels.shape[1]))
        print("current test shape is {} and {} label ".format(pd.DataFrame(all_test_images).shape,
                                                              all_test_labels.shape[1]))

    print("Sorting test data by date and name...")
    sorted_test_data = pd.DataFrame()
    sorted_test_data['date'] = all_test_dates
    sorted_test_data['name'] = all_test_names
    sorted_test_data['image'] = [i for i in all_test_images]
    sorted_test_data['label'] = [i for i in all_test_labels]
    sorted_test_data = sorted_test_data.sort_values(by = ['date', 'name'])

    data = {'train_images': pd.DataFrame(all_train_images),
            'test_images': pd.DataFrame(np.asarray([i for i in sorted_test_data['image']])),
            'train_labels': pd.DataFrame(all_train_labels),
            'test_labels': pd.DataFrame(np.asarray([i for i in sorted_test_data['label']])),
            'train_names': pd.DataFrame(all_train_names),
            'train_dates': pd.DataFrame(all_train_dates),
            'test_names': pd.DataFrame(sorted_test_data['name'].values),
            'test_dates': pd.DataFrame(sorted_test_data['date'].values)
    }

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pd.to_pickle(data, save_path+"/last_saved.pickle")

    return data


def calculate_metrics(data):
    metric_function_data = []

    rsi_15 = mt.rsi(data, 15);
    metric_function_data.append(rsi_15)
    rsi_20 = mt.rsi(data, 20);
    metric_function_data.append(rsi_20)
    rsi_25 = mt.rsi(data, 25);
    metric_function_data.append(rsi_25)
    rsi_30 = mt.rsi(data, 30);
    metric_function_data.append(rsi_30)
    # rsi_35 = mt.rsi(data, 35); metric_function_data.append(rsi_35)
    # rsi_40 = mt.rsi(data, 40); metric_function_data.append(rsi_40)
    # rsi_45 = mt.rsi(data, 45); metric_function_data.append(rsi_45)
    #    rsi_50 = mt.rsi(data, 50); metric_function_data.append(rsi_50)

    sma_15 = mt.sma(data, 15);
    metric_function_data.append(sma_15)
    sma_20 = mt.sma(data, 20);
    metric_function_data.append(sma_20)
    sma_25 = mt.sma(data, 25);
    metric_function_data.append(sma_25)
    sma_30 = mt.sma(data, 30);
    metric_function_data.append(sma_30)
    # sma_35 = mt.sma(data, 35); metric_function_data.append(sma_35)
    # sma_40 = mt.sma(data, 40); metric_function_data.append(sma_40)
    # sma_45 = mt.sma(data, 45); metric_function_data.append(sma_45)
    #    sma_50 = mt.sma(data, 50); metric_function_data.append(sma_50)

    macd_26_12 = mt.macd(data, 26, 12);
    metric_function_data.append(macd_26_12)
    macd_28_14 = mt.macd(data, 28, 14);
    metric_function_data.append(macd_28_14)
    macd_30_16 = mt.macd(data, 30, 16);
    metric_function_data.append(macd_30_16)
    # macd_32_18 = mt.macd(data, 32, 18); metric_function_data.append(macd_32_18)
    # macd_32_20 = mt.macd(data, 32, 20); metric_function_data.append(macd_32_20)
    # macd_34_22 = mt.macd(data, 34, 22); metric_function_data.append(macd_34_22)
    # macd_36_24 = mt.macd(data, 36, 24); metric_function_data.append(macd_36_24)
    #    macd_38_26 = mt.macd(data, 38, 26); metric_function_data.append(macd_38_26)

    macd_trigger_9_26_12 = mt.macd_trigger(data, 9, 26, 12);
    metric_function_data.append(macd_trigger_9_26_12)
    macd_trigger_10_28_14 = mt.macd_trigger(data, 10, 28, 14);
    metric_function_data.append(macd_trigger_10_28_14)
    macd_trigger_11_30_16 = mt.macd_trigger(data, 11, 30, 16);
    metric_function_data.append(macd_trigger_11_30_16)
    # macd_trigger_12_32_18 = mt.macd_trigger(data, 12, 32, 18); metric_function_data.append(macd_trigger_12_32_18)
    # macd_trigger_13_34_20 = mt.macd_trigger(data, 13, 34, 20); metric_function_data.append(macd_trigger_13_34_20)
    # macd_trigger_14_36_22 = mt.macd_trigger(data, 14, 36, 22); metric_function_data.append(macd_trigger_14_36_22)
    # macd_trigger_15_38_24 = mt.macd_trigger(data, 15, 38, 24); metric_function_data.append(macd_trigger_15_38_24)
    #    macd_trigger_16_40_26 = mt.macd_trigger(data, 16, 40, 26); metric_function_data.append(macd_trigger_16_40_26)

    willR_14 = mt.williamsR(data, 14);
    metric_function_data.append(willR_14)
    willR_18 = mt.williamsR(data, 18);
    metric_function_data.append(willR_18)
    willR_22 = mt.williamsR(data, 22);
    metric_function_data.append(willR_22)
    # willR_26 = mt.williamsR(data, 26); metric_function_data.append(willR_26)
    # willR_30 = mt.williamsR(data, 30); metric_function_data.append(willR_30)
    # willR_34 = mt.williamsR(data, 34); metric_function_data.append(willR_34)
    # willR_38 = mt.williamsR(data, 38); metric_function_data.append(willR_38)
    #    willR_42 = mt.williamsR(data, 42); metric_function_data.append(willR_42)

    kdHist_14 = mt.kdDiff(data, 14);
    metric_function_data.append(kdHist_14)
    kdHist_18 = mt.kdDiff(data, 18);
    metric_function_data.append(kdHist_18)
    kdHist_22 = mt.kdDiff(data, 22);
    metric_function_data.append(kdHist_22)
    # kdHist_26 = mt.kdDiff(data, 26); metric_function_data.append(kdHist_26)
    # kdHist_30 = mt.kdDiff(data, 30); metric_function_data.append(kdHist_30)
    # kdHist_34 = mt.kdDiff(data, 34); metric_function_data.append(kdHist_34)
    # kdHist_38 = mt.kdDiff(data, 38); metric_function_data.append(kdHist_38)
    #    kdHist_42 = mt.kdDiff(data, 42); metric_function_data.append(kdHist_42)

    ultimateOs_7_14_28 = mt.ulOs(data, 7, 14, 28);
    metric_function_data.append(ultimateOs_7_14_28)
    ultimateOs_8_16_32 = mt.ulOs(data, 8, 16, 32);
    metric_function_data.append(ultimateOs_8_16_32)
    ultimateOs_9_18_36 = mt.ulOs(data, 9, 18, 36);
    metric_function_data.append(ultimateOs_9_18_36)
    # ultimateOs_10_20_40 = mt.ulOs(data, 10, 20, 40); metric_function_data.append(ultimateOs_10_20_40)
    # ultimateOs_11_22_44 = mt.ulOs(data, 11, 22, 44); metric_function_data.append(ultimateOs_11_22_44)
    # ultimateOs_12_24_48 = mt.ulOs(data, 12, 24, 48); metric_function_data.append(ultimateOs_12_24_48)
    # ultimateOs_13_26_52 = mt.ulOs(data, 13, 26, 52); metric_function_data.append(ultimateOs_13_26_52)
    #    ultimateOs_14_28_56 = mt.ulOs(data, 14, 28, 56); metric_function_data.append(ultimateOs_14_28_56)

    mfIndex_14 = mt.mfi(data, 14);
    metric_function_data.append(mfIndex_14)
    mfIndex_18 = mt.mfi(data, 18);
    metric_function_data.append(mfIndex_18)
    mfIndex_22 = mt.mfi(data, 22);
    metric_function_data.append(mfIndex_22)
    # mfIndex_26 = mt.mfi(data, 26); metric_function_data.append(mfIndex_26)
    # mfIndex_30 = mt.mfi(data, 30); metric_function_data.append(mfIndex_30)
    #    mfIndex_34 = mt.mfi(data, 34); metric_function_data.append(mfIndex_34)
    #    mfIndex_38 = mt.mfi(data, 38); metric_function_data.append(mfIndex_38)
    #    mfIndex_40 = mt.mfi(data, 40); metric_function_data.append(mfIndex_40)

    # metric_function_names = ["rsi_15","rsi_20","rsi_25","rsi_30","rsi_35","rsi_40","rsi_45","rsi_50",
    #                         "sma_15","sma_20","sma_25","sma_30","sma_35","sma_40","sma_45","sma_50",
    #                         "macd_26_12","macd_28_14","macd_30_16","macd_32_18","macd_32_20","macd_34_22","macd_36_24","macd_38_26",
    #                         "macd_trigger_9_26_12","macd_trigger_10_28_14","macd_trigger_11_30_16","macd_trigger_12_32_18","macd_trigger_13_34_20",
    #                          "macd_trigger_14_36_22","macd_trigger_15_38_24","macd_trigger_16_40_26",
    #                         "willR_14","willR_18","willR_22","willR_26","willR_30","willR_34","willR_38","willR_42",
    #                         "kdHist_14","kdHist_18","kdHist_22","kdHist_26","kdHist_30","kdHist_34","kdHist_38","kdHist_42",
    #                         "ultimateOs_7_14_28","ultimateOs_8_16_32","ultimateOs_9_18_36","ultimateOs_10_20_40","ultimateOs_11_22_44","ultimateOs_12_24_48",
    #                          "ultimateOs_13_26_52","ultimateOs_14_28_56", "mfIndex_14","mfIndex_18","mfIndex_22","mfIndex_26","mfIndex_30","mfIndex_34","mfIndex_38","mfIndex_40"]

    # metric_function_names = ["rsi_15","rsi_20","rsi_25","rsi_30","rsi_35","rsi_40","rsi_45",
    #                         "sma_15","sma_20","sma_25","sma_30","sma_35","sma_40","sma_45",
    #                         "macd_26_12","macd_28_14","macd_30_16","macd_32_18","macd_32_20","macd_34_22","macd_36_24",
    #                         "macd_trigger_9_26_12","macd_trigger_10_28_14","macd_trigger_11_30_16","macd_trigger_12_32_18","macd_trigger_13_34_20",
    #                          "macd_trigger_14_36_22","macd_trigger_15_38_24",
    #                         "willR_14","willR_18","willR_22","willR_26","willR_30","willR_34","willR_38",
    #                         "kdHist_14","kdHist_18","kdHist_22","kdHist_26","kdHist_30","kdHist_34","kdHist_38",
    #                         "ultimateOs_7_14_28","ultimateOs_8_16_32","ultimateOs_9_18_36","ultimateOs_10_20_40","ultimateOs_11_22_44","ultimateOs_12_24_48",
    #                          "ultimateOs_13_26_52", "mfIndex_14","mfIndex_18","mfIndex_22","mfIndex_26","mfIndex_30"]
    metric_function_names = ["rsi_15", "rsi_20", "rsi_25", "rsi_30",
                             "sma_15", "sma_20", "sma_25", "sma_30",
                             "macd_26_12", "macd_28_14", "macd_30_16",
                             "macd_trigger_9_26_12", "macd_trigger_10_28_14", "macd_trigger_11_30_16",
                             "willR_14", "willR_18", "willR_22",
                             "kdHist_14", "kdHist_18", "kdHist_22",
                             "ultimateOs_7_14_28", "ultimateOs_8_16_32", "ultimateOs_9_18_36",
                             "mfIndex_14", "mfIndex_18", "mfIndex_22"]

    return np.asarray(metric_function_data), metric_function_names
