from src import google_finance_io
import datetime
from src import preprocessing
import pandas as pd
import numpy as np
import os


def get_merged_images_and_labels_data(images_with_labels_path, start_date, end_date):
    all_images = []
    all_labels = []
    all_names = []
    all_dates = []

    # todo: burada şu strig colon işini çöz
    for stock in stock_names:
        data_df = pd.read_csv(images_with_labels_path + "/{}.csv".format(stock), header=None)
        data_df = data_df.loc[(data_df.iloc[:, 1] >= start_date.strftime("%Y-%m-%d")) & (
                               data_df.iloc[:, 1] <= end_date.strftime("%Y-%m-%d"))]
        names = data_df.iloc[:, 0]  # first element
        dates = data_df.iloc[:, 1]  # second element
        images = data_df.iloc[:, 2:-1]  # remaining elements
        labels = data_df.iloc[:, -1:]  # last elements

        print("all images are merging with {} ...".format(stock))

        # todo: need to make data class because above not seems good. -ugurgudelek

        if len(all_images) == 0:
            all_images = np.array(images)
            all_labels = labels.values
            all_names = np.array(names)
            all_dates = np.array(dates)
        else:

            all_images = np.append(all_images, images, axis=0)
            all_labels = np.append(all_labels, labels.values, axis=0)

            all_names = np.append(all_names, names, axis=0)
            all_dates = np.append(all_dates, dates, axis=0)

        print("current shape is {} and {} label ".format(pd.DataFrame(all_images).shape,
                                                         all_labels.shape[1]))

    print("Sorting data by date and name...")
    sorted_data = pd.DataFrame()
    sorted_data['date'] = all_dates
    sorted_data['name'] = all_names
    sorted_data['image'] = [i for i in all_images]
    sorted_data['label'] = all_labels
    sorted_data = sorted_data.sort_values(by=['date', 'name'])

    data = {
        'images': pd.DataFrame(np.asarray([i for i in sorted_data['image']])),
        'labels': pd.DataFrame(sorted_data['label'].values),
        'names': pd.DataFrame(sorted_data['name'].values),
        'dates': pd.DataFrame(sorted_data['date'].values)
    }

    return data


def test(model, data, params, update_model=True, q_ratio=0.0):
    def custom_test_on_batch(model, image, label, q_ratio=0.38):
        prediction = model.predict(image)
        mse = (label - prediction) ** 2
        p_q = quantize(prediction, q_ratio)
        r_q = quantize(label, q_ratio)
        if p_q == r_q:
            return prediction, mse, 1
        else:
            return prediction, mse, 0

    from src.helper import quantize

    test_images = data['images']
    test_labels = data['labels']
    test_names = data['names']
    test_dates = data['dates']

    test_images = test_images.as_matrix().reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)

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

        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))
        # test for next image

        if update_model:
            if cur_date != date:  # update model
                for train_image, train_label in zip(train_again_images, train_again_labels):
                    # train with only 1 more image
                    model.train_on_batch(train_image, train_label)
                train_again_images = []
                train_again_labels = []
                cur_date = date

        prediction, mse, acc_cur = custom_test_on_batch(model, image, label, q_ratio=q_ratio)
        # loss_cur,acc_cur = model.test_on_batch(image,label)
        if update_model:
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

    pred_df = pd.DataFrame({'Name': np.asarray(names), 'Date': np.asarray(dates), 'Prediction': np.asarray(predictions),
                            'Actual': np.asarray(actuals)})
    return pred_df
    # history = {'prediction': predictions, 'loss': losses, 'acc': accuracies }
    # return model, history

def input_processor(stock_names, input_path, is_train, start_date, end_date, train_start_date= None):



    subpath = "train"
    if not is_train:
        subpath = "test"



    raw_data_path = input_path+"/"+subpath+"/raw_data"
    stock_with_metrics_path = input_path+"/"+subpath+"/stock_with_metrics"
    clustered_save_path = input_path
    stock_with_labels_path = input_path+"/"+subpath+"/stock_with_labels"
    images_with_labels_path = input_path+"/"+subpath+"/images_with_labels"
    last_saved_path = input_path+"/"+subpath+"/last_saved"



    if is_train:
        # 1.download data
        google_finance_io.download_data(stock_names, start_date=start_date,
                                        end_date=end_date, verbose=True, path=raw_data_path)
    else:
        # 1.download data
        google_finance_io.download_data(stock_names, start_date=train_start_date,
                                        end_date=end_date, verbose=True, path=raw_data_path)


    # 2.calculate metric for available stocks and save them into csv file
    preprocessing.normalize_and_calculate_metrics(stock_names,
                                                  raw_data_path=raw_data_path,
                                                  path_to_save=stock_with_metrics_path)

    if is_train:
        # 4.cluster features for available stocks and their features then save them into csv file
        preprocessing.cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open', 'adjusted_close'],
                                   stock_with_metric_path=stock_with_metrics_path, save_path=clustered_save_path)



    # 3.calculate labels for available stocks and save them into csv file
    preprocessing.calculate_labels(stock_names,
                                   stock_with_metric_path=stock_with_metrics_path,
                                   stock_with_labels_path=stock_with_labels_path)

    # 5.read sorted (via hierarchical clustering) feature names from file
    sorted_cluster_names = pd.read_csv(clustered_save_path+"/clustered_names.csv", header=None, squeeze=True).values.tolist()

    # 6. create flatten images with data and labels.
    preprocessing.create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_tanh_regr'],
                                          stock_with_labels_path=stock_with_labels_path,
                                          save_path=images_with_labels_path)

    # # 7. merge all available data
    # # data has 'images' and 'labels'
    # data = preprocessing.get_merged_images_and_labels_data(stock_names, labels_are_last=1, train_test_ratio=0.9,
    #                                                        read_path=images_with_labels_path,save_path=last_saved_path)





    data = get_merged_images_and_labels_data(images_with_labels_path, start_date, end_date)
    return data





if __name__ == "__main__":
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    input_path = "../sanity_new"
    is_train = 0
    train_start_date = datetime.date(2000, 1, 1)
    train_end_date = datetime.date(2005,12,31)

    test_start_date = datetime.date(2006,1,1)
    test_end_date = datetime.date(2006, 12, 31)
    model_name = "model_sanity"
    version    = "001"

    # model parameters
    params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}



    if is_train:
        print("===Training====")
        #check paths
        if not os.path.exists(input_path + "/model"):
            os.makedirs(input_path + "/model")
        if os.path.exists(input_path + "/model/" + model_name+"_"+version):
            raise Exception("Increment version")

        data = input_processor(stock_names, input_path, is_train, train_start_date, train_end_date)
        from src.cnn_keras_regr import construct_cnn
        # construct cnn
        print("CNN constructing...")
        model = construct_cnn(params=params)

        # fit data
        print("CNN fit session started...")


        def fit(model, data, params):
            train_images = data['images'].as_matrix()
            train_labels = data['labels'].as_matrix()
            train_images = train_images.reshape(train_images.shape[0], params["input_w"], params["input_h"], 1)

            print("model will be trained with {}".format(train_images.shape))
            # fit the model to the training data
            print("Fitting model to the training data...")
            print("")
            model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
                      validation_data=None)

            return model

        model = fit(model, data, params)

        # save model before test

        model.save(input_path + "/model/" + model_name+"_"+version)

    else:
        print("===Testing====")
        data = input_processor(stock_names, input_path, is_train, test_start_date, test_end_date, train_start_date)
        if data['images'].shape[0] % len(stock_names) != 0:
            raise Exception("duplicate alert. pls check for sanity")

        from keras.models import load_model
        model = load_model(input_path + "/model/" + model_name+ "_"+ version)
        pred_df = test(model, data, params, update_model=False)

        #save predictions
        pred_path = input_path+"/"+"predictions"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        pred_df.to_csv(pred_path+"predictions_"+version+".csv")