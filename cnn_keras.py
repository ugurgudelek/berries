import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def train_cnn(data, labels, params):
    """Trains and evaluates CNN on the given train and test data, respectively."""

    data = data.as_matrix()
    labels = labels.as_matrix()

    data = data.reshape(data.shape[0], params["input_w"], params["input_h"], 1)

    recalls = []
    precisions = []
    fprs = []
    tprs = []
    accuracies = []
    losses = []

    # stratified cross validation
    skf = StratifiedKFold(n_splits=params["num_folds"], shuffle=True)

    # fold number
    foldNum = 1
    for train_index, test_index in skf.split(np.zeros(data.shape[0]), np.argmax(labels, 1)):
        # print fold number
        print("Fold", foldNum)
        print()

        # CNN model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(params["input_w"], params["input_h"], 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(params["num_classes"], activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        trainData = data[train_index, :]
        testData = data[test_index, :]
        trainLabels = labels[train_index, :]
        testLabels = labels[test_index, :]

        # fit the model to the training data
        print("Fitting model to the training data")
        print("")
        model.fit(trainData, trainLabels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
                  validation_data=None)

        # predict the class labels
        print("Predicting test data")
        print()
        cur_score = model.evaluate(testData, testLabels, verbose=0)
        y_pred = model.predict(testData)[:, 1]
        y_true = np.argmax(testLabels, 1)

        # calculate roc curve
        print("Calculating ROC and PR curves")
        print()
        cur_fpr, cur_tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        cur_precision, cur_recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
        fprs.append(cur_fpr.tolist())
        tprs.append(cur_tpr.tolist())
        precisions.append(cur_precision.tolist())
        recalls.append(cur_recall.tolist())
        losses.append(cur_score[0])
        accuracies.append(cur_score[1])

        # increase fold number
        foldNum += 1

    # plot the roc curves
    print("Plotting ROC curves")
    print()
    plt.figure()
    plt.title("ROC Curves for all Folds")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    foldNum = 1
    for i in range(0, len(fprs)):
        plt.plot(fprs[i], tprs[i], label=("Fold" + str(foldNum)))
        foldNum += 1
    plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), "k--", label="Baseline")
    plt.legend(loc=4)

    print("Plotting PR curves")
    print()
    plt.figure()
    plt.title("PR Curves for all Folds")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    foldNum = 1
    for i in range(0, len(recalls)):
        plt.plot(recalls[i], precisions[i], label=("Fold" + str(foldNum)))
        foldNum += 1
    pr_baseline = np.sum(np.argmax(labels, 1) == 1) / (
    np.sum(np.argmax(labels, 1) == 1) + np.sum(np.argmax(labels, 1) == 0))
    plt.plot(np.linspace(1, 0, 5), np.ones(5) * pr_baseline, "r--", label="Baseline")
    plt.legend(loc=4)

    print("Losses:")
    for i in losses:
        print(i)

    print()

    print("Accuracies:")
    for i in accuracies:
        print(i)

    print()

    print("Mean Loss:")
    print(np.mean(losses))

    print()

    print("Mean Accuracy:")
    print(np.mean(accuracies))

    plt.show()
