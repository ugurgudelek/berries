'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def next_batch(data, lower, upper):
    data_size = data.shape[0]
    lower %= data_size
    upper %= data_size

    if lower > upper:
        return np.vstack((data[lower:], data[:upper]))

    return data[lower:upper]


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout, image_shape=(28, 28)):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_shape[0], image_shape[1], 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def launch_cnn(train_images, train_labels, test_images, test_labels,image_shape =(28,28),
               parameters={'learning_rate': 0.01, 'training_iters': 30000, 'batch_size': 128, 'dropout': 0.9}):


    # Network Parameters
    n_input = image_shape[0]*image_shape[1]  # data input (img shape: 28*28)
    n_classes = test_labels.shape[1]  # total classes (less,more)

    # Parameters
    learning_rate = parameters['learning_rate']
    training_iters = parameters['training_iters']
    batch_size = parameters['batch_size']
    display_step = 10
    dropout = parameters['dropout']  # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
        # 5x5 conv, 64 inputs, 128 outputs
        'wc3': tf.Variable(tf.random_normal([4, 4, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([image_shape[0] // 8 * image_shape[0] // 8 * 128, 3072])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([3072, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([3072])),
        'out': tf.Variable(tf.random_normal([n_classes]))
        
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob, image_shape=image_shape)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            lower = ((step - 1) * batch_size)
            upper = (step * batch_size)

            batch_x = next_batch(train_images.values, lower, upper)
            batch_y = next_batch(train_labels.values, lower, upper)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc), end='')
                print("\t lower {}, upper {}".format(lower, upper))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: test_images.values,
                                            y: test_labels.values,
                                            keep_prob: 1.}))

        # calculate precision-recall and roc
        prob = tf.nn.softmax(pred)
        y_pred = sess.run(prob,feed_dict={x:test_images.values,keep_prob:1.})[:,1]
        y_true = np.argmax(test_labels.values, 1)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
        print(precision)
        print(recall)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # plot precision-recall and roc
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc = "lower right")
        plt.show()

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
