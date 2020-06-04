# -*- coding: utf-8 -*-
# @Time   : 5/10/2020 11:32 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : neptune-ai-test.py

import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined.

neptune.init('ugurgudelek/sandbox')
# neptune.create_experiment(name='minimal_example')
#
# # log some metrics
#
# for i in range(100):
#     neptune.log_metric('loss', 0.95**i)
#
# neptune.log_metric('AUC', 0.96)

neptune.log_metric('loss', 0.95**i)


# Define parameters

PARAMS = {'decay_factor' : 0.5,
          'n_iterations' : 117}

# Create experiment with defined parameters

neptune.create_experiment(name='example_with_parameters',
                          params=PARAMS)

# Log image data

import numpy as np

array = np.random.rand(10, 10, 3)*255
array = np.repeat(array, 30, 0)
array = np.repeat(array, 30, 1)
neptune.log_image('mosaics', array)

# Log text data

neptune.log_text('top questions', 'what is machine learning?')


# # log some file
#
# # replace this file with your own file from local machine
# neptune.log_artifact('model_weights.pkl')
#
# # log file to some specific directory (see second parameter below)
#
# # replace this file with your own file from local machine
# neptune.log_artifact('model_checkpoints/checkpoint_3.pkl', 'training/model_checkpoints/checkpoint_3.pkl')
#
# # Upload source code
#
# # replace these two source files with your own files.
# neptune.create_experiment(upload_source_files=['main.py', 'model.py'])
