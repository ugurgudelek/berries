import metric
from financeIO import LocalIO
import datetime
import pandas as pd
import label
import image
import cnn
import engine

# done 1: download 1-day data
# todo: semi-done 2: calculate metrics if available
# todo 3: maybe cluster features
# done 4: calculate labels
# done 5: create images if available
# todo 6: train & validation
# todo 7: test
# todo 8: plot
# todo 9: make it parallel


# done: fix metric engine. it should store wrt stock_names....
# todo: make all stationary processes in financeIO.py

DATE_FORMAT = '%d-%m-%Y'
START_DATE = datetime.datetime.strptime('01-01-2000', DATE_FORMAT).date()
END_DATE = datetime.datetime.strptime('31-12-2014', DATE_FORMAT).date()
STOCK_NAMES = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']
# STOCK_NAMES = ['spy']
MAKE_STATIONARY = True
NORMALIZE_IMAGE = True
APPLY_TANH = True

financeIO = LocalIO()
metric_engine = metric.MetricEngine(stock_names=STOCK_NAMES)
metric_engine.add_default_metrics()
label_engine = label.LabelEngine(financeIO=financeIO, make_stationary=MAKE_STATIONARY, apply_tanh=APPLY_TANH)
image_engine = image.ImageEngine(stock_names=STOCK_NAMES, split_period=28, normalize=NORMALIZE_IMAGE)
params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 10, "epochs": 100}
cnn_engine = cnn.CNNEngine(params=params, model_save_path='../model')



main_engine = engine.Engine(financeIO=financeIO,
                            metric_engine=metric_engine,
                            label_engine=label_engine,
                            image_engine=image_engine,
                            cnn_engine=cnn_engine,
                            stock_names=STOCK_NAMES,
                            make_stationary=MAKE_STATIONARY,
                            apply_tanh=APPLY_TANH,
                            verbose=True)

main_engine.run(start_date=START_DATE, end_date=END_DATE)

print()
