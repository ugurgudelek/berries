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

# todo:IMPORTANT!!!!
# todo: fix metric engine. it should store wrt stock_names....

DATE_FORMAT = '%d-%m-%Y'
start_date = datetime.datetime.strptime('01-09-2015', DATE_FORMAT)
end_date = datetime.datetime.strptime('01-12-2016', DATE_FORMAT)

financeIO = LocalIO()
metric_engine = metric.MetricEngine()
metric_engine.add_default_metrics()
label_engine = label.LabelEngine(financeIO=financeIO, make_stationary=True)
image_engine = image.ImageEngine(split_period=28, normalize=True)
params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 10, "epochs": 100}
cnn_engine = cnn.CNNEngine(params=params)

stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']

main_engine = engine.Engine(financeIO=financeIO,
                            metric_engine=metric_engine,
                            label_engine=label_engine,
                            image_engine=image_engine,
                            cnn_engine=cnn_engine,
                            stock_names=stock_names,
                            make_stationary=True,
                            verbose=True)

main_engine.run(start_date=start_date, end_date=end_date)

print()
