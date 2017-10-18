from src import sanity_testmodule as test_module
from src import sanity_prediction_module as prediction_module
import numpy as np
from tqdm import tqdm


train_filepath = "../sanity_new/train"
test_filepath = "../sanity_new2/test/raw_data"

stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']

tm = test_module.TestModule(filepath=test_filepath, stock_names=stock_names)

all_res = []
for i in tqdm(range(tm.data.shape[0]//17-2)):
    print("{} day is starting...".format(tm.current_day))
    (predictions, images) = prediction_module.fast_update_data_and_predict(tm.get_current_day_data_dict())
    print("Predictions: {}".format(predictions))
    tm.update_current_day()
    (result, labels) = tm.evalute(predictions)
    print(("Labels: {}".format(labels)))
    prediction_module.update_model(images,labels)
    all_res.append(result)
    print("\nres: ",result," mean: ", np.mean(all_res))
