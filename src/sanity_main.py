from src import sanity_testmodule as test_module
from src import sanity_prediction_module as prediction_module
import numpy as np
from tqdm import tqdm


train_filepath = "../sanity_input/train"
test_filepath = "../sanity_input/test"

stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']

tm = test_module.TestModule(filepath=test_filepath, stock_names=stock_names)

all_res = []
for i in tqdm(range(tm.data.shape[0]-2)):
    (predictions, images) = prediction_module.fast_update_data_and_predict(tm.get_current_day_data_dict())

    tm.update_current_day()
    (result, labels) = tm.evalute(predictions)
    prediction_module.update_model(images,labels)
    all_res.append(result)
    print("\nres: ",result," mean: ", np.mean(all_res))
