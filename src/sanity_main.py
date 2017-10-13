from src import sanity_testmodule as test_module
from src import sanity_prediction_module as prediction_module



train_filepath = "../sanity_input/train"
test_filepath = "../sanity_input/test"

stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']

tm = test_module.TestModule(filepath=test_filepath, stock_names=stock_names)

prediction_module.update_data_and_predict(tm.get_current_day_data_dict())
