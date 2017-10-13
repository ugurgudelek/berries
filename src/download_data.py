from src import google_finance_io
import datetime
stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']


google_finance_io.download_data(stock_names, start_date=datetime.date(2005, 1,1),
                                        end_date=datetime.date(2006, 12, 31), verbose=True, path="../sanity_input/test")