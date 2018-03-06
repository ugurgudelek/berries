import pandas as pd
import numpy as np
from talib import abstract

spy = pd.read_csv('../input/raw_input/spy.csv')

sma = abstract.SMA(spy['close'].values, timeperiod=25)

