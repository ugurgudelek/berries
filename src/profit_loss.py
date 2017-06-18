import pandas as pd
from helper import quantize

def buy_and_hold(prices,capital=10000):
    buy_price = prices[0]
    sell_price = prices[-1]

    # buy goods and update capital
    goods_quantity = capital // buy_price
    capital = capital - goods_quantity * buy_price

    # sell goods and update capital
    capital = capital + goods_quantity * sell_price

    return capital


def buy_sell_wrt_signal(prices,buy_hold_sell_signal):
    pass


predictions = pd.read_csv("../result/predictions.csv", header=None,index_col=0).values
predictions = predictions.reshape(predictions.shape[0])
predictions_regr = pd.Series(predictions)
predictions_class = predictions.apply(quantize)


