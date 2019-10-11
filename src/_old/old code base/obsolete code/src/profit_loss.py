import pandas as pd
from helper import quantize
import  numpy as np
from preprocessing import get_last_saved_data
import matplotlib.pyplot as plt

class Profit_Loss:

    def __init__(self, capital):
        self.init_capital = capital
        self.capital = capital
        self.good_qty = 0

    def construct_buy_hold_sell_signal(self, stock_names, method, prediction):

        signal = pd.DataFrame()
        for stock_name in stock_names:
            signal[stock_name] = np.zeros(shape=prediction.shape)
            if method == 'buy_and_hold':
                signal[stock_name].iloc[0] = 1
                signal[stock_name].iloc[-1] = -1
            elif method == 'custom':
                for i,pred in enumerate(prediction):
                    if pred < -0.38:
                        signal[stock_name].iloc[i] = -1
                    elif pred > 0.38:
                        signal[stock_name].iloc[i] = 1

        return signal

    def transaction(self, prices, signal):

        capital_l = []
        good_qty_l = []

        for i,pulse in enumerate(signal.values):
            if pulse == 1: # buy
                new_good_qty = self.capital // prices[i]
                self.good_qty += new_good_qty
                self.capital = self.capital - new_good_qty * prices[i] # remaining capital
            elif pulse == -1:  # sell
                self.capital += self.good_qty * prices[i]
                self.good_qty = 0


            capital_l.append(self.capital)
            good_qty_l.append(self.good_qty)
        t = pd.DataFrame()
        t['capital'] = capital_l
        t['good_qty'] = good_qty_l
        profit = t['capital'].iloc[-1] + t['good_qty'].iloc[-1] * prices[-1] - self.init_capital
        return t, profit



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









p_l = Profit_Loss(capital=100000)
s = p_l.construct_buy_hold_sell_signal(['spy'], method="custom", prediction=predictions_test)
t,profit = p_l.transaction(prices_test, s)
plt.scatter(list(range(len(s))),s)
plt.show()
print(profit)

