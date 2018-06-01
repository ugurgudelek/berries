"""
Ugur Gudelek
29.05.2018
"""

import pandas as pd
import numpy as np


class BuySell:

    def __init__(self, capital):
        self.initial_capital = capital
        self.current_capital = capital
        self.share_amount = 0


    def buyandhold(self, dataframe):
        """

        Args:
            dataframe: (pd.DataFrame) should contains price and directive, date columns

        Returns: (float) profit

        """
        first_price = dataframe.iloc[0].loc['price']
        last_price  = dataframe.iloc[-1].loc['price']

        self.current_capital, self.share_amount = BuySell._buy(self.initial_capital, first_price)
        money, self.share_amount = BuySell._sell(self.share_amount, last_price)
        self.current_capital += money

        profit = self.current_capital - self.initial_capital

        return profit

    @staticmethod
    def _buy(money, price):
        """buys if appliable
        Returns: (float, int) current_money, share_amount"""

        remaining_money = money
        share_amount = 0

        if money > price:
            share_amount = int(money / price)
            remaining_money = money - share_amount * price

        return remaining_money, share_amount

    @staticmethod
    def _sell(share_amount, price):
        """sells all shares :)
        Returns: (float, int) current_money, share_amount"""
        money = share_amount * price
        return money, 0



    def process(self, dataframe):
        """

        Args:
            dataframe: (pd.DataFrame) should contains price and directive, date columns

        Returns: (float) profit

        """
        capital_list = list()
        share_amount_list = list()

        for idx,row in dataframe.iterrows():
            if row['directive'] == 'buy':
                remaining_money, share_amount = BuySell._buy(self.current_capital, row['price'])
                self.current_capital = remaining_money
                self.share_amount += share_amount

            if row['directive'] == 'sell':
                money, share_amount = BuySell._sell(self.share_amount, row['price'])
                self.current_capital += money
                self.share_amount = share_amount
            capital_list.append(self.current_capital)
            share_amount_list.append(self.share_amount)

        dataframe['current_capital'] = capital_list
        dataframe['share_amount'] = share_amount_list
        dataframe['total_capital'] = dataframe['current_capital'] + dataframe['share_amount']*dataframe['price']

        dataframe['profit'] = dataframe['total_capital'] - self.initial_capital


        return dataframe





if __name__ == "__main__":
    np.random.seed(42)
    initial_capital = 100000
    # size = 100
    # fake_dataframe = pd.DataFrame({'price':np.random.randint(100,150, size=size),
    #                                'directive':np.random.choice(['buy','hold','sell'], size=size, replace=True),
    #                                'name':np.random.choice(['xlp','xlu','xlv','xly','dia','ewa','ewc','ewg','ewh','ewj','eww','spy','xlb','xle','xlf','xli','xlk'], size=size, replace=True)})

    dataframe = pd.read_csv('../experiment/finance_cnn3/result.csv')
    prices = pd.read_csv('../dataset/finance/stocks/stocks.csv')

    dataframe = pd.merge(dataframe, prices, on=['date', 'name']).drop(['open','high','low','close', 'volume'], axis=1)

    dataframe = dataframe.rename(columns={'adjusted_close': 'price'})

    def custom_argmax(row):
        row = row[['psell','pbuy','phold']]

        argmax_idx = np.argmax(row.values)


        if argmax_idx == 0:
            return 'sell'
        if argmax_idx == 1:
            return 'buy'
        return 'hold'

    def custom_argmax2(row):
        row = row[['psell','pbuy','phold']]

        argmax_idx = np.argmax(row.values)


        if argmax_idx == 0:
            return pd.Series([1,0,0])
        if argmax_idx == 1:
            return pd.Series([0,1,0])
        return pd.Series([0,0,1])


    dataframe['directive'] = dataframe.apply(custom_argmax, axis=1)
    dataframe[['pmapsell', 'pmapbuy', 'pmaphold']] = dataframe.apply(custom_argmax2, axis=1)

    dataframe.loc[((dataframe['pmaphold'] == 1) & (dataframe['rhold'] == 1) |
                   (dataframe['pmaphold'] == 0) & (dataframe['rhold'] == 0))].shape[0] /     dataframe.shape[0]

    dataframe.loc[((dataframe['pmapsell'] == 1) & (dataframe['rsell'] == 1) |
                   (dataframe['pmapsell'] == 0) & (dataframe['rsell'] == 0))].shape[0] /     dataframe.shape[0]

    dataframe.loc[((dataframe['pmapbuy'] == 1) & (dataframe['rbuy'] == 1) |
                   (dataframe['pmapbuy'] == 0) & (dataframe['rbuy'] == 0))].shape[0] /     dataframe.shape[0]

    stock_names = dataframe['name'].unique()
    result_dict = dict()
    for stock_name in stock_names:

        search_df = dataframe.loc[dataframe['name'] == stock_name]

        buysell = BuySell(capital=initial_capital)

        # result_dict[stock_name] = buysell.buyandhold(dataframe=search_df)
        result_dict[stock_name] = buysell.process(dataframe=search_df)['profit'].iloc[-1]

    print(result_dict)
