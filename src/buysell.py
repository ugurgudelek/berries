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
        last_price = dataframe.iloc[-1].loc['price']

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

        for idx, row in dataframe.iterrows():
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

        dataframe.loc[:, 'current_capital'] = capital_list
        dataframe.loc[:, 'share_amount'] = share_amount_list
        dataframe.loc[:, 'total_capital'] = dataframe['current_capital'] + dataframe['share_amount'] * dataframe[
            'price']

        dataframe.loc[:, 'profit'] = dataframe['total_capital'] - self.initial_capital

        return dataframe


def label_to_directives(row):
    row = row[['pbuy', 'psell', 'phold']]
    argmax_idx = np.argmax(row.values)

    if argmax_idx == 0:
        return 'buy'
    if argmax_idx == 1:
        return 'sell'
    return 'hold'

def buysell_pipeline_stock():
    initial_capital = 100000
    stock_names = ['dia', 'ewa', 'ewc', 'ewg', 'ewh', 'ewj', 'eww', 'spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlp',
                   'xlu', 'xlv', 'xly']
    exp_name = 'stock_exp'

    final_result_df = None
    for stock_name in stock_names:
        path = '../experiment/finance_cnn/{}/{}/prediction_results.csv'.format(exp_name, stock_name)
        dataframe = pd.read_csv(path)
        dataframe['directive'] = dataframe.apply(label_to_directives, axis=1)
        dataframe['price'] = dataframe['raw_adjusted_close'].values

        # Buy and Hold strategy
        buyandhold_profit = BuySell(capital=initial_capital).buyandhold(dataframe)

        # Our simple strategy
        buysell = BuySell(capital=initial_capital)
        buysell_result_df = buysell.process(dataframe)
        subset = ['name', 'date', 'price', 'directive', 'current_capital', 'share_amount', 'total_capital', 'profit']
        #     subset = buysell_result_df.columns
        if final_result_df is None:
            final_result_df = pd.DataFrame(columns=subset)

        last_row = buysell_result_df[subset].iloc[-1]
        last_row['bah_profit'] = buyandhold_profit

        final_result_df = final_result_df.append(last_row)

    final_result_df = final_result_df.reset_index(drop=True)

    return final_result_df


def buysell_pipeline_stress():
    initial_capital = 100000
    exp_name = 'stress_exp'
    final_result_df = None
    for i in range(19):
        path = '../experiment/finance_cnn/{}/{}/prediction_results.csv'.format(exp_name, i)
        dataframe = pd.read_csv(path)
        dataframe['directive'] = dataframe.apply(label_to_directives, axis=1)
        dataframe['price'] = dataframe['raw_adjusted_close'].values

        # Buy and Hold strategy
        buyandhold_profit = BuySell(capital=initial_capital).buyandhold(dataframe)

        # Our simple strategy
        buysell = BuySell(capital=initial_capital)
        buysell_result_df = buysell.process(dataframe)
        subset = ['name', 'date', 'price', 'directive', 'current_capital', 'share_amount', 'total_capital', 'profit']
        #     subset = buysell_result_df.columns
        if final_result_df is None:
            final_result_df = pd.DataFrame(columns=subset)

        last_row = buysell_result_df[subset].iloc[-1]
        last_row['bah_profit'] = buyandhold_profit
        last_row['deleted_colnum'] = i

        final_result_df = final_result_df.append(last_row)

    final_result_df = final_result_df.reset_index(drop=True)

    final_result_df


