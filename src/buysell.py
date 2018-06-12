import pandas as pd
import numpy as np
from enum import Enum

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

        self.current_capital, self.share_amount, transaction_done = BuySell._buy(self.initial_capital, first_price)
        money, self.share_amount, transaction_done = BuySell._sell(self.share_amount, last_price)
        self.current_capital += money

        profit = self.current_capital - self.initial_capital

        return profit

    @staticmethod
    def _buy(money, price):
        """buys if appliable
        Returns: (float, int) current_money, share_amount"""

        transaction_done = False
        remaining_money = money
        share_amount = 0

        if money > price:
            share_amount = int(money / price)
            remaining_money = money - share_amount * price
            transaction_done = True

        return remaining_money, share_amount, transaction_done

    @staticmethod
    def _sell(share_amount, price):
        """sells all shares :)
        Returns: (float, int) current_money, share_amount"""
        transaction_done = False
        money = share_amount * price
        if share_amount != 0:
            transaction_done = True
        return money, 0, transaction_done


    class TransactionState(Enum):
        ONGOING = 0
        ENDED = 1

    class Transaction:

        def __init__(self):
            pass

        def start(self, price):
            self.state = TransactionState.ONGOING
            self.init_price = price

        def end(self, price):
            self.state = TransactionState.ENDED
            self.end_price = price

        def _is_available(self):
            return self.state == TransactionState.ENDED
        def get_profit(self):
            if self._is_available():
                return self.end_price - self.init_price
            raise Exception('get_profit not available')

        def get_norm_profit(self):
            if self._is_available():
                return self.get_profit() / self.init_price
            raise Exception('get_norm_profit not available')



    def process(self, dataframe, only_valid_transactions=False):
        """

        Args:
            dataframe: (pd.DataFrame) should contains price and directive, date columns

        Returns: (float) profit

        """
        dataframe['capital_before'] = np.nan
        dataframe['share_before'] = np.nan
        dataframe['capital_after'] = np.nan
        dataframe['share_after'] = np.nan

        for idx, row in dataframe.iterrows():
            capital_before = self.current_capital
            share_before = self.share_amount
            transaction_done = None

            if row['directive'] == 'buy':
                remaining_money, share_amount, transaction_done = BuySell._buy(self.current_capital, row['price'])
                self.current_capital = remaining_money
                self.share_amount += share_amount

            if row['directive'] == 'sell':
                money, share_amount, transaction_done = BuySell._sell(self.share_amount, row['price'])
                self.current_capital += money
                self.share_amount = share_amount

            if transaction_done is not None:  # save transaction
                if (not only_valid_transactions) or transaction_done:
                    dataframe.loc[idx, 'capital_before'] = capital_before
                    dataframe.loc[idx, 'share_before'] = share_before
                    dataframe.loc[idx, 'capital_after'] = self.current_capital
                    dataframe.loc[idx, 'share_after'] = self.share_amount


        dataframe.loc[:, 'profit'] = dataframe['capital_after'] - dataframe['capital_before']
        dataframe.loc[:, 'total_capital'] = dataframe['capital_after'] + dataframe['share_after'] * dataframe['price']
        dataframe.loc[:, 'total_profit'] = dataframe['total_capital'] - self.initial_capital
        dataframe = dataframe.dropna(axis=0)  # this keeps only transaction rows i.e buy-sell


        dataframe['date'] = dataframe['date'].astype('datetime64[ns]')
        dataframe.loc[:, 'till_last_transaction'] = dataframe['date'].diff(periods=1)

        return dataframe


    def table(self):

        def annualized_return(initial_capital, final_capital, period_of_days):
            #     P : initial capital
            #     n   : period in year(for day its 365)
            #     t   : num of period observed: 1 means 365 day
            #     A : final capital
            A = final_capital
            P = initial_capital
            n = 365
            t = period_of_days / n # our test period / one year

            return 100 * n * ((A / P) ** (1 / (n * t)) - 1)


        # our_r: our annualized % return average
        def get_our_annualized_return(transactions_df, period_of_days):
            return annualized_return(initial_capital=transactions_df['total_capital'].iloc[0],
                                     final_capital=transactions_df['total_capital'].iloc[-1],
                                     period_of_days=period_of_days)

        # bah_r: bah annualized % return
        def get_bah_annualized_return(init_capital, bah_capital, period_of_days):
            return annualized_return(initial_capital=init_capital,
                                     final_capital=bah_capital,
                                     period_of_days=period_of_days)

        # ant : annualized number of transaction
        def get_annualized_number_of_transaction(transactions_df, period_of_days):
            return transactions_df.shape[0] * 365 / period_of_days

        # pos : percent of success : sum(succ. transaction) / sum(transaction)
        def get_percent_of_success(transactions_df):
            pos_len = transactions_df.loc[transactions_df['return'] > 0].shape[0]
            return 100 * pos_len / transactions_df.shape[0]

        # apt : average percent profit per transactions
        def get_apt(transactions_df):
            return 100 * transactions_df['return'].sum() / transactions_df.shape[0]

        # l   : average transaction length
        def get_l(transactions_df):
            return transactions_df.t_period.sum() / transactions_df.shape[0]

        # mpt : maximum profit percentage in transaction
        def get_mpt(transactions_df):
            return 100 * transactions_df['return'].max()

        # mlt : maximum loss percentage in transaction
        def get_mlt(transactions_df):
            return transactions_df['return'].min()

        # maxc : maximum capital over test period
        def get_maxc(transactions_df):
            return transactions_df.capital_after.max()

        # minc : minimum capital over test period
        def get_minc(transactions_df):
            return transactions_df.capital_after.min()

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
        subset = ['name', 'date', 'price', 'directive', 'capital_after', 'share_after', 'total_capital', 'total_profit']
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


buysell_pipeline_stock()
