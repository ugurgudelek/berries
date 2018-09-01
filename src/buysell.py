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

    def process(self, dataframe, only_valid_transactions=True):
        """
        Args:
            dataframe: (pd.DataFrame) should contains price and directive, date columns
        Returns: (float) profit
        """
        dataframe['capital_before'] = np.nan
        dataframe['share_before'] = np.nan
        dataframe['capital_after'] = np.nan
        dataframe['share_after'] = np.nan

        transaction = Transaction()
        for idx, row in dataframe.iterrows():
            capital_before = self.current_capital
            share_before = self.share_amount
            transaction_done = None

            if transaction.state == TransactionState.ENDED:
                transaction = Transaction()

            if row['directive'] == 'buy' and transaction.state == TransactionState.NA:
                remaining_money, share_amount, transaction_done = BuySell._buy(self.current_capital, row['price'])
                self.current_capital = remaining_money
                self.share_amount += share_amount
                if transaction_done:
                    transaction.start(price=capital_before, date=row['date'])

            if row['directive'] == 'sell' and transaction.state == TransactionState.ONGOING:
                money, share_amount, transaction_done = BuySell._sell(self.share_amount, row['price'])
                self.current_capital += money
                self.share_amount = share_amount
                if transaction_done:
                    transaction.end(price=self.current_capital, date=row['date'])

            if transaction_done is not None:  # save transaction
                if (not only_valid_transactions) or transaction_done:
                    dataframe.loc[idx, 'capital_before'] = capital_before
                    dataframe.loc[idx, 'share_before'] = share_before
                    dataframe.loc[idx, 'capital_after'] = self.current_capital
                    dataframe.loc[idx, 'share_after'] = self.share_amount
                    if transaction.state != TransactionState.NA:
                        dataframe.loc[idx, 'transaction_state'] = transaction.state
                    if transaction.state == TransactionState.ENDED:
                        dataframe.loc[idx, 'transaction_profit'] = transaction.get_profit()
                        dataframe.loc[idx, 'transaction_profit_percentage'] = transaction.get_norm_profit() * 100
                        dataframe.loc[idx, 'transaction_period'] = transaction.get_period()

        dataframe.loc[:, 'profit'] = dataframe['capital_after'] - dataframe['capital_before']
        dataframe.loc[:, 'total_capital'] = dataframe['capital_after'] + dataframe['share_after'] * dataframe['price']
        dataframe.loc[:, 'total_profit'] = dataframe['total_capital'] - self.initial_capital
        dataframe = dataframe.dropna(axis=0, subset=['capital_before'])  # this keeps only transaction rows i.e buy-sell

        dataframe['date'] = dataframe['date'].astype('datetime64[ns]')
        # dataframe.loc[:, 'till_last_transaction'] = dataframe['date'].diff(periods=1)

        buy_df = dataframe.loc[dataframe['directive'] == 'buy'].reset_index(drop=True)
        sell_df = dataframe.loc[dataframe['directive'] == 'sell'].reset_index(drop=True)

        transactions = pd.DataFrame()
        for (i_b, row_buy), (i_s, row_sell) in zip(buy_df.iterrows(), sell_df.iterrows()):
            row = pd.Series()
            row['name'] = row_buy['name']

            row['start_date'] = row_buy['date'].date()
            row['end_date'] = row_sell['date'].date()
            row['period'] = row_sell['transaction_period'].days

            row['start_price'] = row_buy['price']
            row['end_price'] = row_sell['price']
            row['capital_before'] = row_buy['capital_before']
            row['capital_after'] = row_sell['capital_after']
            row['profit'] = row_sell['transaction_profit']
            row['profit_perc'] = row_sell['transaction_profit_percentage']

            transactions = transactions.append(row, ignore_index=True)

        return dataframe, transactions

    @staticmethod
    def generate_table_row(transactions, init_capital, bah_capital, period_of_days):

        """
        Proposed Strategy Annualized Return (OURr),
        Total Capital with Buy and Hold Strategy (BaH),
        Buy and Hold Annualized Return (BaHr),
        Annualized Number of Transaction (AnT)
        Percent of Success (PoS)
        Average Profit Per Transactions (ApT),
        Average Transaction Length (L),
        Maximum Profit in Transaction (MpT),
        Maximum Loss in Transaction (MlT),
        Maximum Capital (MaxC),
        Minimum Capital (MinC),
        Idle Ratio (IdleR)
        Sharpe Ratio (SharpeR)
        """

        def annualized_return(initial_capital, final_capital, period_of_days):
            #     P : initial capital
            #     n   : period in year(for day its 365)
            #     t   : num of period observed: 1 means 365 day
            #     A : final capital
            A = final_capital
            P = initial_capital
            n = 365
            t = period_of_days / n  # our test period / one year

            return 100 * n * ((A / P) ** (1 / (n * t)) - 1)

        # Our Capital
        def our(*args, **kwargs):
            transactions = kwargs['transactions']
            return transactions.iloc[-1]['capital_after']

        # Proposed Strategy Annualized Return (OURr)
        def ourr(*args, **kwargs):
            transactions = kwargs['transactions']
            period_of_days = kwargs['period_of_days']
            return annualized_return(initial_capital=transactions['capital_before'].iloc[0],
                                     final_capital=transactions['capital_after'].iloc[-1],
                                     period_of_days=period_of_days)

        # Buy and Hold Annualized Return (BaHr)
        def bahr(*args, **kwargs):
            init_capital = kwargs['init_capital']
            bah_capital = kwargs['bah_capital']
            period_of_days = kwargs['period_of_days']
            return annualized_return(initial_capital=init_capital,
                                     final_capital=bah_capital,
                                     period_of_days=period_of_days)

        # Annualized Number of Transaction (AnT)
        def ant(*args, **kwargs):
            transactions = kwargs['transactions']
            period_of_days = kwargs['period_of_days']
            transaction_count = transactions.shape[0]
            num_year = period_of_days / 365
            return transaction_count / num_year

        # Percent of Success (PoS)
        def pos(*args, **kwargs):
            transactions = kwargs['transactions']
            successful_transaction_count = np.sum(transactions['profit'] > 0)
            transaction_count = transactions.shape[0]
            return 100 * successful_transaction_count / transaction_count

        # Average Profit Per Transactions (ApT)
        def apt(*args, **kwargs):
            transactions = kwargs['transactions']
            transaction_count = transactions.shape[0]
            transactions_profit_sum = transactions['profit'].sum()
            return transactions_profit_sum / transaction_count

        # Average Transaction Length (L)
        def l(*args, **kwargs):
            transactions = kwargs['transactions']
            transactions_period_sum = transactions['period'].sum()
            transaction_count = transactions.shape[0]
            return transactions_period_sum / transaction_count

        # Maximum Profit in Transaction (MpT)
        def mpt(*args, **kwargs):
            transactions = kwargs['transactions']
            return transactions['profit'].max()

        # Maximum Loss in Transaction (MlT)
        def mlt(*args, **kwargs):
            transactions = kwargs['transactions']
            m = transactions['profit'].min()
            return m if m > 0 else 0

        # Maximum Capital (MaxC)
        def maxc(*args, **kwargs):
            transactions = kwargs['transactions']
            return transactions['capital_after'].max()

        # Minimum Capital (MinC)
        def minc(*args, **kwargs):
            transactions = kwargs['transactions']
            return transactions['capital_after'].min()

        # Idle Ratio (IdleR)
        def idler(*args, **kwargs):
            transactions = kwargs['transactions']
            period_of_days = kwargs['period_of_days']
            transactions_period_sum = transactions['period'].sum()
            return (period_of_days - transactions_period_sum) / period_of_days

        # Sharpe Ratio (SharpeR)
        # todo: implement later

        table_functions = {'OUR': our, 'OURr': ourr, 'BaHr': bahr, 'AnT': ant, 'PoS': pos, 'ApT': apt, 'L': l, 'MpT': mpt,
                           'MlT': mlt, 'MaxC': maxc, 'MinC': minc, 'IdleR': idler}

        row = pd.Series()
        for key, func in table_functions.items():
            row['Stocks'] = transactions.loc[0, 'name']
            row[key] = func(transactions=transactions, init_capital=init_capital,
                            bah_capital=bah_capital, period_of_days=period_of_days)

        return row


class TransactionState(Enum):
    NA = 0
    ONGOING = 1
    ENDED = 2


class Transaction:

    def __init__(self):
        self.state = TransactionState.NA

    def start(self, price, date):
        self.state = TransactionState.ONGOING
        self.init_price = price
        self.init_date = date

    def end(self, price, date):
        self.state = TransactionState.ENDED
        self.end_price = price
        self.end_date = date

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

    def get_period(self):
        if self._is_available():
            return self.end_date - self.init_date


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
        dataframe['date'] = dataframe['date'].astype('datetime64[ns]')
        dataframe['directive'] = dataframe.apply(label_to_directives, axis=1)
        dataframe['directive'].iloc[-1] = 'sell'  # to sell them all at the end
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


def table_experiment():
    initial_capital = 100000
    stock_names = ['dia', 'ewa', 'ewc', 'ewg', 'ewh', 'ewj', 'eww', 'spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly']
    exp_name = 'stock_exp_7day'

    final_result_df = pd.DataFrame()
    for stock_name in stock_names:
        path = '../experiment/finance_cnn/{}/{}/prediction_results.csv'.format(exp_name, stock_name)
        dataframe = pd.read_csv(path)
        dataframe['date'] = dataframe['date'].astype('datetime64[D]')
        dataframe['directive'] = dataframe.apply(label_to_directives, axis=1)
        dataframe['directive'].iloc[-1] = 'sell'  # to sell them all at the end
        dataframe['price'] = dataframe['raw_adjusted_close'].values

        # Buy and Hold strategy
        bah_profit = BuySell(capital=initial_capital).buyandhold(dataframe)

        # Our simple strategy
        buysell_df, transactions = BuySell(capital=initial_capital).process(dataframe)
        result_row = BuySell.generate_table_row(transactions=transactions,
                                                init_capital=initial_capital,
                                                bah_capital=initial_capital + bah_profit,
                                                period_of_days=(dataframe['date'].iloc[-1] - dataframe['date'].iloc[
                                                    0]).days)

        final_result_df = final_result_df.append(result_row, ignore_index=True)


    order = ['Stocks', 'OUR', 'OURr', 'BaHr', 'AnT', 'PoS', 'ApT', 'L', 'MpT', 'MlT', 'MaxC', 'MinC', 'IdleR']
    final_result_df = final_result_df[order]

    final_result_df.applymap(lambda x:'{0:.2f}'.format(x) if type(x) != str else x).to_csv('transactions.csv', index=False)
    print()
table_experiment()