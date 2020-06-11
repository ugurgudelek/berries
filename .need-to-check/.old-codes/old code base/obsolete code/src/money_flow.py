
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(style='ticks')
import loss_profit
from datetime import datetime


# In[10]:


# this is wrapper for transaction function
# model = 2-class regression
# commission = 5 USD
# our_strategy vs buy_and_hold

def make_transactions(stock_names):
    
    if type(stock_names) is not list:
        stock_names = [stock_names]
#     print("Preparing adjusted close dataframe...")
    prices = loss_profit.prepare_adj_close(stock_names)

#     print("Calculating final capital using prediction model...")
    capital, shares, record_transactions_df, record_shares_df = loss_profit.buy_sell_regr(
        stock_names = stock_names, 
        predictions_name = 'predictions_model_regr_100epoch_qratio_0_2017_07_16 20_25_16_566179', 
        adj_close = prices, 
        buy_thr=.0, 
        sell_thr=-.0, 
        transaction_cost=5,verbose=False)
    #capital, shares = loss_profit.buy_sell_class3(predictions_name = 'predictions_model_class_100epoch_2017_07_11 17_08_21_002137', adj_close = prices, transaction_cost=5)
    #capital, shares = loss_profit.buy_sell_class2(predictions_name = 'predictions_model_class_100epoch_2017_07_13 13_15_09_200788', adj_close = prices, transaction_cost=5)
   
#     print("Final captial:")
#     print(capital)
#     print("Final shares:")
#     print(shares)
    
    return record_transactions_df, record_shares_df

def buy_and_hold_transaction(stock_names):
    if type(stock_names) is not list:
        stock_names = [stock_names]
    prices = loss_profit.prepare_adj_close(stock_names)
    buy_hold_final_capital, buy_hold_final_shares = loss_profit.buy_hold(stock_names, prices,verbose=False)
    return buy_hold_final_capital, buy_hold_final_shares
    


# In[11]:


# create transaction dataframe
def create_transaction_dataframe(record_buy_sell_df,record_shares_df,stock_names):
    if type(stock_names) is not list:
        stock_names = [stock_names]
    return_df = pd.DataFrame()
    for stock_name in stock_names:
        buy_sell_df = record_buy_sell_df.loc[record_buy_sell_df.Names == stock_name]
        
        trans = pd.DataFrame()
        buy_df = buy_sell_df.loc[buy_sell_df.Operations == 'buy'].reset_index(drop=True)
        sell_df = buy_sell_df.loc[buy_sell_df.Operations == 'sell'].reset_index(drop=True)

        trans['sell_date'] = sell_df['Dates'].copy()
        trans['buy_date'] = buy_df['Dates'].copy()
        
        c_df = create_capital_dataframe(stock_names, record_shares_df, record_buy_sell_df)
        trans['capital_before'] = trans.buy_date.apply(lambda x: c_df.capital.loc[c_df.date == x].values[0] )
        trans['capital_after'] = trans.sell_date.apply(lambda x: c_df.capital.loc[c_df.date == x].values[0] )

        trans['return'] = (trans.capital_after - trans.capital_before)/trans.capital_before
        trans['t_period'] = trans.apply(lambda x: day_between(x.sell_date,x.buy_date).days, axis=1)
        trans['name'] = [stock_name for i in range(trans.shape[0])]
        return_df = pd.concat([return_df,trans])
    return_df = return_df.sort_values(['sell_date'],axis=0).reset_index(drop=True)
    return return_df


# In[12]:


def get_amount_by_name_and_date(record_shares_df, name, date):
    return record_shares_df[name].loc[record_shares_df.date == date].values[0]

def create_capital_dataframe(stock_names,record_shares_df,record_buy_sell_df):
    if type(stock_names) is not list:
        stock_names = [stock_names]
    prices_df = loss_profit.prepare_adj_close(stock_names)   
    capitals =[]
    record_shares_df.date = record_buy_sell_df.Dates
    for day in record_shares_df.date:
        capital = record_buy_sell_df.Capitals.loc[record_buy_sell_df.Dates == day].values[0]
        for stock_name in stock_names:
            stock_amount = get_amount_by_name_and_date(record_shares_df, stock_name, day)
            if stock_amount > 0:
                price = prices_df['Adj_Close'].loc[np.logical_and((prices_df.Date == day),(prices_df.Name == stock_name))].values[0]
                capital += price*stock_amount
        capitals.append(capital)
    return pd.DataFrame.from_dict({'capital':capitals,'date':record_shares_df.date.copy()})


# In[13]:


# create main money flow dataframe
# name : name of ETF
# our : final capital using our algorithm
# bah : final capital using buy and hold
# our_r: our annualized % return
# bah_r: bah annualized % return
# ant : annualized number of transaction
# pos : percent of success : sum(succ. transaction) / sum(transaction)
# apt : average percent profit per transactions
# l   : average transaction length
# mpt : maximum profit percentage in transaction
# mlt : maximum loss percentage in transaction
# maxc : maximum capital over test period
# minc : minimum capital over test period
money_flow_df = pd.DataFrame(columns=['name','our','bah','our_r','bah_r','ant','pos','apt','l','mpt','mlt','maxc','minc'])



# In[14]:


import math
def day_between(date1_str,date2_str):
    return datetime.strptime(date1_str, '%Y-%m-%d') - datetime.strptime(date2_str, '%Y-%m-%d')

# name : name of ETF
def get_name(stock_name):
    return stock_name

# our : final capital using our algorithm
def get_our(final_capital):
    return final_capital

# bah : final capital using buy and hold
def get_bah(buy_and_hold_capital):
    return buy_and_hold_capital

# our_r: our annualized % return average
def get_our_r(transactions_df, period_of_days):
#     P : initial capital
#     n   : period in year(for day its 365)
#     t   : num of period observed: 1 means 365 day
#     A : final capital
    
    A = transactions_df.capital_after.iloc[-1]
    P = transactions_df.capital_before.iloc[0]
    n = 365
    t = period_of_days / n # our test period / one year
    r = n*((A/P)**(1/(n*t))-1)
    
    return 100*r

# bah_r: bah annualized % return
def get_bah_r(buy_and_hold_capital, period_of_days):
    A = buy_and_hold_capital
    P = 10000
    n = 365
    t = period_of_days / n # our test period / one year
    r = n*((A/P)**(1/(n*t))-1)
    
    return 100*r

# ant : annualized number of transaction
def get_ant(transactions_df,period_of_days):
    return transactions_df.shape[0] * 365 / period_of_days

# pos : percent of success : sum(succ. transaction) / sum(transaction)
def get_pos(transactions_df):
    pos_len = transactions_df.loc[transactions_df['return'] > 0].shape[0]
    return 100*pos_len / transactions_df.shape[0]

# apt : average percent profit per transactions
def get_apt(transactions_df):
    return 100*transactions_df['return'].sum() / transactions_df.shape[0]

# l   : average transaction length
def get_l(transactions_df):
    return transactions_df.t_period.sum() / transactions_df.shape[0]

# mpt : maximum profit percentage in transaction
def get_mpt(transactions_df):
    return 100*transactions_df['return'].max()

# mlt : maximum loss percentage in transaction
def get_mlt(transactions_df):
    return transactions_df['return'].min()

# maxc : maximum capital over test period
def get_maxc(transactions_df):
    return transactions_df.capital_after.max()

# minc : minimum capital over test period
def get_minc(transactions_df):
    return transactions_df.capital_after.min()


# # Unit Test
# # name : name of ETF
# get_name('spy')
# # our : final capital using our algorithm
# get_our(t_df)
# # bah : final capital using buy and hold
# get_bah('spy')
# # our_r: our annualized % return
# get_our_r(t_df)
# # bah_r: bah annualized % return
# get_bah_r('spy',420)
# # ant : annualized number of transaction
# get_ant(t_df,420)
# # pos : percent of success : sum(succ. transaction) / sum(transaction)
# get_pos(t_df)
# # apt : average percent profit per transactions
# get_apt(t_df)
# # l   : average transaction length
# get_l(t_df)
# # mpt : maximum profit percentage in transaction
# get_mpt(t_df)
# # mlt : maximum loss percentage in transaction
# get_mlt(t_df)
# # maxc : maximum capital over test period
# get_maxc(t_df)
# # minc : minimum capital over test period
# get_minc(t_df)


# In[15]:


def create_money_flow_dataframe(stock_names, period_of_days = 420 * 365 / 250):
    col_order = ['name','our','bah','our_r','bah_r','ant','pos','apt','l','mpt','mlt','maxc','minc']

    money_flow_dict = {'name':[],
                       'our'  :[],
                       'bah'  :[],
                       'our_r':[],
                       'bah_r':[],
                       'ant'  :[],
                       'pos'  :[],
                       'apt'  :[],
                       'l'    :[],
                       'mpt'  :[],
                       'mlt'  :[],
                       'maxc' :[],
                       'minc' :[]                  
                      }
    for stock_name in stock_names:
           
        record_buy_sell_df, record_shares_df = make_transactions(stock_name)
        t_df = create_transaction_dataframe(record_buy_sell_df,record_shares_df, stock_name)
        final_capital = t_df.capital_after.iloc[-1]
        bah_capital = buy_and_hold_transaction(stock_name)[0]
        if type(stock_name) is list:
            print('ALL')
            money_flow_dict['name'].append(get_name('ALL'))
        else:
            print(stock_name+',', end=' ')    
            money_flow_dict['name'].append(get_name(stock_name))
            
        money_flow_dict['our'].append(get_our(final_capital))
        money_flow_dict['bah'].append(get_bah(bah_capital))
        money_flow_dict['our_r'].append(get_our_r(t_df,period_of_days))
        money_flow_dict['bah_r'].append(get_bah_r(bah_capital, period_of_days))
        money_flow_dict['ant'].append(get_ant(t_df, period_of_days))
        money_flow_dict['pos'].append(get_pos(t_df))
        money_flow_dict['apt'].append(get_apt(t_df))
        money_flow_dict['l'].append(get_l(t_df))
        money_flow_dict['mpt'].append(get_mpt(t_df))
        money_flow_dict['mlt'].append(get_mlt(t_df))
        money_flow_dict['maxc'].append(get_maxc(t_df))
        money_flow_dict['minc'].append(get_minc(t_df))

    return pd.DataFrame.from_dict(money_flow_dict)[col_order]

    


# In[16]:


stock_names = [['spy', 'xlf', 'xlu', 'xle','xlp', # this second list means take all shares and conduct our CTA
                'xli', 'xlv', 'xlk', 'ewj',
                'xlb', 'xly', 'eww', 'dia',
                'ewg', 'ewh', 'ewc', 'ewa'],
    
                    'spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa'
               
              ]
period_of_days = 420 * 365 / 250 #tfixed for working days test sample for each
# stock_names = ['spy']

m_flow_df = create_money_flow_dataframe(stock_names)
m_flow_df.ix[:,1:] = m_flow_df.ix[:,1:].applymap(lambda x: "{0:.2f}".format(x))
# m_flow_df.to_csv("../result/money_flow_data.csv",index=False)

#
# # In[ ]:
#
#
# m_flow_df
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# A = s_t_df.capital_after.iloc[-1]
# P = s_t_df.capital_before.iloc[0]
# n = 365
# t = 1.68
# r = n*((A/P)**(1/(n*t))-1)
# s_t_df





