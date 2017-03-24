import quandl

quandl.ApiConfig.api_key = "DKSRWntQEian-RpuyGHu"
aapl = quandl.get_table('WIKI/PRICES', ticker = 'AAPL')
msft = quandl.get_table('WIKI/PRICES', ticker= 'MSFT')

print(aapl.head()['date'])
print(aapl.tail()['date'])

print(msft.head()['date'])
print(msft.tail()['date'])