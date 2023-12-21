from yahooquery import Ticker

stocklist =['DIS','GM','HD','BABA','AAPL','APPS','PLTR','EXPR','MARA','BABA','SPCE','GME','RIOT','BB','RKT','HD','NIO']

t = Ticker(stocklist, asynchronous=True)

df = t.option_chain

# df.columns
# Index(['contractSymbol', 'strike', 'currency', 'lastPrice', 'change',
#        'percentChange', 'volume', 'openInterest', 'bid', 'ask', 'contractSize',
#        'lastTradeDate', 'impliedVolatility', 'inTheMoney'],
#       dtype='object')

# df.index.unique(level=0)
# Index(['AAPL', 'APPS', 'BABA', 'BB', 'DIS', 'EXPR', 'GM', 'GME', 'HD', 'MARA',
#        'NIO', 'PLTR', 'RIOT', 'RKT', 'SPCE'],
#       dtype='object', name='symbol')

# df.shape
# (22360, 14)

print(df)