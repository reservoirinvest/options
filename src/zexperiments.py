import pandas_market_calendars as mcal
import datetime
from utils import market_is_open
print(market_is_open('NSE'))
# cal = mcal.get_calendar('NYSE')
# print(cal.schedule(datetime.datetime.now(), datetime.datetime.now()))