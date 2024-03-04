import pandas_market_calendars as mcal
import datetime

def market_is_open(market: str, date: datetime.datetime=None) -> bool:
    """SOMETIMES DOESN'T WORK!!!
    ---
    
    True if market is open.
    market: 'SNP' (converted to NYSE) | 'NSE'.
    date: <optional>. Takes datetime.now() if no date is given. 
    """

    if not date:
        date = datetime.datetime.now()

    market = market.upper()
    if market == 'SNP':
        market = 'NYSE'

    cal = mcal.get_calendar(market)
    my_cal = cal.schedule(start_date=date, end_date=date)

    start = datetime.datetime.fromtimestamp(my_cal.market_open.iloc[0].timestamp())
    end = datetime.datetime.fromtimestamp(my_cal.market_close.iloc[0].timestamp())

    result = start <= date <= end

    return result


print(market_is_open('NSE'))
# cal = mcal.get_calendar('NYSE')
# print(cal.schedule(datetime.datetime.now(), datetime.datetime.now()))