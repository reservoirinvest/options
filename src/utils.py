import asyncio
import datetime
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd
import pytz
import yaml
from from_root import from_root
from ib_insync import IB
from loguru import logger
from tqdm.asyncio import tqdm

ROOT = from_root()

@dataclass
class Timediff:
    """Stores time difference for file_age"""
    td: datetime.timedelta
    days: int
    hours: int
    minutes: int
    seconds: float

class Vars:
    """Variables from var.yml"""
    def __init__(self, MARKET: str) -> None:

        MARKET = MARKET.upper()
        self.MARKET = MARKET

        with open(ROOT / 'config' / 'var.yml', "rb") as f:
            data = yaml.safe_load(f)

        for k, v in data["COMMON"].items():
            setattr(self, k.upper(), v)

        for k, v in data[MARKET].items():
            setattr(self, k.upper(), v)

def split_time_difference(diff: datetime.timedelta):
    """Splits time diference into d, h, m, s"""
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += diff.microseconds / 1e6

    output = Timediff(*(diff, days, hours, minutes, seconds))

    return output

def get_closest_values(myArr: list, 
                       myNumber: float, 
                       how_many: int=0) -> list:
    
    """Get closest values in a list
    
    how_many: 0 gives the closest value\n
              1 | 2 | ... gives closest greater values (Call)\n
             -1 | -2 | ... gives closest lesser values (Put)"""

    i = 0

    result = []

    while i <= abs(how_many):

        if how_many > 0:
            # going right
            val = myArr[myArr > myNumber].min()
            
        elif how_many < 0:
            # going left
            val = myArr[myArr < myNumber].max()
            
        else:
            val = min(myArr.tolist(), key=lambda x: abs(x-myNumber))

        result.append(val)
        myNumber = val
        i +=1

    output = list(result[0:abs(1 if how_many == 0 else abs(how_many))])

    return output


def get_file_age(file_path: Path) -> Timediff:
    """Gets age of a file in timedelta and d,h,m,s"""

    time_now = datetime.datetime.now()

    try:

        file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    except FileNotFoundError as e:

        logger.info(f"{str(file_path)} file is not found")
        
        file_age = None

    else:

        # convert time difference to days, hours, minutes, secs
        td = (time_now - file_time)

        file_age = split_time_difference(td)

    return file_age


def get_dte(dt: Union[datetime.datetime, datetime.date, str], 
            exchange: str, # 'NSE' or 'SNP'
            time_stamp: bool=False) -> float:
    """Get accurate dte.
    Args: dt as datetime.datetime | datetime.date\n 
          exchange as 'nse'|'snp'\n
          time_stamp boolean gives market close in local timestamp\n
    Rets: dte as float | timestamp in local timzeone"""

    if type(dt) is str:
        dt = pd.to_datetime(dt)

    tz_dict = {'nse': ('Asia/Kolkata', 18), 'snp': ('EST', 16)}
    tz, hr = tz_dict[exchange.lower()]

    mkt_tz = pytz.timezone(tz)
    mkt_close_time = datetime.time(hr, 0)

    now = datetime.datetime.now(tz=mkt_tz)

    mkt_close = datetime.datetime.combine(dt.date(), mkt_close_time).astimezone(mkt_tz)

    dte = (mkt_close-now).total_seconds()/24/3600

    if time_stamp:
        dte = mkt_close

    return dte


def get_a_stdev(iv: float, price: float, dte: float) -> float:

    """Gives 1 Standard Deviation value.\n
    Assumes iv as annual implied volatility"""

    return iv*price*math.sqrt(dte/365)


async def qualify_unds(contracts: list, port: int=1300):
    """Qualify underlying contracts asynchronously"""

    with await IB().connectAsync(port=port) as ib:

        tasks = [ib.qualifyContractsAsync(c) for c in contracts]

        results = [await task_ 
                   for task_ 
                   in tqdm.as_completed(tasks, total=len(tasks), desc='Qualifying Unds')]

        return results


def make_dict_of_underlyings(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

    contracts_dict = {c[0].symbol: c[0] for c in qualified_contracts if c}

    return contracts_dict


def pickle_me(obj, file_name_with_path: Path):
    """Pickles objects in a given path"""
    
    with open(str(file_name_with_path), 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_with_age_check(obj: dict, 
                file_name_with_path: Path, 
                minimum_age_in_days: int=1):
    """Pickles an object after checking file age"""

    existing_file_age = get_file_age(file_name_with_path)

    if existing_file_age is None: # No file exists
        to_pickle = True
    elif existing_file_age.days > minimum_age_in_days:
        to_pickle = True
    else:
        to_pickle = False

    if to_pickle:
        pickle_me(obj, file_name_with_path)
        logger.info(f"Pickled underlying contracts to {file_name_with_path}")
    else:
        logger.info(f"Not pickled as existing file age {existing_file_age.days} is < {minimum_age_in_days}")


def get_pickle(path: Path):
    """Gets pickled object"""

    output = None # initialize

    try:
        with open(path, 'rb') as f:
            output = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"file not found: {path}")
    
    return output


async def get_ohlc_bars(ib: IB,
               c,
               WHAT_TO_SHOW: str='TRADES',
               DURATION: int=365,
               BAR_SIZE = "1 day",
               ) -> list:
    """Get Historical OHLC bars from IB"""

    DUR = str(DURATION) + " D"
    
    ohlc_bars = await ib.reqHistoricalDataAsync(
        contract=c,
        endDateTime=datetime.datetime.now(),
        durationStr=DUR,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=True,
        formatDate=2,  # UTC format
    )

    return ohlc_bars

async def get_option_chains(ib: IB, c):

    """Get Option Chains from IB"""

    chain = await ib.reqSecDefOptParamsAsync(
    underlyingSymbol=c.symbol,
    futFopExchange="",
    underlyingSecType=c.secType,
    underlyingConId=c.conId,
    )

    chain = chain[-1] if isinstance(chain, list) else chain

    return chain

async def get_market_data(ib: IB, 
                          c,
                          sleep:float = 1):

    """
    Get marketPrice including implied volatility\n   
    Pretty quick when market is closed
    """
    tick = ib.reqMktData(c, genericTickList="106")
    await asyncio.sleep(sleep)
    ib.cancelMktData(c)

    return tick


async def get_tick_data(ib: IB, 
                          c,):

    """
    Gets tick-by-tick data\n  
    Quick when market is open \n   
    Takes ~6 secs after market hours. \n  
    No impliedVolatility"""

    ticker = await ib.reqTickersAsync(c)
    ticker = ticker[-1] if isinstance(ticker, list) else ticker

    return ticker


if __name__ == "__main__":
    print(Vars('NSE'))




