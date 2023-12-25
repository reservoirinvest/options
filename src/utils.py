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
from ib_insync import IB, Contract, util
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
                       how_many: int=0):
    
    """Get closest values in a list
    
    how_many: 0 gives the closest value\n
              1 | 2 | ... use for CALL fences\n
             -1 | -2 | ... use for PUT fences"""

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

    output = result[0:abs(1 if how_many == 0 else abs(how_many))]

    # if len(output) == 1:
    #     value = output[0]
    # else:
    #     value = output

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


async def qualify_me(ib: IB, contracts: list):
    """Qualify underlying contracts asynchronously"""

    # tasks = [ib.qualifyContractsAsync(c) for c in contracts]

    # results = [await task_ 
    #             for task_ 
    #             in tqdm.as_completed(tasks, total=len(tasks), desc='Qualifying Contracts')]

    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in contracts]

    await tqdm.gather(*tasks)

    result = [r for t in tasks for r in t.result()]

    return result


def make_dict_of_qualified_contracts(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

    # contracts_dict = {c[0].symbol: c[0] for c in qualified_contracts if c}
    contracts_dict = {c.symbol: c for c in qualified_contracts if c}

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

async def get_an_option_chain(ib: IB, contract:Contract):
    """Get Option Chains from IB"""

    chain = await ib.reqSecDefOptParamsAsync(
    underlyingSymbol=contract.symbol,
    futFopExchange="",
    underlyingSecType=contract.secType,
    underlyingConId=contract.conId,
    )

    chain = chain[-1] if isinstance(chain, list) else chain

    df = chain_to_df(chain, contract)

    return df

def chain_to_df(chain, contract:Contract) -> pd.DataFrame:
    """Convert an option chain to df"""

    df = pd.DataFrame([{'symbol': contract.symbol, 
                        'undId': contract.conId, 'key': 1}])

    df1 = pd.DataFrame([chain])

    dtes = [get_dte(x, 'nse') for x in df1.expirations[0]]
    df_expiries = pd.DataFrame(df1.expirations[0], columns=["expiry"]).assign(dte = dtes)

    # Do a cartesian merge for strikes and expiries
    df2 = (pd.merge(df_expiries.assign(key=1),
        pd.DataFrame(df1.strikes[0], columns=["strike"]).assign(key=1),
        on="key"))
    
    df3 = pd.merge(df, df2, on='key').drop('key', axis=1)

    return df3


async def get_market_data(ib: IB, 
                          c:Contract,
                          sleep:float = 2):

    """
    Get marketPrice including implied volatility\n   
    Pretty quick when market is closed
    """
    tick = ib.reqMktData(c, genericTickList="106")
    await asyncio.sleep(sleep)
    ib.cancelMktData(c)

    return tick


async def get_tick_data(ib: IB, 
                          c:Contract,
                          delay: float=0):

    """
    Gets tick-by-tick data\n  
    Quick when market is open \n   
    Takes ~6 secs after market hours. \n  
    No impliedVolatility"""

    ticker = await ib.reqTickersAsync(c)
    await asyncio.sleep(delay)
    # ticker = ticker[-1] if isinstance(ticker, list) else ticker

    return ticker


async def get_price_iv(ib, contract, sleep: float=2) -> dict:

    """Computes price and IV of a contract
    
    Could take upto 12 seconds in case live prices are not available"""

    mkt_data = await get_market_data(ib, contract, sleep)

    if math.isnan(mkt_data.marketPrice()):
        logger.info(f"mkt_data for {contract.localSymbol} has no market price")

        if math.isnan(mkt_data.close):
            tick_data = await get_tick_data(ib, contract)
            tick_data = tick_data[0]

            if math.isnan(tick_data.marketPrice()):
                undPrice = tick_data.close
                if math.isnan(undPrice):
                    logger.info(f"No price for {contract.localSymbol}!!!")
            else:
                undPrice = tick_data.marketPrice()
        else:
            undPrice = mkt_data.close

    else:
        undPrice = mkt_data.marketPrice()

    iv = mkt_data.impliedVolatility

    price_iv = (contract.localSymbol, undPrice, iv)

    logger.remove()

    return price_iv


def get_a_stdev(iv: float, price: float, dte: float) -> float:

    """Gives 1 Standard Deviation value.\n
    Assumes iv as annual implied volatility"""

    return iv*price*math.sqrt(dte/365)


async def combine_a_chain_with_stdev(ib: IB, contract, sleep: float=3) -> pd.DataFrame:
    """makes a dataframe from chain and stdev for a symbol"""

    price_iv = await get_price_iv(ib, contract, sleep)
    symbol, undPrice, iv = price_iv

    chain = await get_an_option_chain(ib, contract)
    
    df = chain.assign(localSymbol=contract.localSymbol, undPrice=undPrice, iv=iv)
    df = df.assign(sigma=df[['iv', 'undPrice', 'dte']].\
                    apply(lambda x: get_a_stdev(x.iv, x.undPrice, x.dte), axis=1))

    df = df.assign(sdev = (df.strike - df.undPrice) / df.sigma)

    df = df.assign(right = df.sdev.apply(lambda sdev: 'P' if sdev < 0 else 'C'))

    return df

def chunk_me(data: list, size: int=25) -> list:
    """cuts the list into chunks"""

    if type(data) is not list:
        logger.error(f"Data type needs to be a `list`, not {type(data)}")
        output = None
    else:
        output = [data[x: x+size] for x in range(0, len(data), size)]

    return output


async def make_chains(ib: IB, 
                      contracts: list, 
                      chunk_size: int=25, 
                      sleep: float=7
                      ) -> pd.DataFrame:
    """Makes chains for a list of contracts. 2m 11s for 186 contracts.\n
    ...for optimal off-market, chunk-size of 25 and sleep of 7 seconds."""

    chunks = tqdm(chunk_me(contracts, chunk_size), desc="Getting chains")
    dfs = []

    for cts in chunks:

        tasks = [asyncio.create_task(combine_a_chain_with_stdev(ib, c, sleep)) for c in cts]

        results = await asyncio.gather(*tasks)

        df = pd.concat(results, ignore_index=True)

        dfs.append(df)

    df_chains = pd.concat(dfs, ignore_index=True)

    return df_chains


def make_target_option_contracts(df_target):

    """make option contracts from target df"""

    option_contracts = [Contract(
            symbol=symbol,
            strike=strike,
            lastTradeDateOrContractMonth=expiry,
            right=right,
            exchange='NSE',
            currency='INR',
            secType='OPT')
        for symbol, strike, expiry, right \
            in zip(df_target.symbol,
                df_target.strike,
                df_target.expiry,
                df_target.right,
                )]
    
    return option_contracts

async def make_naked_puts(ib: IB, 
                    df_chains: pd.DataFrame, 
                    MARKET: str,
                    how_many: int=-2,
                    ) -> pd.DataFrame:
    
    """Make naked puts from chains"""
    
    _vars = Vars(MARKET.upper())
    
    target_puts = df_chains.groupby('symbol').sdev.\
        apply(lambda x: get_closest_values(x, _vars.PUTSTDMULT, how_many))

    target_puts.name = 'puts_sd'

    # integrate puts_sd range and compare with sdev
    df_ch1 = df_chains.set_index('symbol').join(target_puts)

    # filter target puts
    df_ch2 = df_ch1[df_ch1.apply(lambda x: x.sdev in x.puts_sd, axis=1)].drop('puts_sd', axis=1)

    df_target = df_ch2[['strike', 'expiry', 'right',]].reset_index()

    options_list = make_target_option_contracts(df_target)

    # qualify target options
    options = await qualify_me(ib, options_list)

    # generate target puts
    df_puts = util.df(options).iloc[:, 2:6].\
        rename(columns={"lastTradeDateOrContractMonth": "expiry"}).\
            assign(naked_puts=options)
    
    cols = df_puts.columns[:-1].to_list()

    # weed out other chain options
    df_output = df_chains.set_index(cols).join(df_puts.set_index(cols), on=cols)
    df_out = df_output.dropna().drop('localSymbol', axis=1).reset_index()

    return df_out


if __name__ == "__main__":
    pass




