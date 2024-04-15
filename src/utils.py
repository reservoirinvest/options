import asyncio
import datetime
import glob
import io
import json
import logging
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import requests
import yaml
from from_root import from_root
from ib_insync import (IB, Contract, Index, LimitOrder, MarketOrder, Option,
                       Stock, util)
from loguru import logger
# from nsepython import fnolist, nse_get_fno_lot_sizes
from pytz import timezone
from scipy.integrate import quad
from tqdm.asyncio import tqdm, tqdm_asyncio

from dclass import OpenOrder, Portfolio

ROOT = from_root()
BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar}{r_bar}"

# * SUPPORT FUNCTIONS
# ===================

@dataclass
class Timediff:
    """Stores time difference for file_age"""
    td: datetime.timedelta
    days: int
    hours: int
    minutes: int
    seconds: float

class Timer:
    """Timer providing elapsed time"""
    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")

        print(
            f'\n{self.name} started at {time.strftime("%d-%b-%Y %H:%M:%S", time.localtime())}'
        )

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time

        print(
            f"\n...{self.name} took: " +
            f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} seconds\n"
        )

        self._start_time = None

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


def to_list(data):
    """Converts any iterable to a list, and non-iterables to a list with a single element.

    Args:
        data: The data to be converted.

    Returns:
        A list containing the elements of the iterable, or a list with the single element if the input is not iterable.
    """

    try:
        return list(data)
    except TypeError:
        return [data]


def is_unique(s: Union[pd.Series, pd.DataFrame]) -> bool:
    """A fast unique checker of series or df"""
    a = s.to_numpy()
    return (a[0] == a).all()


async def isMarketOpen(MARKET: str) -> bool:
    """[async] Determines if market is open or not

    Args:

        (ib): as connection object,
        (MARKET): ('NSE'|'SNP')

    Returns:
        bool

    Note:
     - Though IB uses UTC elsewhere, in contract details `zone` is available as a string!
     - ...hence times are converted to local market times for comparison

    """
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    # Establish the timezones
    tzones = {
        "UTC": timezone("UTC"),
        "Asia/Calcutta": timezone("Asia/Kolkata"),
        "US/Eastern": timezone("US/Eastern"),
    }

    with await IB().connectAsync(port=PORT) as ib:

        if MARKET.upper() == "NSE":
            ct = await ib.qualifyContractsAsync(
                Stock(symbol="RELIANCE", exchange="NSE", currency="INR"))
        elif MARKET.upper() == "SNP":
            ct = await ib.qualifyContractsAsync(
                Stock(symbol="INTC", exchange="SMART", currency="USD"))
        else:
            print(f"\nUnknown market {MARKET}!\n")
            return None

        ctd = await ib.reqContractDetailsAsync(ct[0])

    hrs = util.df(ctd).liquidHours.iloc[0].split(";")
    zone = util.df(ctd).timeZoneId.iloc[0].split(" ")[0]


    # Build the time dataframe by splitting hrs
    tframe = pd.DataFrame([re.split(":|-|,", h) for h in hrs]).rename(columns={
        0: "from",
        1: "start",
        2: "to",
        3: "end"
    })
    tframe["zone"] = zone

    tframe["now"] = datetime.datetime.now(tzones[zone])

    tframe = tframe.dropna()

    open = pd.to_datetime(tframe["from"] + tframe["start"]).apply(
        lambda x: x.replace(tzinfo=tzones[zone]))
    close = pd.to_datetime(tframe["to"] + tframe["end"]).apply(
        lambda x: x.replace(tzinfo=tzones[zone]))

    tframe = tframe.assign(open=open, close=close)
    tframe = tframe.assign(isopen=(tframe["now"] >= tframe["open"])
                            & (tframe["now"] <= tframe["close"]))

    market_open = any(tframe["isopen"])

    return market_open


def floatxtract(input: list):
    """Extracts float from given list if possible"""

    mixedList = re.split('( |, )', input )

    out = next(iter([float(y) 
                     for y in mixedList 
                        if re.match(r'^-?\d+(?:\.\d+)$', y)]))
    return out


def chunk_me(data: list, size: int=25) -> list:
    """cuts the list into chunks"""

    if type(data) is not list:
        logger.error(f"Data type needs to be a `list`, not {type(data)}")
        output = None
    else:
        output = [data[x: x+size] for x in range(0, len(data), size)]

    return output


def delete_files(file_list: list):
    """Delete files with paths as a string"""

    for file_path in file_list:

        if isinstance(file_path, Path):
            file_path = str(file_path)

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)

        except Exception as e:
            print(f"Failed to delete {file_path}. Error: {str(e)}")
            with open(file_path, "w") as file:
                file.write('')


def delete_all_pickles(MARKET: str):
    """Deletes all pickle files for the MARKET"""

    pickle_folder_path = ROOT / 'data' / MARKET.lower()
    file_pattern = glob.glob(str(pickle_folder_path / '*.pkl'))

    delete_files(file_pattern)

    return None


def join_my_df_with_another(my_df: pd.DataFrame, 
                            other_df: pd.DataFrame,
                            idx: Union[str, None] = None) -> pd.DataFrame:
    """
    Joins my df with other, protecting my columns
    my_df: original df
    other_df: df columns to be imported from
    idx: index as string. Should be a common field.

    """
    
    if idx:
        try:
            my_df = my_df.set_index(idx)
        except KeyError:
            pass

        try:
            other_df = other_df.set_index(idx)
        except KeyError:
            pass

    if other_df.index.name != my_df.index.name:
        logger.error(f"Index names of the m_df: {my_df.index.name} and other_df: {other_df.index.name} are different")
        return None


    cols2keep = [c for c in other_df.columns if c not in my_df.columns]
    df_out = my_df.join(other_df[cols2keep])

    return df_out


def get_prec(v, base):
    """Gives the precision value

    Args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05
    Returns:
        the precise value"""

    try:
        output = round(
            round((v) / base) * base, -int(math.floor(math.log10(base))))
    except Exception:
        output = None

    return output


def get_prob(sd):
    """Compute probability of a normal standard deviation

    Arg:
        (sd) as standard deviation
    Returns:
        probability as a float

    """
    prob = quad(lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), -sd, sd)[0]
    return prob


def flatten(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def split_time_difference(diff: datetime.timedelta):
    """Splits time diference into d, h, m, s"""
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += diff.microseconds / 1e6

    output = Timediff(*(diff, days, hours, minutes, seconds))

    return output


def move_column(df, column_name, new_position):
  """
  Moves a column to a specific position in a DataFrame.

  Args:
      df: The DataFrame to modify.
      column_name: The name of the column to move.
      new_position: The desired position for the column.

  Returns:
      A new DataFrame with the column moved to the specified position.
  """
  # Ensure modifications are done on a copy to prevent unintended changes
  new_df = df.copy()
  
  # Check if column exists
  if column_name not in new_df.columns:
    raise ValueError(f"Column '{column_name}' not found in DataFrame.")

  # Pop and insert the column
  col = new_df.pop(column_name)
  new_df.insert(new_position, column_name, col)

  return new_df


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
    Rets: dte as float | timestamp in local timzeone (or) NaN"""

    if type(dt) is str:
        dt = pd.to_datetime(dt)

    tz_dict = {'nse': ('Asia/Kolkata', 18), 'snp': ('EST', 16)}
    tz, hr = tz_dict[exchange.lower()]

    mkt_tz = pytz.timezone(tz)
    mkt_close_time = datetime.time(hr, 0)

    now = datetime.datetime.now(tz=mkt_tz)

    try:
        mkt_close = datetime.datetime.combine(dt.date(), mkt_close_time).astimezone(mkt_tz)
    except OSError:
        dte = np.nan
    else:
        dte = (mkt_close-now).total_seconds()/24/3600

        if time_stamp:
            dte = mkt_close

    return dte


def make_a_choice(choice_list: list,
                  msg_header: str="Choose from the following:"):

    """
    Outputs a dictionary from the choices made
    Use next(iter(output.keys())) to get the choice number
    """

    # Processing
    choice_dict = {i+1 : v for i, v in enumerate(choice_list)}
    choice_dict[0] = 'Exit'

    # loop through choice
    ask = f"{msg_header}\n"

    for choice, value in choice_dict.items():
        if choice > 0:
            ask = ask + f"{str(choice)}) {value}\n"
        else:
            ask = ask + f"\n{str(choice)}) {value}\n"

    while True:
        data = input(ask) # check for int in input
        try:
            selected_int = int(data)
        except ValueError:
            print("\nI didn't understand what you entered. Try again!\n")
            continue  # Loop again
        if not selected_int in choice_dict.keys():
            print(f"\nWrong number! choose between {list(choice_dict.keys())}...")
        else:
            output= {selected_int: choice_dict[selected_int]}
            break

    if selected_int == 0:
        print(f"...Exited")

    return output


def pickle_me(obj, file_name_with_path: Path):
    """Pickles objects in a given path"""
    
    with open(str(file_name_with_path), 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_with_age_check(obj: dict, 
                file_name_with_path: Path, 
                minimum_age_in_days: float=1):
    """Pickles an object after checking file age"""

    existing_file_age = get_file_age(file_name_with_path)
    
    seconds_in_a_day = 24*60*60
    if existing_file_age:
        file_age_in_days = existing_file_age.td.total_seconds() / seconds_in_a_day
    else:
        file_age_in_days = 0
        
    if existing_file_age is None: # No file exists
        to_pickle = True
    elif file_age_in_days >= minimum_age_in_days:
        to_pickle = True
    else:
        to_pickle = False

    if to_pickle:
        pickle_me(obj, file_name_with_path)
        # logger.info(f"Pickled underlying contracts to {file_name_with_path}")
    else:
        logger.info(f"Not pickled as {file_name_with_path}'s age {existing_file_age.days} is < {minimum_age_in_days}")


def get_pickle(path: Path, print_msg: bool=True):
    """Gets pickled object"""

    output = None # initialize

    try:
        with open(path, 'rb') as f:
            output = pickle.load(f)
    except FileNotFoundError:
        if print_msg:
            logger.error(f"file not found: {path}")
    
    return output


def get_exchange_name(df: pd.DataFrame) -> str:
    """Gets the exchange name <`NSE`> | <`SMART`>"""
    
    try:
        out = set(df.exchange.to_numpy()).pop()
    except AttributeError:
        try: 
            out = set(df.market.to_numpy()).pop()
        
        except AttributeError:
            logger.error("Unknown Exchange in df")
            out = None

    if out == 'SNP':
        out = 'SMART'

    return out


def get_market_name(df: pd.DataFrame) -> str:
    """Returns MARKET: `SNP` | `NSE`"""

    out = get_exchange_name(df) 
    if out != 'NSE':
        out = 'SNP'
    
    return out


# * IB SUPPORT FUNCTIONS
# ======================

def make_ib_contracts(symbols: Union[list, set], MARKET: str, secType: str = 'STK'):

    """makes equity contracts from a symbol list"""

    symbols = to_list(symbols)

    symbols = pd.Series(symbols, name='symbol')

    ROOT = from_root()
    DATAPATH = ROOT / 'data' / MARKET.lower()

    und_dict = get_pickle(DATAPATH / 'unds.pkl')

    if und_dict:
        contracts = [und_dict.get(s, None) for s in symbols]

    else:
        EXCHANGE = 'NSE' if MARKET.upper() == 'NSE' else 'SMART'

        contracts = [Contract(
                symbol=symbol,
                exchange=EXCHANGE,
                currency= 'USD' if MARKET.upper() == 'SNP' else 'INR',
                secType=secType)
            for symbol in symbols]
    
    return contracts


async def qualify_me(ib: IB, 
                     contracts: list,
                     desc: str = 'Qualifying contracts'):
    """[async] Qualify contracts asynchronously"""

    contracts = to_list(contracts) # to take care of single contract

    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in contracts]

    await tqdm_asyncio.gather(*tasks, desc=desc)

    result = [r for t in tasks for r in t.result()]

    return result


async def qualify_conIds(PORT: int, 
                         conIds: list,
                         desc: str = "Qualifying conIds"):
    """[async] Makes and qualifies contracts from conId list"""

    conIds = to_list(conIds) # to take care of single conId
    
    contracts = [Contract(conId=conId) for conId in conIds]

    with await IB().connectAsync(port=PORT) as ib:
        qualified_opts = await qualify_me(ib, contracts, desc=desc)

    return qualified_opts


def get_lots(contract):
    """
    Retrieves lots based on contract.

    Args:
        contract: ib_insync contract. could be Contract, Option, Stock or Index type.
        lots_path: A dictionary from `lots.pkl` for retrieving values associated with symbols (if required).

    Returns:
        The lots. If no match found returns None.
    """

    MARKET = contract.exchange.upper()
    DATAPATH = ROOT / 'data' / MARKET.lower()

    lots = get_pickle(DATAPATH / 'lots.pkl') if MARKET == 'NSE' else None

    match (contract.secType, contract.exchange):
        
        # Gets the lots of NSE options
        case ('OPT', 'NSE'):
            output = lots.get(contract.symbol, None)

        # Lots for SNP options
        case ('OPT', 'SMART'):
            output = 1

        # Any exchange other than 'NSE' for options
        case ('OPT', _):  
            output = 1
        
        # Any security type on the 'NSE' exchange other than options
        case (_, 'NSE'):
            output = lots.get(contract.symbol, None)
        
        # Any other combination (non-options on non-'NSE' exchanges)
        case (_, _):  
            output = 100

    return output
        

def clean_ib_util_df(contracts: Union[list, pd.Series]) -> pd.DataFrame:

    """Cleans ib_insync's util.df to keep only relevant columns"""

    df1 = pd.DataFrame([]) # initialize 

    if isinstance(contracts, list):
        df1 = util.df(contracts)
    elif isinstance(contracts, pd.Series): 
        try:
            contract_list = list(contracts)
            df1 = util.df(contract_list) # it could be a series
        except (AttributeError, ValueError):
            logger.error(f"cannot clean type: {type(contracts)}")
    else:
        logger.error(f"cannot clean unknowntype: {type(contracts)}")
        
    if not df1.empty:
    
        df1.rename({"lastTradeDateOrContractMonth": "expiry"},
                                        axis="columns",
                                        inplace=True)

        df1 = df1.assign(expiry = pd.to_datetime(df1.expiry))
        cols = list(df1.columns[:6])
        cols.append('multiplier')
        df2 = df1[cols]
        df2 = df2.assign(contract=contracts)
    
    else:
        df2 = None
    

    return df2


# * BUILDING THE BASE
# ===================

def get_unds(MARKET: str, MAX_UND_DAYS: float=1):

    """
    Gets underlyings for the market and pickles
    MARKET: `NSE` | `SNP`
    MAX_UND_DAYS: will recreate unds if pickle is older than this"""
    
    DATAPATH = ROOT / 'data' / MARKET.lower()
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    UNDS_PATH = DATAPATH / 'unds.pkl'
    unds = get_pickle(UNDS_PATH)

    #... check file age
    unds_file_age = get_file_age(UNDS_PATH)
    if not unds_file_age:
        unds_age_in_days = MAX_UND_DAYS
    else:
        unds_age_in_days = unds_file_age.td.total_seconds()/60/60/12 

    if unds_age_in_days >= MAX_UND_DAYS:
        # Assemble underlyings
        if MARKET == 'SNP':
            unds = asyncio.run(assemble_snp_underlyings(PORT))
        else:
            unds = asyncio.run(assemble_nse_underlyings(PORT))

        pickle_me(unds, UNDS_PATH)

    return unds


def get_unds_with_prices(MARKET: str, max_age_in_mins:float=10) -> pd.DataFrame:
    """
    Gets underlyings with prices for a market
    age_in_mins: checks freshness of `df_und_prices.pkl`
    
    """

    DATAPATH = ROOT / 'data' / MARKET.lower()
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    unds = get_unds(MARKET=MARKET, MAX_UND_DAYS=0)

    # Make df from unds
    df_unds = pd.DataFrame.from_dict(unds.items())
    df_unds.columns = ['symbol', 'contract']

    # Get und prices
    # ... check file age.
    UND_PRICE_PATH = DATAPATH / 'df_und_prices.pkl'
    und_price_file_age = get_file_age(UND_PRICE_PATH)

    if und_price_file_age:
        age_in_mins = und_price_file_age.td.total_seconds()/60
    else:
        age_in_mins = max_age_in_mins

    if age_in_mins >= max_age_in_mins:
        df_und_prices = asyncio.run(get_mkt_prices(port=PORT, contracts=df_unds.contract))
        pickle_me(df_und_prices, UND_PRICE_PATH)
    else:
        df_und_prices = get_pickle(UND_PRICE_PATH)

    return df_und_prices


def get_chains(MARKET:str, MAX_DAYS_OLD: float=1) -> pd.DataFrame:
    """
    Returns / build chains. Recalculates sigma, iv.
    MAX_DAYS_OLD: will build chains if pickled chains are older than this

    """
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    # Get und contracts
    unds = get_unds(MARKET)
    contracts = unds.values()

    # Get chains
    DATAPATH = ROOT / 'data' / MARKET.lower()
    CHAIN_PATH = DATAPATH / 'df_chains.pkl'
    chains_file_age = get_file_age(CHAIN_PATH)

    if not chains_file_age:
        age_in_days = MAX_DAYS_OLD
    else:
        age_in_days = chains_file_age.td.total_seconds()/60/60/12
        
    if age_in_days >= MAX_DAYS_OLD:
        df_chains = asyncio.run(make_chains(port=PORT, contracts=contracts, MARKET=MARKET))
    else:
        df_chains = get_pickle(CHAIN_PATH)
    
    if age_in_days >= 1:
        pickle_me(df_chains, CHAIN_PATH)

    return df_chains


def update_chain_strikes(df_chains: pd.DataFrame) -> pd.DataFrame:
    """
    Updates dte, strike and sd of chains
    """
    # Update dte
    MARKET = df_chains.exchange.iloc[0]
    dte = pd.to_datetime(df_chains.expiry).apply(lambda x: get_dte(x, MARKET))

    df_chains = df_chains.assign(dte = dte)

    # Integrate undPrices and undIV to chains
    df_und_prices = get_unds_with_prices(MARKET)

    und_price_dict = df_und_prices.set_index('symbol').price.dropna().to_dict()
    und_iv_dict = df_und_prices.set_index('symbol').iv.dropna().to_dict()

    # Replace undPrice and ivs where available
    df_chains.undPrice = df_chains.symbol.map(und_price_dict).fillna(df_chains.undPrice)
    df_chains.iv = df_chains.symbol.map(und_iv_dict).fillna(df_chains.iv)

    # Recalculate `sigma` and `strike_sdev`
    df_chains = compute_strike_sd_right(df_chains)

    return df_chains


def make_chains_with_margins(MARKET: str, MAX_DAYS_OLD: int=0) -> pd.DataFrame:
    """
    Makes chains with margins for all options and pickles
    """

    DATAPATH = ROOT / 'data' / MARKET.lower()
    CHAIN_PATH = DATAPATH / 'df_chains.pkl'
    
    df_chains = get_chains(MARKET=MARKET, MAX_DAYS_OLD=MAX_DAYS_OLD)

    # Update chain strikes
    df_chains = update_chain_strikes(df_chains)

    # Get chains with strikes closest to underlying
    df_ch1 = get_strike_closest_to_und(df_chains)

    # Get underlying margins of chains
    df_chund_margins = opt_margins_with_lot_check(df_ch1)

    # update df_ch_unds with latest margin and commission
    opt_cols = ['symbol', 'expiry', 'strike', 'right']
    margin_comm = ['margin', 'comm']
    df_ch2 = join_my_df_with_another(df_ch1, df_chund_margins[opt_cols+margin_comm])

    # second run
    df_rem = df_ch2[df_ch2.margin.isnull()]

    # replace df_ch2 with new margin and comm
    if not df_rem.empty:
        dfr = opt_margins_with_lot_check(df_rem, multiply_lot=False)
        dfr1 = dfr.set_index(opt_cols).dropna(subset=['margin'])
        df_ch2.set_index(opt_cols).update(dfr1, overwrite=False)

    # Integrate margins and commissions
    df_ch_all = join_my_df_with_another(df_chains, 
                                        df_ch2[opt_cols+margin_comm], 
                                        idx=opt_cols).reset_index()
    
    # fill missing commissions with max per symbol
    commissions = df_ch2.groupby('symbol').comm.max().to_dict()
    df_ch_all.comm = df_ch_all.comm.fillna(df_ch_all.symbol.map(commissions))

    # fill remaining commissions
    df_ch_all.comm = df_ch_all.comm.fillna(max(commissions.values()))

    # fill margins
    mgn_dict = df_ch_all.groupby('symbol').margin.max().to_dict()
    cond = df_ch_all.margin.isnull()
    df_ch_all.loc[cond, 'margin'] = df_ch_all[cond].symbol.map(mgn_dict)

    # make zero margin as nan
    zero_margin_condition = df_ch_all.margin == 0
    df_ch_all.loc[zero_margin_condition, 'margin'] = np.nan

    # compute margins for remaining options
    if MARKET == 'SNP':
        df = compute_snp_df_margins(df_ch_all)
    else:
        df = df_ch_all # Usually there are no remaining NSE null margins.

    pickle_me(df, CHAIN_PATH)

    return df


def raw_target_opts(df: pd.DataFrame,
                    how_many: int=2) -> pd.DataFrame:
    """
    Generate target options from df.

    Ensures xtra standard deviation is added to dte near expiries
    
    `how_many`: positive integer gives records closest to std dev"""

    MARKET = get_market_name(df)
    _vars = Vars(MARKET)

    COLS = _vars.COLS[0]

    DTESTDEVLOW = _vars.DTESTDEVLOW
    DTESTDEVHI = _vars.DTESTDEVHI
    DECAYRATE = _vars.DECAYRATE

    CALLSTDMULT = _vars.CALLSTDMULT
    PUTSTDMULT = _vars.PUTSTDMULT 

    MARKET_IS_OPEN = asyncio.run(isMarketOpen(MARKET))
    GAPBUMP = 0 if MARKET_IS_OPEN else _vars.GAPBUMP

    # Get corresponding sdev for dtes
    g = df.groupby('symbol')

    # Get extra sdev for dtes closer to expiry
    xtra_sd = g.dte.transform(lambda d: sdev_for_dte(dte=d, 
                                                DTESTDEVHI=DTESTDEVHI, 
                                                DTESTDEVLOW=DTESTDEVLOW, 
                                                DECAYRATE=DECAYRATE, 
                                                GAPBUMP=GAPBUMP))

    # add puts / calls mult to xtra_sd, that is signed -ve to puts / +ve to calls
    # ... this correctly `adds` xtra_sd
    sd_mults = np.where(df.right == 'C', CALLSTDMULT, PUTSTDMULT)
    new_sd = sd_mults + np.where(df.right == 'C', 1, -1)*xtra_sd

    # get the closest values
    df_ch = df.assign(new_sd = new_sd)[COLS + ['undPrice', 'dte', 'strike_sdev', 'new_sd']]

    # sort first
    argsorted = df_ch.iloc[abs(df_ch.strike_sdev - df_ch.new_sd).argsort()]

    # keep only positive differences
    mask_diff = argsorted.strike_sdev > argsorted.new_sd
    sorted_and_masked = argsorted.loc[mask_diff]

    # keep only the top `how_many`
    tops = sorted_and_masked.groupby(['symbol', 'expiry', 'right']).head(how_many)

    # sort the tops on expiry and right
    idx = tops.sort_values(['symbol', 'expiry', 'strike']).index
    df_out = df.loc[idx, :]

    return df_out


def qualify_naked_opts(df: pd.DataFrame) -> pd.DataFrame:
    """Qualify raw target option chains and get margins and comm"""

    MARKET = get_market_name(df)

    _vars = Vars(MARKET)
    PORT = _vars.PORT


    EXCHANGE = get_exchange_name(df)
    target_opts = [make_a_raw_contract(symbol=symbol, EXCHANGE=EXCHANGE, secType='OPT', strike=strike, right=right, expiry=expiry) 
    for symbol, expiry, strike, right 
            in zip(df.symbol, df.expiry, df.strike, df.right)]

    with IB().connect(port=PORT) as ib:
        tgt_opts = asyncio.run(qualify_me(ib, target_opts, desc=f"{MARKET} naked quals"))

    contracts = [c if c.conId else None for c in tgt_opts]

    df_clean = clean_ib_util_df(contracts=contracts).drop(columns=['secType', 'multiplier', 'conId'])
    df_clean.expiry = df_clean.expiry.dt.strftime("%Y%m%d")
    df_out = pd.merge(df, df_clean).drop(columns='undId')
    
    df_out.drop(columns = ['localSymbol', 'undId', 'margin', 'comm'], errors='ignore', inplace=True)
    df_out.insert(0, 'conId', [int(c.conId) for c in df_out.contract])

    df_margins = asyncio.run(get_margins(port=PORT, 
                                         contracts=df_out.contract, 
                                         desc=f'{MARKET} naked opt margins'))

    df_return = pd.merge(df_out, df_margins[['conId', 'margin', 'comm']], on='conId')

    return df_return


def nakeds_with_expPrice(df: pd.DataFrame) -> pd.DataFrame:
    """Makes a df with expected price"""

    MARKET = get_market_name(df)
    _vars = Vars(MARKET)

    PORT = port = _vars.PORT

    MINEXPROM = _vars.MINEXPROM
    PREC = _vars.PREC
    MINOPTSELLPRICE = _vars.MINOPTSELLPRICE
    COLS = _vars.COLS[0]

    # rename df iv
    df.rename(columns={'iv':'und_iv'}, inplace=True, errors='ignore')

    # get prices of nakeds
    df_prices = asyncio.run(get_prices_with_ivs(port=PORT, input_contracts=df.contract))

    # convert expiry to string to combine with df
    dfp = df_prices.assign(expiry = df_prices.expiry.dt.strftime("%Y%m%d"))
    dfp[COLS + ['optPrice']]

    # integrate option price
    df_nakeds = pd.merge(df, dfp[COLS + ['optPrice']], on=COLS)

    # make zero margin as nan
    zero_margin_condition = df_nakeds.margin == 0
    df_nakeds.loc[zero_margin_condition, 'margin'] = np.nan

    # remove nans from margins
    df_nakeds = df_nakeds[~df_nakeds.margin.isnull()].reset_index(drop=True)

    # fill missing commissions with max per symbol
    commissions = df_nakeds.groupby('symbol').comm.max().to_dict()
    df_nakeds.comm = df_nakeds.comm.fillna(df_nakeds.symbol.map(commissions))

    # fill remaining commissions
    df_nakeds.comm = df_nakeds.comm.fillna(max(commissions.values()))

    # fill margins
    mgn_dict = df_nakeds.groupby('symbol').margin.max().to_dict()
    cond = df_nakeds.margin.isnull()
    df_nakeds.loc[cond, 'margin'] = df_nakeds[cond].symbol.map(mgn_dict)

    # Get precise expected prices
    xp = ((MINEXPROM*df_nakeds.dte/365*df_nakeds.margin) +
            df_nakeds.comm)/df_nakeds.lot/df_nakeds.multiplier

    # Set the minimum option selling price
    xp[xp < MINOPTSELLPRICE] = MINOPTSELLPRICE

    # Make the expected Price
    expPrice = pd.concat([xp, 
                            df_nakeds.optPrice], axis=1)\
                            .max(axis=1)
    expPrice = expPrice.apply(lambda x: get_prec(x, PREC))
    dfn = df_nakeds.assign(expPrice = expPrice)

    # Limit the number of options to trade
    cond = dfn.expPrice < 3* dfn.optPrice
    dfn = dfn[cond]

    return dfn


def prepare_to_sow(market: str,
                   PAPER: bool=False,
                   build_from_scratch: bool=False,
                   save_sow: bool=True,
                   ) -> pd.DataFrame:
    """Prepares the naked sow dataframe"""

    MARKET = market.upper()
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    if PAPER:
        PORT = _vars.PAPER

    # puts_only = True if MARKET == 'SNP' else False
    puts_only = False

    if build_from_scratch:

        # delete logs
        files = [ROOT/'log'/ f'{MARKET.lower()}_app.log',
                 ROOT/'log'/ f'{MARKET.lower()}_ib.log']
        delete_files(file_list=files)

        delete_all_pickles(MARKET)

        build_base(market=MARKET,
                   PAPER=PAPER,
                   puts_only=puts_only)

    df = create_target_opts(market=MARKET)

    # Get the portfolio and open orders        
    df_openords, df_pf = asyncio.run(get_order_pf(PORT))

    # Remove targets which are already in the portfolio
    if isinstance(df_pf, pd.DataFrame):
        df = df[~df.symbol.isin(set(df_pf.symbol))]      

    # Remove open orders from df
    if ~df_openords.empty:
        df = df[~df.symbol.isin(set(df_openords.symbol))]

    # Keep options with margin and expected price only
    margins_only = ~df.margin.isnull()
    with_expPrice = ~df.expPrice.isnull()

    cleaned = margins_only & with_expPrice

    df = df[cleaned]

    # Sort with most likely ones on top
    df = df.loc[(df.expPrice/df.optPrice)\
                   .sort_values().index]\
                    .reset_index(drop=True)

    if save_sow:
        DATAPATH = ROOT / 'data' / MARKET
        SOW_PATH = DATAPATH / 'df_sow.pkl'
        pickle_me(df, SOW_PATH)

        # Store sows in temp path for analysis
        MARKET_IS_OPEN = asyncio.run(isMarketOpen(MARKET))
        pickle_the_sow(df, _vars, MARKET_IS_OPEN)
        
    return df


def build_base(market: str,
               PAPER: bool,
               puts_only: bool = True) -> pd.DataFrame:
    """Freshly build the base and pickle"""

    # Set variables
    MARKET = market.upper()
    _vars = Vars(MARKET)

    if PAPER:
        PORT = _vars.PAPER
    else:
        PORT = _vars.PORT

    CALLSTDMULT = _vars.CALLSTDMULT
    PUTSTDMULT = _vars.PUTSTDMULT

    DTESTDEVLOW = _vars.DTESTDEVLOW
    DTESTDEVHI = _vars.DTESTDEVHI
    DECAYRATE = _vars.DECAYRATE

    # Set paths for nse pickles
    unds_path = ROOT / 'data' / MARKET / 'unds.pkl'
    chains_path = ROOT / 'data' / MARKET / 'df_chains.pkl'
    lots_path = ROOT / 'data' / MARKET / 'lots.pkl'

    qualified_puts_path = ROOT / 'data' / MARKET / 'df_qualified_puts.pkl'
    qualified_calls_path = ROOT / 'data' / MARKET / 'df_qualified_calls.pkl'

    opt_prices_path = ROOT / 'data' / MARKET / 'df_opt_prices.pkl'
    opt_margins_path = ROOT / 'data' / MARKET / 'df_opt_margins.pkl'

    naked_targets_path = ROOT / 'data' / MARKET / 'df_naked_targets.pkl'

    # Delete log files
    log_folder_path = ROOT / 'log' / str(MARKET.lower()+"*.log")
    file_pattern = glob.glob(str(log_folder_path))

    delete_files(file_pattern)

    # Set the logger with logpath
    IBI_LOGPATH = ROOT / 'log' / f'{MARKET.lower()}_ib.log'
    LOGURU_PATH = ROOT / 'log' / f'{MARKET.lower()}_app.log'

    util.logToFile(IBI_LOGPATH, level=logging.ERROR)
    logger.add(LOGURU_PATH, rotation='10 MB', compression='zip', mode='w')

    # Assemble underlyings
    if MARKET == 'SNP':
        unds = asyncio.run(assemble_snp_underlyings(PORT))
    else:
        unds = asyncio.run(assemble_nse_underlyings(PORT))

    # pickle underlyings
    # pickle_with_age_check(unds, unds_path, 0) # No need to age-check, for fresh base build
    pickle_me(unds, unds_path)

    # Make chains for underlyings and limit the dtes
    df_chains = asyncio.run(make_chains(port=PORT,
                                        MARKET=MARKET, 
                                        contracts=list(unds.values())))
    df_chains = df_chains[df_chains.dte <= _vars.MAXDTE].reset_index(drop=True)
    pickle_with_age_check(df_chains, chains_path, 0)

    # Qualified put and options generated from the chains
    df_qualified_puts = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=PUTSTDMULT,
                            DTESTDEVLOW = DTESTDEVLOW,
                            DTESTDEVHI = DTESTDEVHI,
                            DECAYRATE=DECAYRATE,
                            how_many=-1, desc=f'Qualifying {MARKET} Puts'))     
    pickle_with_age_check(df_qualified_puts, qualified_puts_path, 0)

    df_qualified_calls = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=CALLSTDMULT,
                            DTESTDEVLOW = DTESTDEVLOW,
                            DTESTDEVHI = DTESTDEVHI,
                            DECAYRATE=DECAYRATE,
                            how_many=1, desc=f"Qualifying {MARKET} Calls"))     
    pickle_with_age_check(df_qualified_calls, qualified_calls_path, 0)

    if puts_only:
        df_all_qualified_options = df_qualified_puts
    else:
        df_all_qualified_options = pd.concat([df_qualified_calls, 
                                          df_qualified_puts], 
                                          ignore_index=True)

    # Get the option prices
    df_opt_prices = asyncio.run(get_prices_with_ivs(PORT, df_all_qualified_options, desc="Getting option prices"))
    pickle_with_age_check(df_opt_prices, opt_prices_path, 0)

    # Get the lots for nse
    if MARKET == 'NSE':
        lots = make_nse_lots()        
        # pickle_with_age_check(lots, lots_path, 0) # make_nse_lots automatically pickles

    # Get the option margins
    if MARKET == 'NSE':
        df_opt_margins = asyncio.run(get_margins(port=PORT, 
                                                 contracts=df_all_qualified_options,))
    else: # no need for lots_path
        df_opt_margins = asyncio.run(get_margins(port=PORT, 
                                                 contracts=df_all_qualified_options,))
    pickle_with_age_check(df_opt_margins, opt_margins_path, 0)

    # Get all the options
    df_naked_targets = create_target_opts(market=MARKET)
    pickle_with_age_check(df_naked_targets, naked_targets_path, 0)

    return df_naked_targets


def build_base_and_pickle(MARKET: str, PAPER: bool) -> pd.DataFrame:

    # Start the timer
    program_timer = Timer(f"{MARKET} base building")
    program_timer.start()

    df = prepare_to_sow(MARKET,
                        PAPER,
                        build_from_scratch=True,
                        save_sow=True)
    
    program_timer.stop()

    return df


def get_sows_from_pickles(MARKET: str, PAPER: bool) -> pd.DataFrame:

    df = prepare_to_sow(MARKET, 
                        PAPER,
                        build_from_scratch=False,
                        save_sow=False)

    return df


def sow_me(MARKET: str, 
           PAPER: bool = False,
           build_from_scratch: bool = False,
           save_sow: bool = False):
    """
    Cancels existing open orders and sows afresh

    Sows always will delete existing API open orders
    """
    _vars = Vars(MARKET)
    ACTIVE_STATUS = Vars(MARKET).ACTIVE_STATUS

    if PAPER:
        PORT = _vars.PAPER
    else:
        PORT = _vars.PORT

    df = prepare_to_sow(MARKET, 
            build_from_scratch = build_from_scratch,
            save_sow = save_sow,
            )
    
    # get portfolio
    df_current_openords, df_pf = asyncio.run(get_order_pf(PORT))

    # get current open orders not be cancelled.
    # ...this could be due to manual TWS orders or `reap` orders
    df_2protect = df_current_openords[df_current_openords.symbol.isin(df_pf.symbol)]   

    orders_to_cancel = df_current_openords[~df_current_openords.symbol.isin(df_2protect.symbol)].order.to_list()

    asyncio.run(cancel_orders(PORT, orders_to_cancel))
    
    # Place new orders
    # ... Make (contract, order) tuple

    cos = [(contract , LimitOrder('Sell', qty, price))
        for contract, qty, price in zip(df.contract, df.lot, df.expPrice)]

    orders = asyncio.run(place_orders_async(MARKET=MARKET, 
                            PAPER=PAPER, 
                            cos=cos))
    
    success = [o for o in orders 
            if o.orderStatus.status in ACTIVE_STATUS]
    
    logger.info(f"Sowed {len(success)} orders")

    return df


def build_base_without_pickling(MARKET: str, PAPER: bool) -> pd.DataFrame:

    # Start the timer
    program_timer = Timer(f"Getting {MARKET} sows from pickles")
    program_timer.start()

    df = prepare_to_sow(MARKET,
                        PAPER,
                        build_from_scratch=True,
                        save_sow=False)
    
    program_timer.stop()

    return df


def make_a_raw_contract(symbol: str, EXCHANGE: str, 
                    secType: str=None, strike: float=None, 
                    right: str=None, expiry: str=None) -> Contract:
    
    """Makes a stock or option contract"""
    SECTYPE = "STK" if secType == None else secType
    CURRENCY = "INR" if EXCHANGE == "NSE" else "USD"

    if (SECTYPE == 'OPT') & (CURRENCY == 'USD'):
        MULTPLIER = 100 # for SNP Options
    else:
        MULTPLIER = 1 # for NSE Options
    

    if SECTYPE == 'STK':
        ct = Stock(symbol=symbol, exchange=EXCHANGE, currency=CURRENCY)
    elif SECTYPE == 'OPT':
        ct = Contract(secType='OPT', symbol=symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right=right, exchange=EXCHANGE, multiplier=MULTPLIER, currency=CURRENCY)
    else:
        logger.error(f"Unknown symbol {symbol} in {EXCHANGE} market")

    return ct


def make_dict_of_qualified_contracts(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

    contracts_dict = {c.symbol: c for c in qualified_contracts if c}

    return contracts_dict


def make_target_option_contracts(df_target: list, MARKET: str):

    """make option contracts from target df"""

    EXCHANGE = 'NSE' if MARKET.upper() == 'NSE' else 'SMART'

    option_contracts = [Contract(
            symbol=symbol,
            strike=strike,
            lastTradeDateOrContractMonth=expiry,
            right=right,
            exchange=EXCHANGE,
            currency= 'USD' if MARKET.upper() == 'SNP' else 'INR',
            secType='OPT')
        for symbol, strike, expiry, right \
            in zip(df_target.symbol,
                df_target.strike,
                df_target.expiry,
                df_target.right,
                )]
    
    return option_contracts


def target_options_with_adjusted_sdev(df_chains: pd.DataFrame,
                                      STDMULT: float,
                                      how_many: int,
                                      DTESTDEVLOW: float, 
                                      DTESTDEVHI: float,
                                      DECAYRATE: float,
                                      MARKET_IS_OPEN: bool) -> pd.DataFrame:
    
    """Adjust the standard deviation to DTE, penalizes DTES closer to zero"""

    # Get the extra SD adjusted to DTE
    # xtra_sd = 1-(df_chains.dte/100)

    MARKET = df_chains.exchange.unique()[0]

    _vars = Vars(MARKET)

    # Factor a bump to dev if market is not open
    if MARKET_IS_OPEN:
        GAPBUMP = 0
    else:
        GAPBUMP = _vars.GAPBUMP

    xtra_sd = df_chains.dte.apply(lambda dte: sdev_for_dte(dte=dte,
                                                           DTESTDEVLOW=DTESTDEVLOW, 
                                                           DTESTDEVHI=DTESTDEVHI,
                                                           DECAYRATE=DECAYRATE,
                                                           GAPBUMP=GAPBUMP
                                                           ))

    # Build the series for revised SD
    sd_revised = STDMULT + xtra_sd if STDMULT > 0 else STDMULT - xtra_sd

    # Identify the closest standerd devs to the revised SD\
    df_ch = df_chains.assign(sd_revised=sd_revised)

    # closest_sdevs = df_ch.groupby(['symbol', 'dte'])[['sdev', 'sd_revised']]\
    #     .apply(lambda x: get_closest_values(x.sdev, 
    #                                         x.sd_revised.min(), 
    #                                         how_many))
    
    closest_sdevs = df_ch.groupby(['symbol', 'dte'])[['strike_sdev', 'sd_revised']]\
        .apply(lambda x: get_closest_values(x.strike_sdev, 
                                            x.sd_revised.min(), 
                                            how_many))

    closest_sdevs.name = 'sdev1' 

    # Join the closest chains to the closest revised SD
    df_ch1 = df_ch.set_index(['symbol', 'dte']).join(closest_sdevs)

    # Get the target chains
    df_ch2 = df_ch1[df_ch1.apply(lambda x: x.strike_sdev in x.sdev1, axis=1)] \
                        .reset_index()
    
    return df_ch2


async def make_qualified_opts(port:int, 
                    df_chains: pd.DataFrame, 
                    MARKET: str,
                    STDMULT: int,
                    how_many: int,
                    DTESTDEVLOW: float,
                    DTESTDEVHI: float,
                    DECAYRATE:float,
                    CID: int=0,
                    desc: str='Qualifying Options'
                    ) -> pd.DataFrame:
    
    """[async] Make naked puts from chains, based on PUTSTDMULT"""

    MARKET_IS_OPEN = await isMarketOpen(MARKET)

    df_ch2 = target_options_with_adjusted_sdev(df_chains = df_chains, 
                                               STDMULT = STDMULT,
                                               how_many = how_many,
                                               DTESTDEVLOW = DTESTDEVLOW,
                                               DTESTDEVHI = DTESTDEVHI,
                                               DECAYRATE=DECAYRATE,
                                               MARKET_IS_OPEN=MARKET_IS_OPEN,
                                               )

    df_target = df_ch2[['symbol', 'strike', 'expiry', 'right',]].reset_index()

    options_list = make_target_option_contracts(df_target, MARKET=MARKET)

    # qualify target options
    with await IB().connectAsync(port=port, clientId=CID) as ib:
        options = await qualify_me(ib, options_list, desc=desc)

    # generate target puts
    df_puts = util.df(options).iloc[:, 2:6].\
        rename(columns={"lastTradeDateOrContractMonth": "expiry"}).\
            assign(contract=options)
    
    cols = df_puts.columns[:-1].to_list()

    # weed out other chain options
    df_output = df_chains.set_index(cols).join(df_puts.set_index(cols), on=cols)
    df_out = df_output.dropna().drop('localSymbol', axis=1).reset_index()

    return df_out


# * HANDLING HISTORY
# ==================

async def get_ohlc_bars(ib: IB,
               c,
               WHAT_TO_SHOW: str='TRADES',
               DURATION: int=365,
               BAR_SIZE = "1 day",
               ) -> list:
    """[async] Get Historical OHLC bars from IB"""

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


# * HANDLING STANDARD DEVIATIONS
# ==============================

def get_a_stdev(iv: float, price: float, dte: float) -> float:

    """Gives 1 Standard Deviation value.\n
    Assumes iv as annual implied volatility"""

    return iv*price*math.sqrt(dte/365)


def compute_strike_sd_right(df: pd.DataFrame) -> pd.DataFrame:
    """Computes strike's standard deviation and right for option chains
    
    NOTE:
    ----
    Function needs `iv`, `undprice` and `dte` columns in df"""

    # remove dtes <= 0 to prevent math failure
    df = df[df.dte > 0].reset_index(drop=True)

    # compute strike's standard deviation
    df = df.assign(sigma=df[['iv', 'undPrice', 'dte']].\
                    apply(lambda x: get_a_stdev(x.iv, x.undPrice, x.dte), axis=1))

    df = df.assign(strike_sdev = (df.strike - df.undPrice) / df.sigma)

    # determine the right
    df = df.assign(right = df.strike_sdev.apply(lambda strike_sdev: 'P' if strike_sdev < 0 else 'C'))

    return df


def sdev_for_dte(dte: float,
                 DTESTDEVLOW: float, DTESTDEVHI: float, 
                 DECAYRATE: float=0.1,
                 GAPBUMP: float=0):
    """
    Calculates the standard deviation (sdev) for a given dte from a curve.

    Args:
        dte: The time value (days to end) for which to calculate the sdev.
        DECAYRATE: The decay rate (default: 0.1).
        DTESTDEVLOW: The lower limit of the sdev range (default: 0.5).
        DTESTDEVHI: The upper limit of the sdev range (default: 1.7).
        GAPBUMP: Upward penalty for sdev if market is not open to avoid gap ups / downs.

    Returns:
        The calculated standard deviation value.
    """

    y_range = DTESTDEVHI - DTESTDEVLOW

    result = DTESTDEVLOW + y_range * np.exp(-DECAYRATE * dte)
    
    result = result + GAPBUMP

    return result


# * HANDLING OPTION CHAINS
# ========================

async def get_an_option_chain(ib: IB, contract:Contract, MARKET: str):
    """[async] Get Option Chains from IB"""

    chain = await ib.reqSecDefOptParamsAsync(
    underlyingSymbol=contract.symbol,
    futFopExchange="",
    underlyingSecType=contract.secType,
    underlyingConId=contract.conId,
    )

    # print(f"chain for {contract.symbol} is {chain}") # !!! TEMP

    if chain:
        chain = chain[-1] if isinstance(chain, list) else chain
        df = chain_to_df(chain, contract, MARKET)
    else:
        df = pd.DataFrame([])

    return df


def chain_to_df(chain, contract:Contract, MARKET: str) -> pd.DataFrame:
    """Convert an option chain to df"""

    df = pd.DataFrame([{'symbol': contract.symbol, 
                        'undId': contract.conId, 'key': 1}])

    df1 = pd.DataFrame([chain])

    dtes = [get_dte(x, MARKET) for x in df1.expirations[0]]
    df_expiries = pd.DataFrame(df1.expirations[0], columns=["expiry"]).assign(dte = dtes)

    # Do a cartesian merge for strikes and expiries
    df2 = (pd.merge(df_expiries.assign(key=1),
        pd.DataFrame(df1.strikes[0], columns=["strike"]).assign(key=1),
        on="key"))
    
    df3 = pd.merge(df, df2, on='key').drop('key', axis=1)

    # remove dtes <=0 from chains
    df4 = df3[df3.dte > 0]

    df_out = df4.assign(multiplier = contract.multiplier)

    return df_out


async def combine_a_chain_with_stdev(ib: IB, contract, MARKET: str=None, sleep: float=3, ) -> pd.DataFrame:
    """[async] makes a dataframe from chain and stdev for a symbol"""

    price_iv = await get_a_price_iv(ib, contract, sleep)
    _, undPrice, iv = price_iv

    if pd.isnull(undPrice):
        ticker = asyncio.run(get_prices_as_tickers(ib, contract))
        undPrice = next(iter(ticker)).marketPrice()
        if math.isnan(undPrice):
            undPrice = next(iter(ticker)).close

    chain = await get_an_option_chain(ib, contract, MARKET)
    
    if not chain.empty:
        df = chain.assign(localSymbol=contract.localSymbol, undPrice=undPrice, iv=iv, multiplier=contract.multiplier)
        df = compute_strike_sd_right(df=df)
    else:
        df = pd.DataFrame([])

    return df


async def make_chains(contracts: list,
                      port: int = None,
                      MARKET: str=None,
                      chunk_size: int=25, 
                      sleep: float=7,
                      CID: int=0,
                      ) -> pd.DataFrame:
    """[async] Makes chains for a list of contracts. 2m 11s for 186 contracts.\n
    ...for optimal off-market, chunk-size of 25 and sleep of 7 seconds."""

    contracts = to_list(contracts)

    # Get market and PORT for a single contract
    if not MARKET:
        MARKET = get_market_name(util.df(to_list(contracts)))

    _vars = Vars(MARKET)
    PORT = port = _vars.PORT

    with await IB().connectAsync(port=port, clientId=CID) as ib:

        chunks = tqdm(chunk_me(contracts, chunk_size), desc=f"Getting {MARKET} chains")
        dfs = []

        for cts in chunks:

            tasks = [asyncio.create_task(combine_a_chain_with_stdev(ib=ib, contract=c, MARKET=MARKET, sleep=sleep)) for c in cts]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # clean_res = [r for r in results if not r.empty]

            df = pd.concat(results, ignore_index=True)

            dfs.append(df)

        df_chains = pd.concat(dfs, ignore_index=True)

        # Add multiplier
        if MARKET == 'SNP':
            MULTIPLIER = 100
        else:
            MULTIPLIER = 1

        lots_dict = {c.conId: get_lots(c) for c in contracts}
        LOTS = df_chains.undId.map(lots_dict)

        df_chains = df_chains.assign(multiplier = MULTIPLIER, 
                                     market = MARKET,
                                     lot=LOTS/MULTIPLIER)
        
    return df_chains


def update_chains_dte(df: pd.DataFrame, MARKET: str):
    """
    Updates dte of chains
    df: df_chains dataframe
    MARKET: `SNP` | `NSE` """

    # ...check if chains are indexed
    if df.index.names[0] == None:
        df_ch = df.set_index(['symbol', 'expiry'])
    else:
        df = df_ch

    # Make non-duplicated chain of dtes

    df_ch = df_ch[~df_ch.index.duplicated(keep='last')].dte
    new_dte = df_ch.index.map(lambda x: get_dte(x[1], MARKET))
    df_ch = df_ch.reset_index().assign(dte = new_dte)

    # correct the chains by merging
    df = pd.merge(df, df_ch, on=['symbol', 'expiry'], suffixes=('', '_y'))
    df = df.assign(dte = df.dte_y).drop(columns=['dte_y'])

    return df


def get_strike_closest_to_und(df_chains: pd.DataFrame, 
                              how_many: int= -1) -> pd.DataFrame:
    
    """
    Gets option contracts closest to strike for every expiry\n
    Useful to get reference margins.
    int: -1 for closest Put
    """

    df1 = df_chains.loc[df_chains.groupby(['symbol', 'strike', 'right']).dte.idxmin()]\
                            .reset_index(drop=True)

    strk_near_und = df1[['symbol', 'strike', 'undPrice']].groupby('symbol')\
                                    .apply(lambda x: get_closest_values(x.strike, 
                                                                        x.undPrice.min(), 
                                                                        how_many=how_many))
    strk_near_und.name = 'strk_near_und'
    df_ch1 = df_chains.set_index(['symbol', 'dte']).join(strk_near_und)
    df_ch = df_ch1[df_ch1.apply(lambda x: x.strike in x.strk_near_und, axis=1)] \
                            .reset_index()

    # # for SNP limit to lowest dte
    # if get_exchange_name(df_chains) == 'SNP':
    #     df_ch = df_ch.loc[df_ch.groupby('symbol').dte.idxmin()].reset_index(drop=True)
    
    return df_ch


def create_target_opts(market: str) -> pd.DataFrame:

    """Final naked target options with expected price
    
    NOTE:
    ---
    MINEXPROM, MINOPOTSELLINGPRICE and MAXNAKEDORDERS comes from config / var.yml"""
    
    MARKET = market.upper()
    DATAPATH = ROOT / 'data' / MARKET

    _vars = Vars(MARKET)

    MINEXPROM = _vars.MINEXPROM
    PREC = _vars.PREC
    MINOPTSELLPRICE = _vars.MINOPTSELLPRICE
    MAXNAKEDORDERS = _vars.MAXNAKEDORDERS      

    df_opt_prices = get_pickle(DATAPATH / 'df_opt_prices.pkl')
    df_opt_margins = get_pickle(DATAPATH / 'df_opt_margins.pkl')
            

    cols = [x for x in list(df_opt_margins) if x not in list(df_opt_prices)]
    df_naked_targets = pd.concat([df_opt_prices, df_opt_margins[cols]], axis=1)

    # remove NaN's with a commission value
    df_naked_targets.comm.fillna(df_naked_targets.comm.dropna().unique()[0], inplace=True)

    # add DTE
    df_naked_targets = df_naked_targets.assign(dte = df_naked_targets.expiry.apply(lambda x: get_dte(x, MARKET)))
    # df_naked_targets.insert(4, 'dte', df_naked_targets.expiry.apply(lambda x: get_dte(x, MARKET)))

    # make multiplier an integer
    df_naked_targets.multiplier = df_naked_targets.multiplier.astype('int')

    # Get precise expected prices
    xp = ((MINEXPROM*df_naked_targets.dte/365*df_naked_targets.margin) +
            df_naked_targets.comm)/df_naked_targets.lot/df_naked_targets.multiplier

    # Set the minimum option selling price
    xp[xp < MINOPTSELLPRICE] = MINOPTSELLPRICE

    # Make the expected Price
    expPrice = pd.concat([xp, 
                            df_naked_targets.optPrice], axis=1)\
                            .max(axis=1)
    expPrice = expPrice.apply(lambda x: get_prec(x, PREC))
    df_naked_targets = df_naked_targets.assign(expPrice=expPrice)

    # clean the nakeds
    price_not_null = ~df_naked_targets.expPrice.isnull()
    price_greater_than_zero = df_naked_targets.expPrice > 0

    df = df_naked_targets[price_not_null & price_greater_than_zero].reset_index(drop=True)

    # bump option price for those with expPrice = optPrice
    opt_price_needs_bump = df.expPrice <= df.optPrice
    new_opt_price = df[opt_price_needs_bump].expPrice + PREC
    df.loc[opt_price_needs_bump, 'expPrice'] = new_opt_price

    # re-establish the rom after bump
    rom=(df.expPrice*df.lot*df.multiplier-df.comm)/df.margin*365/df.dte

    df = df.assign(prop = df.sdev.apply(get_prob),
                    rom=rom)    

    # weed out ROMs which are infinity or not upto expectations
    df = df[df.rom >= _vars.MINEXPROM]
    df = df[df.rom != np.inf] # Eliminates `zero` margins

    # restrict number of orders per symbol and right
    df = df.sort_values(['symbol', 'rom'], ascending=[True, False]).groupby(['symbol', 'right']).head(MAXNAKEDORDERS)

    df = df.reset_index(drop=True)

    return df


# * GETTING PRICES
# ================

async def get_mkt_prices(port: int, 
                         contracts: list, 
                         chunk_size: int=44, 
                         sleep: int=7) -> pd.DataFrame:
    
    """[async] A faster way to get market prices.
    """

    contracts = to_list(contracts)
    chunks = tqdm(chunk_me(contracts, chunk_size), desc="Mkt prices with IVs")
    results = dict()
    with await IB().connectAsync(port=port) as ib:
        for cts in chunks:
            tasks = [asyncio.create_task(get_a_price_iv(ib, c, sleep)) for c in cts]
            res = await asyncio.gather(*tasks)

            for r in res:
                symbol, price, iv = r
                results[symbol] = (price, iv)

    df_prices = split_symbol_price_iv(results)
    df_prices = pd.merge(clean_ib_util_df(contracts).iloc[:, :6], df_prices, on='symbol')

    # remove unnecessary columns (for secType == `STK`)
    keep_cols = ~((df_prices == 0).all() | \
              (df_prices == "").all() | \
                df_prices.isnull().all())
    
    df_prices = df_prices.loc[:, keep_cols[keep_cols == True].index]

    return df_prices


def split_symbol_price_iv(prices_dict: dict) -> pd.DataFrame:

    """Splits symbol, prices and ivs into a df.

    To be used after get_mkt_prices()"""

    symbols = []
    prices = []
    ivs = []

    for symbol, v in prices_dict.items():
        price, iv = v
        
        symbols.append(symbol)
        prices.append(price)
        ivs.append(iv)
        
    df_prices = pd.DataFrame({'symbol': symbols, 'price': prices, 'iv': ivs })

    return df_prices


async def get_market_data(ib: IB, 
                          c:Contract,
                          sleep:float = 2):

    """
    [async] Get marketPrice including implied volatility\n   
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
    [async] Gets tick-by-tick data\n  
    Quick when market is open \n   
    Takes ~6 secs after market hours. \n  
    No impliedVolatility"""

    ticker = await ib.reqTickersAsync(c)
    await asyncio.sleep(delay)

    return ticker


async def get_a_price_iv(ib, contract, sleep: float=2) -> dict:

    """[async] Computes price and IV of a contract.

    OUTPUT: dict{localsymbol, price, iv}
    
    Could take upto 12 seconds in case live prices are not available"""

    mkt_data = await get_market_data(ib, contract, sleep)

    if math.isnan(mkt_data.marketPrice()):

        if math.isnan(mkt_data.close):
            tick_data = await get_tick_data(ib, contract)
            tick_data = tick_data[0]

            if math.isnan(tick_data.marketPrice()):
                undPrice = tick_data.close
                if math.isnan(undPrice):
                    logger.info(f"No price found for {contract.localSymbol}!")
            else:
                undPrice = tick_data.marketPrice()
        else:
            undPrice = mkt_data.close

    else:
        undPrice = mkt_data.marketPrice()

    iv = mkt_data.impliedVolatility

    price_iv = (contract.localSymbol, undPrice, iv)

    return price_iv


async def get_prices_as_tickers(ib:IB,
                         contracts: list, 
                         chunk_size: int=44,
                         desc: str = "Getting contract prices:"):
    """[async] Gets option prices"""

    contracts = to_list(contracts)

    results = []

    pbar = tqdm(total=len(contracts),
                desc = desc,
                bar_format = BAR_FORMAT,
                ncols=80,
                leave=True,
            )

    chunks = chunk_me(contracts, chunk_size)

    for cts in chunks:

        tasks = [asyncio.create_task(get_tick_data(ib, contract), name = contract.localSymbol) 
                for contract in cts]
    
        ticks = await asyncio.gather(*tasks)

        results.append(ticks)
        pbar.update(len(cts))

    flat_results =list(flatten(results))
    pbar.refresh()

    return flat_results


async def get_prices_with_ivs(input_contracts: Union[pd.DataFrame, list],
                            port: int = None,
                            HIGHEST: bool = True,
                            desc: str = "Getting ticker prices") -> pd.DataFrame:
    """
    [async] gets contract prices and IVs from tickers
    * More detailed than get_mkt_prices with models, but misses IVs in closed market"""

    try:
        contracts = input_contracts.contract.to_list()
        df_cts = input_contracts # defaults to the input df to save columns like sdev
    except AttributeError:
        contracts = to_list(input_contracts)
        df_cts = clean_ib_util_df(contracts)

    # Gets the port based on contract's market
    if not port:
        MARKET = get_market_name(util.df(to_list(contracts)))
        _vars = Vars(MARKET)
        PORT = port = _vars.PORT
    
    with await IB().connectAsync(port=port) as ib:

        ct_prices = await get_prices_as_tickers(ib, contracts, desc=desc)

    df_prices = util.df(ct_prices)
    cols = ['contract', 'time', 'bidSize', 'askSize', 'lastSize', 
    'volume', 'high', 'low', 'bid', 'ask', 'last', 'close']

    df_ps = df_prices[cols]
    
    df = df_cts.merge(df_ps)

    greek_cols = [ 'undPrice', 'impliedVol', 'delta', 'gamma',
    'vega', 'theta', 'optPrice']

    try:
        model_df= util.df(list(util.df(ct_prices).modelGreeks.values))[greek_cols]
    except TypeError:
        pass # ignore adding Greeks
    else:
        ask_df = util.df(list(util.df(ct_prices).askGreeks.values))[greek_cols]
        bid_df = util.df(list(util.df(ct_prices).bidGreeks.values))[greek_cols]

        df = df.join(model_df, lsuffix='Model').\
                    join(ask_df, lsuffix='Ask').\
                        join(bid_df, lsuffix='Bid')
    
    # add secType
    secType = [c.secType for c in df.contract]
    df = df.assign(secType = secType)
    
    # puts optPrice / price column, if not in df
    df = arrange_prices(df, HIGHEST)

    return df


def arrange_prices(df: pd.DataFrame, 
                        HIGHEST: bool,
                        *args) -> pd.DataFrame:
    
    """
    Used to compute min or max of option prices provided for `get_prices_with_ivs()`.

    Arguments:
    --
    df: Input datframe. Should contain fields specified in args
    HIGHEST: True gives max values, else min values in OptPrice
    *args: should contain float-type price columns. Else bid, ask, last, close defaulted.

    Usage:
    -- 
    recompute_opt_price(df, True, 'bid', 'close') to get max of bid and close
    """
    if args:
        price_cols = list(args)
    else:
        price_cols = ['bid', 'ask', 'last', 'close']
    
    # cols = list(df.columns[:6]) + price_cols

    # remove -1 from price columns
    # df_n = df[cols]
    df_n1 = df[price_cols].replace(-1, np.nan)

    # replace bid and ask with average
    try:
        df_n2 = df_n1.assign(avgPrice=(df_n1.ask - df_n1.bid)/2).drop(columns=['bid', 'ask'])
    except AttributeError: # bid or ask is not present in the list
        df_n2 = df_n1

    if HIGHEST:
        optPrice = df_n2.assign(optPrice = df_n2.max(axis=1)).optPrice
    else:
        optPrice = df_n2.assign(optPrice = df_n2.min(axis=1)).optPrice

    # replace optPrice of df with cleansed optPrice
    df = df.assign(optPrice = optPrice)

    # for stock contracts replace optprice to price
    if 'OPT' not in df.secType.unique():
        df.rename(columns={'optPrice': 'price'}, inplace=True)

    return df


# * GETTING MARGINS AND COMMISSIONS
# =================================

def clean_a_margin(wif, conId):
    """Clean up wif margins"""

    d = dict()

    df = util.df([wif])[["initMarginChange", "maxCommission",
                                "commission"]].astype('float')

    df = df.assign(
        comm=df[["commission", "maxCommission"]].min(axis=1),
        margin=df.initMarginChange,
        conId = conId
    )

    # Correct unrealistic margin and commission
    df = df.assign(conId=conId,
        margin=np.where(df.initMarginChange > 1e7, np.nan, df.initMarginChange),
        comm=np.where(df.comm > 1e7, np.nan, df.comm))

    d[conId] = df[['margin', 'comm']].iloc[0].to_dict()

    return d


async def get_a_margin(ib: IB, 
                       contract,
                       order: Union[MarketOrder, None]=None,
                       ACTION: str='SELL',
                       ) -> dict:
    
    """
    [async] Gets a margin\n
    Gives negative margin for `BUY` trades"""

    lot = get_lots(contract)

    if not order: # Uses ACTION instead of order
        order = MarketOrder(ACTION, lot)

    def onError(reqId, errorCode, errorString, contract):
        logger.error(f"{contract.localSymbol} with reqId: {reqId} has errorCode: {errorCode} error: {errorString}")

    ib.errorEvent += onError
    wif = await ib.whatIfOrderAsync(contract, order)
    ib.errorEvent -= onError
    logger.remove()

    # to deal with option contracts with timezone errors
    if int(contract.conId) == 0:
        cid = contract.symbol + contract.lastTradeDateOrContractMonth + str(contract.strike) + contract.right
    else:
        cid = contract.conId

    try:
        output = clean_a_margin(wif, cid)

    except KeyError:
        output = { cid: {
                  'margin': None,
                  'comm': None}}
        
        logger.error(f"{contract.localSymbol} has no margin and commission")

    lot = order.totalQuantity
    # output[cid]['lot_size'] = lot  #!!! non-standard field name
    output[cid]['lot'] = lot

    return output


async def get_margins(contracts: Union[pd.DataFrame, pd.Series, list, Contract],
                      port: int=None, 
                      orders: Union[pd.Series, list, MarketOrder, None]=None,
                      ACTION: str='SELL',
                      desc: str='Getting margins:',
                      chunk_size: int=100) -> pd.DataFrame:
    """

    [async] Gets margins for options contracts with `orders` or `ACTION`

    Parameters
    ---
    contracts: df with `contract` field | list
    order: list of `MarketOrder` | `None` requires an `ACTION`
    ACTION: `BUY` or `SELL` needed if no `orders` are provided. Defaults to `SELL`
    """
    
    if isinstance(contracts, pd.DataFrame):
        opt_contracts = to_list(contracts.contract)
    else:
        opt_contracts = to_list(contracts)

    # Gets the port based on contract's market
    if not port:
        MARKET = get_market_name(util.df(to_list(contracts)))
        _vars = Vars(MARKET)
        PORT = port = _vars.PORT

    pbar = tqdm(total=len(opt_contracts),
                    desc=desc,
                    bar_format = BAR_FORMAT,
                    ncols=80,
                    leave=True,
                )
    
    # prepare orders
    if orders:
        orders = to_list(orders)

        if len(orders) == 1: # single order
            orders = orders*len(opt_contracts)

    else:
        orders = [None]*len(opt_contracts)
    
    results = list()

    df_contracts = clean_ib_util_df(opt_contracts)

    # to take care of expiry date discrepancy between Chicago options
    conId = [c.conId if int(c.conId) > 0 
            else c.symbol + c.lastTradeDateOrContractMonth + str(c.strike) + c.right 
            for c in opt_contracts]

    df_contracts = df_contracts.assign(conId=conId, order=orders)\
                                .set_index('conId')

    cos = list(zip(df_contracts.contract, df_contracts.order))

    chunks = chunk_me(cos, chunk_size)

    with await IB().connectAsync(port=port) as ib:
        
        for cts in chunks:

            tasks = [asyncio.create_task(get_a_margin(ib=ib, 
                                                        contract=contract,
                                                        order=order,
                                                        ACTION=ACTION,), 
                                                        name= contract.localSymbol) 
                        for contract, order in cts]        


            margin = await asyncio.gather(*tasks)

            results += margin
            pbar.update(len(cts))
            pbar.refresh()

    flat_results ={k: v for r in results for k, v in r.items()}
    df_mgncomm = pd.DataFrame(flat_results).T
    df_out = df_contracts.join(df_mgncomm).reset_index()

    df_margins = df_out.assign(conId=pd.to_numeric(df_out.conId, errors='coerce'))

    if np.issubdtype(df_margins.expiry, np.datetime64):
        df_margins = df_margins.assign(expiry=df_margins.expiry.dt.strftime('%Y%m%d'))

    pbar.close()

    return df_margins


def opt_margins_with_lot_check(df: pd.DataFrame, 
                           multiply_lot: bool=True) -> pd.DataFrame:
    """Gets margins and commissions **only** for option chains with `MARKET` field.
    df_ch: option df
    multiply_lot: mutiplies lot with multiplier and then divides to get margin for 1 lot

    Note:
    ----
    For SNP run this function twice. Once with `multiply_lot` as True and then for remaining with it as False
    """

    # try:
    #     MARKET = df.iloc[0].exchange
    # except AttributeError:
    #     logging.error(f"MARKET is unknown for {df.head(1)}")
    #     return None

    # if MARKET != 'NSE':
    #     MARKET = 'SNP'

    MARKET = get_market_name(df)
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    EXCHANGE = get_exchange_name(df)

    opt_contracts = [make_a_raw_contract(symbol=symbol, EXCHANGE=EXCHANGE, secType='OPT', strike=strike, right=right, expiry=expiry)
    for symbol, strike, right, expiry in zip(df.symbol, df.strike, df.right, df.expiry)]

    if multiply_lot:
        orders = [MarketOrder(action='SELL', totalQuantity=qty) for qty in df.lot*df.multiplier]
        df_margins = asyncio.run(get_margins(port=PORT, contracts=opt_contracts, orders=orders, desc=f'{MARKET} margins first run:'))
        df_margins.margin = df_margins.margin / df_margins.lot
        df_margins.comm = df_margins.comm / df_margins.lot

    else: # orders without multiplier
        orders = [MarketOrder(action='SELL', totalQuantity=qty) for qty in df.lot]
        df_margins = asyncio.run(get_margins(port=PORT, contracts=opt_contracts, orders=orders,desc=f'{MARKET} margins second run:'))

    # ensure comm for margin is available
    max_margin = df_margins.comm.max()
    cond = ~df_margins.margin.isnull() & df_margins.comm.isnull()
    df_margins.loc[cond, 'comm'] = max_margin

    df_margins.drop(columns=['contract', 'secType', 'conId', 'multiplier', 'lot'], inplace=True)

    return df_margins


# * NSE SPECIFIC FUNCTIONS
# ========================

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                        'like Gecko) '
                        'Chrome/80.0.3987.149 Safari/537.36',
    'accept-language': 'en,gu;q=0.9,hi;q=0.8',
    'accept-encoding': 'gzip, deflate, br'}


def get_nse_payload(url: str) -> requests.models.Response:

    """
    Returns response

    url: (samples)
    * https://archives.nseindia.com/content/fo/fo_mktlots.csv for lots
    * https://nsearchives.nseindia.com/content/nsccl/C_CATG.T01 for margin groups
    * https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O for fno equity list

    Response processing examples:
    ---
    `df = pd.read_csv(io.StringIO(response.text))` for csv type response output 

    """

    base_url = 'https://www.nseindia.com'
    session = requests.Session()

    # r = session.get(base_url, headers=headers, timeout=5)
    # response = session.get(url, timeout=5, headers=headers)
    response=requests.get(url, headers=headers)

    return response


def nse_ban_list() -> list:
    """Gets scrips banned today
    """

    URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
    response = get_nse_payload(URL)

    df = pd.read_csv(io.StringIO(response.text))
    ban_list = df.iloc[:,0].tolist()

    return ban_list


def get_lots_from_nse() -> dict:
    """
    Gets lots from nse site without any symbol cleansing
    """

    MKT_LOTS_URL = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'

    response = get_nse_payload(MKT_LOTS_URL).text

    res_dict = {} # unclean symbol results dictionary

    for line in response.split('\n'):
        if line != '' and re.search(',', line) and (line.casefold().find('symbol') == -1):
            (code, name) = [x.strip() for x in line.split(',')[1:3]]
            res_dict[code] = int(name)
    
    return res_dict


def get_nse_native_fno_list() -> list:
    """Gets a dictionary of native nse symbols from nse.com"""

    # MKT_LOTS_URL = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'

    # response = get_nse_payload(MKT_LOTS_URL).text

    # res_dict = {} # unclean symbol results dictionary

    # for line in response.split('\n'):
    #     if line != '' and re.search(',', line) and (line.casefold().find('symbol') == -1):
    #         (code, name) = [x.strip() for x in line.split(',')[1:3]]
    #         res_dict[code] = int(name)

    res_dict = get_lots_from_nse()

    nselist = [k for k, _ in res_dict.items()]

    return nselist


def nse2ib(nselist: list, as_dict: bool = False) -> Union[list, dict]:
    """Convert NSE symbols to IB friendly ones"""

    path_to_yaml_file = ROOT / 'data' / 'master' / 'nse2ibkr.yml'

    # get substitutions from YAML file
    with open(path_to_yaml_file, 'r') as f:
        subs = yaml.load(f, Loader=yaml.FullLoader)
    
    list_without_percent_sign = list(map(subs.get, nselist, nselist))

    # fix length to 9 characters
    ib_fnos = [s[:9] for s in list_without_percent_sign]

    if as_dict:
        ib_fnos = dict(zip(nselist, ib_fnos))

    return ib_fnos


def make_nse_lots(save: bool=True) -> dict:

    """Makes lots for NSE on cleansed IB symbols without BLACKLIST"""

    MARKET = 'NSE'
    DATAPATH = ROOT / 'data' / MARKET.lower()

    res_dict = get_lots_from_nse()
    nselist = list(res_dict.keys())

    path_to_yaml_file = ROOT / 'data' / 'master' / 'nse2ibkr.yml'

    # get substitutions from YAML file
    with open(path_to_yaml_file, 'r') as f:
        subs = yaml.load(f, Loader=yaml.FullLoader)

    list_without_percent_sign = list(map(subs.get, nselist, nselist))

    # fix length to 9 characters
    nse_symbols = [s[:9] for s in list_without_percent_sign]

    # make a dictionary to map nse symbols to ib friendly symbols
    nse2ibsymdict = dict(zip(nselist, nse_symbols))

    # correct the nse symbols
    result = {nse2ibsymdict[k]: v for k, v in res_dict.items()}

    # remove BLACKLIST
    BLACKLIST = Vars(MARKET).BLACKLIST

    clean_lots = {k: v for k, v 
                    in result.items() 
                    if k not in BLACKLIST} 

    ## To prevent missing NIFTY50, which sometimes gets missed!
    # if 'NIFTY50' not in clean_lots.keys():
    #     clean_lots['NIFTY50'] = 50

    if save:
        pickle_me(clean_lots, DATAPATH / 'lots.pkl')

    return clean_lots


def fnolist():

    """Generates a list of fnos"""
    
    fnos = list(make_nse_lots().keys())

    return fnos


def make_unqualified_nse_underlyings(symbols: list) -> list:
    """Makes raw underlying contracts for NSE"""

    contracts = [Index(symbol, 'NSE', 'INR') 
           if 'NIFTY' in symbol 
                else Stock(symbol, 'NSE', 'INR') 
            for symbol in symbols]
    
    return contracts


async def assemble_nse_underlyings(PORT: int) -> dict:
    """[async] Assembles a dictionary of NSE underlying contracts"""

    # get FNO list
    fnos = fnolist()

    # ignore these...
    BLACKLIST = Vars('NSE').BLACKLIST
    BANLIST = nse_ban_list()

    # remove blacklisted symbols - like NIFTYIT that doesn't have options
    nselist = [n for n in fnos if n not in set(BLACKLIST+BANLIST)]

    # clean to get IB FNOs
    # nse2ib_yml_path = ROOT / 'data' / 'master' / 'nse2ibkr.yml'
    ib_fnos = nse2ib(nselist)

    # make raw underlying fnos
    raw_nse_contracts = make_unqualified_nse_underlyings(ib_fnos)
    
    with await IB().connectAsync(port=PORT) as ib:

        # qualify underlyings
        qualified_unds = await qualify_me(ib, raw_nse_contracts, desc='Qualifying NSE Unds')

    unds_dict = make_dict_of_qualified_contracts(qualified_unds)

    return unds_dict


# * SNP SPECIFIC FUNCTION
# =====================

def read_weeklys() -> pd.DataFrame:
    """gets weekly cboe symbols"""

    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
    df = pd.read_html(dls)[0]

    return df


def rename_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """standardizes column names of cboe"""
    
    df.columns=['desc', 'symbol']

    return df


def remove_non_char_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """removes symbols with non-chars - like dots (BRK.B)"""
    
    df = df[df.symbol.str.extract("([^a-zA-Z])").isna()[0]]

    return df


def make_weekly_cboes() -> pd.DataFrame:
    """
    Generates a weekly cboe symbols dataframe
    """

    df = (
        read_weeklys()
        .pipe(rename_weekly_columns)
        .pipe(remove_non_char_symbols)
        )
    
    # add exchange
    df = df.assign(exchange='SMART')
    
    return df


def get_snps() -> pd.Series:
    """
    gets snp symbols from wikipedia
    """
    snp_url =  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    snps = pd.read_html(snp_url)[0]['Symbol']
    return snps


def add_snp_indexes(df: pd.DataFrame, path_to_yaml_file: str) -> pd.DataFrame:
    """
    add indexes from config/snp_indexes.yaml
    """
    with open(path_to_yaml_file, 'r') as f:
        kv_pairs = yaml.load(f, Loader=yaml.FullLoader)

    dfs = []
    for k in kv_pairs.keys():
        dfs.append(pd.DataFrame(list(kv_pairs[k].items()), 
            columns=['symbol', 'desc'])
            .assign(exchange = k))
        
    more_df = pd.concat(dfs, ignore_index=True)

    df_all = pd.concat([df, more_df], ignore_index=True)
    
    return df_all


def split_snp_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """differentiates stocks and index"""
    
    df = df.assign(secType=np.where(df.desc.str.contains('Index'), 'IND', 'STK'))

    return df


def make_snp_weeklies(indexes_path: Path):
    """Makes snp weeklies with indexes"""

    # get snp stock weeklies
    df_weekly_cboes = make_weekly_cboes()
    snps = get_snps()

    # filter weekly snps
    df_weekly_snps = df_weekly_cboes[df_weekly_cboes.symbol.isin(snps)] \
                    .reset_index(drop=True)

    # add index weeklies
    df_weeklies = (
        add_snp_indexes(df_weekly_snps, indexes_path)
        .pipe(split_snp_stocks_and_index)
        )
    
    return df_weeklies


def make_unqualified_snp_underlyings(df: pd.DataFrame) -> pd.DataFrame:
    """Build underlying contracts"""

    contracts = [Stock(symbol=symbol, exchange=exchange, currency='USD') 
                if 
                    secType == 'STK'
                else 
                    Index(symbol=symbol, exchange=exchange, currency='USD') 
                for 
                    symbol, secType, exchange in zip(df.symbol, df.secType, df.exchange)]
    
    df = df.assign(contract = contracts)

    return df
    

async def assemble_snp_underlyings(PORT: int) -> dict:
    """[async] Assembles a dictionary of SNP underlying contracts"""

    CID = Vars('SNP').CID

    indexes_path = ROOT / 'data' / 'master' / 'snp_indexes.yml'
    
    df = make_snp_weeklies(indexes_path) \
         .pipe(make_unqualified_snp_underlyings)
    
    contracts = df.contract.to_list()

    with await IB().connectAsync(port=PORT, clientId=CID) as ib:
    
        qualified_contracts = await qualify_me(ib, contracts, desc="Qualifying SNP Unds")

    underlying_contracts = make_dict_of_qualified_contracts(qualified_contracts)

    return underlying_contracts


def compute_snp_df_margins(df = pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds margin to snp with null values\n
    df needs `margin`, `undPrice` and `multiplier` columns
    """

    # filter the nulls

    df_comp = df[df.margin.isnull()]

    # boolean masks
    thirty_pct_limit = 16.67
    thirty_pct_margin_cond = (df_comp.undPrice > thirty_pct_limit)
    five_per_share_cond = df_comp.undPrice.between(5, thirty_pct_limit, inclusive='right')
    hundred_pct_mkt_val_cond = df_comp.undPrice.between(2.5, 5, inclusive='right')
    hundred_pct_at_2_5_cond = df_comp.undPrice.between(0, 2.5, inclusive='right')

    # conditions
    condlist = [thirty_pct_margin_cond, five_per_share_cond, hundred_pct_mkt_val_cond, hundred_pct_at_2_5_cond]

    # values
    choicelist = [0.3*df_comp.undPrice*df_comp.multiplier, 
                0.3*5*df_comp.multiplier, 
                df_comp.undPrice*df_comp.multiplier, 
                2.5*df_comp.multiplier]

    # default
    default = np.nan

    s = df_comp.assign(computed_mgn = np.select(condlist, choicelist, default)).computed_mgn
    df.margin = df.margin.fillna(s)

    return df


# * SOWING NAKEDS
# ===============

def pickle_the_sow(df: pd.DataFrame, 
                   _vars: dict, 
                   MARKET_IS_OPEN: bool) -> dict:
    
    """
    Pickles sow with key parameters for analysis
    Outputs a dict(`df_sow`, `configed`, `market_open`)"""
                   
    # Determine the filename and path

    MARKET = 'NSE' if df.exchange.unique()[0] == 'NSE' else 'SNP'

    ROOT = from_root()
    _vars = Vars(MARKET)
    DATAPATH = ROOT / 'data' / MARKET.lower()

    dt = datetime.datetime.now().strftime("_%I_%M_%p_on_%d-%b-%Y")
    file_name = "".join([MARKET,'_sow', dt , '.pkl'])
    TEMP_SOW_PATH = DATAPATH.parent / 'raw' / file_name

    # Get config variables
    vars_d = _vars.__dict__
    # Filter relevant fields
    want_fields = ['GAPBUMP', 'MINDTE', 'MAXDTE', 'CALLSTDMULT', 'PUTSTDMULT', 
    'DTESTDEVLOW', 'DTESTDEVHI', 'DECAYRATE', 
    'MINEXPROM', 'MINOPTSELLPRICE']

    config_dict = {k: v for k, v in vars_d.items() if k in want_fields}

    # Make the object to pickle
    obj_to_pickle = {'df_sow': df, 
    'configed' : config_dict,
    'market_open': MARKET_IS_OPEN}

    # Pickle
    pickle_me(obj_to_pickle, TEMP_SOW_PATH)

    return obj_to_pickle


# * ORDER HANDLING
# ================

async def get_open_orders(ib) -> pd.DataFrame:
    """[async] Gets open orders in a df"""

    # ACTIVE_STATUS is common for nse and snp
    ACTIVE_STATUS = Vars('SNP').ACTIVE_STATUS 

    df_openords = OpenOrder().empty() #Initialize open orders

    await ib.reqAllOpenOrdersAsync()

    trades = ib.trades()

    if trades:

        all_trades_df = clean_ib_util_df([t.contract for t in trades]) \
            .join(util.df(t.orderStatus for t in trades)) \
            .join(util.df(t.order for t in trades), lsuffix="_")
        
        order = pd.Series([t.order for t in trades], name='order')

        all_trades_df = all_trades_df.assign(order=order)
        
        # all_trades_df = (util.df(t.contract for t in trades).join(
        #     util.df(t.orderStatus
        #             for t in trades)).join(util.df(t.order for t in trades),
        #                                     lsuffix="_"))

        all_trades_df.rename({"lastTradeDateOrContractMonth": "expiry"},
                                axis="columns",
                                inplace=True)
        
        trades_cols = df_openords.columns

        dfo = all_trades_df[trades_cols]
        dfo = dfo.assign(expiry=pd.to_datetime(dfo.expiry))
        df_openords = dfo[all_trades_df.status.isin(ACTIVE_STATUS)]

    return df_openords


async def get_order_pf(PORT):
    """[async] Gets the open orders and portfolios"""
    with await IB().connectAsync(port=PORT) as ib:
        df_openords = await get_open_orders(ib)
        df_pf = quick_pf(ib)

        return df_openords, df_pf
 

def place_orders(ib: IB, cos: Union[Tuple, List], blk_size: int = 25) -> List:
    """!@ CAUTION @!: This places orders in the system
    NOTE: cos could be a single (contract, order)
          or a tuple/list of ((c1, o1), (c2, o2)...)
          made using tuple(zip(cts, ords))"""

    trades = []

    if isinstance(cos, (tuple, list)) and (len(cos) == 2):
        c, o = cos
        trades.append(ib.placeOrder(c, o))

    else:
        cobs = [cos[i:i + blk_size] for i in range(0, len(cos), blk_size)]

        for b in tqdm(cobs):
            for c, o in b:
                td = ib.placeOrder(c, o)
                trades.append(td)
            asyncio.sleep(0.75)

    return trades


async def place_orders_async(MARKET:str, PAPER: bool, 
                            cos:list, blk_size: int=25):
    """[async] Places orders in the system"""

    _vars = Vars(MARKET)

    if PAPER:
        PORT = _vars.PAPER
    else:
        PORT = _vars.PORT

    with await IB().connectAsync(port=PORT) as ib:

        trades = []

        if isinstance(cos, (tuple, list)) and (len(cos) == 2):
            c, o = cos
            trades.append(ib.placeOrder(c, o))

        else:
            cobs = [cos[i:i + blk_size] for i in range(0, len(cos), blk_size)]

            for b in tqdm(cobs, desc=f"Placing {MARKET} orders"):
                for c, o in b:
                    td = ib.placeOrder(c, o)
                    trades.append(td)
                await asyncio.sleep(0.75)

        # logger.info(f"Placed {len(trades)} orders!")

    return trades


async def cancel_all_api_orders(MARKET):

    """
    [async] Cancels all API orders.
    Does not cancel TWS direct orders.

    PORT: TWS / IB port for cancellation
    """

    _vars = Vars(MARKET)
    PORT = _vars.PORT
    ACTIVE_STATUS = _vars.ACTIVE_STATUS
    blk_size = 25

    cancelled_orders = []

    with await IB().connectAsync(port=PORT) as ib:

        await ib.reqAllOpenOrdersAsync()

        trades = ib.trades()
        open_orders = [t.order for t in trades if t.orderStatus.status in ACTIVE_STATUS]

        if open_orders:

            cobs = [open_orders[i:i + blk_size] for i in range(0, len(open_orders), blk_size)]
            for order_blk in tqdm(cobs):
                for order in order_blk:
                    cancelled_ord = ib.cancelOrder(order)
                    cancelled_orders.append(cancelled_ord)
                await asyncio.sleep(0.75)
        
        else:
            logger.info("Nothing to cancel!")
            cancelled_orders = None

    return cancelled_orders


async def cancel_orders(PORT: int,
                        orders_to_cancel: list,
                        chunk_size: int=25, 
                        delay: float=0.75) -> list:
    """
    [async] Cancel given api orders in chunks
    orders_to_cancel: list of orders
    """

    chunks = tqdm(chunk_me(data=orders_to_cancel, size=chunk_size), desc="Cancelling non-portfolio orders")

    cancels = []

    with await IB().connectAsync(port=PORT) as ib:

        for chunk in chunks:
            for order in chunk:
                cancelled = ib.cancelOrder(order)
                cancels.append(cancelled)

            await asyncio.sleep(delay)

    logger.info(f"{len(orders_to_cancel)} open orders were cancelled")

    return cancels


# * PORTFOLIO HANDLING
# ====================

def quick_pf(ib) -> Union[None, pd.DataFrame]:
    """Gets the portfolio dataframe"""
    pf = ib.portfolio()  # returns an empty [] if there is nothing in the portfolio

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(columns=["account"]))
        df_pf = df_pf.rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry",
                "marketPrice": "mktPrice",
                "marketValue": "mktVal",
                "averageCost": "avgCost",
                "unrealizedPNL": "unPnL",
                "realizedPNL": "rePnL",
            })
    else:
        df_pf = Portfolio().empty()

    return df_pf


async def async_pf(PORT):
    """[asynch] version of quick_pf"""
    
    with await IB().connectAsync(port=PORT) as ib:
        df_pf = quick_pf(ib)

        return df_pf
    

def get_portfolio_with_margins(MARKET: str) -> pd.DataFrame:

    """Gets current portfolio with margins of them"""
    
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    # get unds, open orders and portfolio
    df_pf = asyncio.run(async_pf(PORT))

    desc = "Qualifiying Portfolio"
    pf_contracts = asyncio.run(qualify_conIds(PORT, df_pf.conId, desc=desc))

    # ...integrate df_pf with multiplier
    df1 = clean_ib_util_df(pf_contracts).set_index('conId')
    df2 = df_pf.set_index('conId')

    cols_to_use = df2.columns.difference(df1.columns)
    df_pf = df1.join(df2[cols_to_use])

    # join the multiplier
    s = pd.to_numeric(df_pf.multiplier)
    s.fillna(1, inplace=True)
    df_pf = df_pf.assign(multiplier=s)

    # Get DTEs
    df_pf.insert(4, 'dte', df_pf.expiry.apply(lambda x: get_dte(x, MARKET)))
    df_pf.loc[df_pf.dte <=0, "dte"] = 0

    # Get the costPrice
    df_pf.insert(9, 'costPrice', abs(df_pf.avgCost/df_pf.position))

    # Assign the actions
    df_pf = df_pf.assign(action=np.where(df_pf.position < 0, "BUY", "SELL"))

    # build the orders
    wif_order = [MarketOrder(action, totalQuantity) 
                for action, totalQuantity 
                in zip(df_pf.action, abs(df_pf.position).astype('int'))]
    df_pf = df_pf.assign(wif_order = wif_order)

    contracts = to_list(pf_contracts)
    orders = df_pf.wif_order.to_list()

    df_m = asyncio.run(get_margins(port=PORT, contracts=contracts, orders = orders))

    df_pfm = join_my_df_with_another(df_pf, df_m, 'conId').drop(columns = ['wif_order', 'action', 'costPrice', 'mktVal'])

    return df_pfm.drop(columns='order')


# * JOURNALING & REPORT EXTRACTION
# ================================

def get_sowed_pickles(market_paths: list) -> dict:
    """Assembles the pickles
    market_paths: list of paths for a particular market"""


    # get the dictionaries
    ds = [get_pickle(mp) for mp in market_paths]

    return ds


def add_utc_times(ds: list, market_paths: list) -> list:
    """Adds utc time"""

    # get the times of the files
    utc_times = [datetime.datetime.fromtimestamp(mp.stat().st_mtime, tz=datetime.timezone.utc) 
                for mp in market_paths]

    # add utc_times to ds (dictionaries)
    for i, d in enumerate(ds):
        d['utc_time'] = utc_times[i]

    return ds


def dicts2sorted_dfs(ds:list) -> list:
    """Assemble dfs from sowed pickles"""

    # ... extract the unique dictionary keys
    d_keys = {k for d in ds for k, _ in d.items()}

    result = []

    for d in ds:

        df_set = []

        for k in d_keys:

            obj = d[k]

            if isinstance(obj, pd.DataFrame):
                df_set.append(obj)
            elif isinstance(obj, dict):
                df_set.append(pd.DataFrame([obj]))
            elif isinstance(obj, Union[bool, datetime.datetime]):
                df_set.append(pd.DataFrame([{k:obj}]))
            else:
                logger.error(f"untreatable object type {type(obj)}")
        
        # gets df_sow first
        dfs_sorted = sorted(df_set, key=len, reverse=True)

        result.append(dfs_sorted)
    
    return result


def join_dfs_by_columns(dfs_collection: list) -> pd.DataFrame:
    """Joins a base df with single-row dfs
    dfs_collection: collection of sorted dfs, with base df as the first one."""

    dfs_collection = to_list(dfs_collection)
    
    all_dfs = []

    for r in dfs_collection:

        x = r[0] # the first df in the sorted df collection
        rows = len(x)

        dfs = [df for df in r[1:]] # remaining dfs

        for df in dfs:
            df = pd.concat([df] * rows, ignore_index=True)  # Repeat rows efficiently
            x = x.join(df)

        all_dfs.append(x)

    # dfs2concat = [d for d in all_dfs if not d.empty]

    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    return df


def get_archived_sows(sow_path: Path, MARKET: str) -> pd.DataFrame:

    """make dfs out of archved sows
    sow_path: path where the sows exist
    MARKET: 'NSE' | 'SNP' """

    # get list of sow paths for the market
    paths = glob.glob(str(sow_path / '*.pkl'))
    market_paths = [Path(p) for p in paths if Path(p).parts[-1][:3] == MARKET]

    ds = get_sowed_pickles(market_paths)
    d = add_utc_times(ds, market_paths)
    dfs = dicts2sorted_dfs(d)
    df = join_dfs_by_columns(dfs)
    return df


def local2symbols(syms: Union[pd.Series, list]) -> pd.DataFrame:
    """Converts to df with symbol, expiry, strike, right.

    NOTE:
    ---
     - This is used only for reports generated in IBKR portal"""
    
    # Needed to rectify space between BRK and B
    
    def correct_symbols_with_a_gap(text: str) -> str:
        """Corrects `BRK B`"""

        # new_text = copy.deepcopy(text)
        new_text = text

        t_split = text.split(' ')
        if len(t_split) > 1:
            if not t_split[1][-1].isdigit():
                new_text = ''.join(t_split[:2]) + ' ' + ' '.join(t_split[2:])

        return new_text
    
    slist = [correct_symbols_with_a_gap(s) for s in syms]

    s_series = pd.Series(slist, name='localSymbol')

    split_syms = s_series.str.split(' ').copy(deep=True)

    symbols = [s[0] if len(s) > 1 else s[0] for s in split_syms]
    expiries = [pd.to_datetime(s[1], dayfirst=True).date() if len(s) > 1 else pd.NaT for s in split_syms]
    strikes = [float(s[2]) if len(s) > 1 else np.nan for s in split_syms]
    rights = [s[3] if len(s) > 1 else '' for s in split_syms]

    df = pd.DataFrame({"symbol": symbols,
                       "expiry": expiries,
                       "strike": strikes,
                       "right": rights})
    
    return df


def clean_df_positions(df_posn: pd.DataFrame) -> pd.DataFrame:
    """Cleans up df_position dataframe

    NOTE:
    ---
     - This is used only for reports generated in IBKR portal
    
    """

    df_posn.insert(0, 'xnType', 'position')

    secType = np.where(df_posn['Asset Category'] == 'Stocks', 'STK', 'OPT')
    df_posn.insert(1, 'secType', secType)
    df_pos = df_posn.drop(columns = 
                        ['Open Positions', 'Header', 'DataDiscriminator', 'Unrealized P/L',
                                    'Mult', 'Asset Category', 'Cost Basis', 'Value'])

    pos_cols = {'Currency': 'currency',
    'Symbol': 'localSymbol',
    'Open': 'time',
    'Quantity': 'qty',
    'Cost Price': 'cost',
    'Close Price': 'price',
    'Code': 'code'
    }
    df_pos = df_pos.rename(columns=pos_cols)

    # standardize the time
    df_pos.time = pd.to_datetime(df_pos.time)

    # get multiplier for option quantities
    mult = np.where((df_pos.currency == 'USD') & (df_pos.secType == 'OPT') & (df_pos.qty.astype('float').abs() == 1), 100, 1)
    insert_mult_at = list(df_pos.columns).index('qty') + 1
    df_pos.insert(insert_mult_at, 'mult', mult)

    # generate trade pnl
    num_cols = ['qty', 'mult', 'price', 'cost']
    df_pos[num_cols] = df_pos[num_cols].apply(pd.to_numeric)

    df_pos = df_pos.assign(pnl= (df_pos.price - df_pos.cost)*df_pos.qty*df_pos.mult,
                           exchange= "--")

    # integrate
    df_pos = local2symbols(df_pos.localSymbol).join(df_pos)

    df_pos = df_pos.sort_values(['symbol', 'time'])

    return df_pos


def clean_df_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up trade entries in ibkr portal report    
    """

    df.insert(0, 'xnType', 'trade')
    secType = np.where(df.Symbol.str.len() > 9, 'OPT', 'STK')
    df.insert(1, 'secType', secType)

    trade_cols = {'Currency': 'currency',
        'Symbol': 'localSymbol',
        'Date/Time': 'time',
        'Date': 'time',
        'Exchange': 'exchange',
        'Quantity': 'qty',
        'C. Price': 'cost',
        'T. Price': 'price',
        'Comm/Fee': 'commission',
        'Code': 'code',
        'Type': 'code'
        }

    drop_cols = ['Trades', 'Header', 'DataDiscriminator',
                'Asset Category', 'Proceeds', 'Basis', 
                'Realized P/L', 'MTM P/L', 'Option Exercises, Assignments and Expirations',
                'Headers', 'Transaction Type']
            
    df = df.drop(columns=drop_cols, errors='ignore', axis=1)
    df.rename(columns=trade_cols, inplace=True)

    # Remove differeentiator line between `Stocks` and `Equity and Index Options`
    df = df[df.localSymbol != 'Symbol'].reset_index(drop=True)

    # standardize the time
    df.time = pd.to_datetime(df.time, format='mixed', yearfirst=True)

    # integrate
    df = local2symbols(df.localSymbol).join(df)

    # get multiplier for option quantities
    df.qty = pd.to_numeric(df.qty.str.replace(',', '')).astype('float')
    mult = np.where((df.currency == 'USD') & (df.secType == 'OPT') & (df.qty.astype('float').abs() == 1), 100, 1)
    insert_mult_at = list(df.columns).index('qty') + 1
    df.insert(insert_mult_at, 'mult', mult)

    # generate trade pnl
    num_cols = ['qty', 'mult', 'price', 'cost', 'commission']
    df[num_cols] = df[num_cols].apply(pd.to_numeric)

    df = df.assign(pnl= (df.price - df.cost)*df.qty.abs()*df.mult+df.commission)

    # organize the symbols
    df = df.sort_values(['symbol', 'time', 'secType'], ascending=[True, True, True])\
            .reset_index(drop=True)
    
    return df


def clean_df_dividends(df: pd.DataFrame) -> pd.DataFrame:

    """Cleans dividend file extracted from IBKR portal"""

    div_cols = {"Currency": "currency",
                "Date": "time",
                "Description": "desc",
                "Amount": "amt"}

    df = df.rename(columns=div_cols)

    # remove smallcase columns (not in div_cols)
    keep_div_cols = [c for c in df.columns if c.islower()]
    df = df[keep_div_cols]

    # insert symbols
    symbols = df.desc.str.split('(').str[0]
    df.insert(0, 'symbol', symbols)

    df = df.assign(dividend=df.desc.apply(floatxtract))
    df = df.assign(time = pd.to_datetime(df.time), 
                    amt = pd.to_numeric(df.amt))

    return df


def clean_an_ib_portal_report(market_path: Path) -> dict:
    """Cleans an IB portal extract
    market_path: pathlib.Path has the csv to be cleaned"""

    # put col nammes in csv to prevent old reports from failing
    col_names = ['col'+str(i) for i in range(1, 20)]

    try:
        data = pd.read_csv(market_path, names=col_names)
    except FileNotFoundError:
        logging.error(f"File {market_path} not found.")
        return None

    # remove rows containing items from remove list
    remove_list = ['Statement',  'Summary', 'Total', 'Order', 'SubTotal', 'ClosedLot']

    data = data[~data.isin(remove_list).any(axis=1)]        
    statement_types = data.col1.unique()

    # make the dicts
    df_dicts = dict()

    for s in statement_types:

        d = data[data.col1 == s].reset_index(drop=True)

        # column names as the first row
        d.columns = d.iloc[0]

        # remove columns with NaN
        d = d.loc[:, d.columns.notna()]

        # remove the first row
        d = d[1:]

        # # remove rows containing items from remove list
        # d = d[~d.isin(remove_list).any(axis=1)]
        
        # clearn approrpriately
        match s:

            case 'Open Positions':
                s = 'positions'
                df_posn = d.copy(deep=True).reset_index(drop=True)
                d = clean_df_positions(df_posn)

            case 'Trades':
                s = 'trades'
                df_trades = d.copy(deep=True).reset_index(drop=True)
                d = clean_df_trades(df_trades)

            case 'Option Exercises, Assignments and Expirations':
                s = 'expass'
                df = d.copy(deep=True).reset_index(drop=True)
                d = clean_df_trades(df)

            case 'Dividends':
                s = 'dividends'
                df = d.copy(deep=True).reset_index(drop=True)
                d = clean_df_dividends(df)

        year = int(market_path.parts[-1][4:8])
        d.insert(0, 'year', year)           

        df_dicts[s] = d

    # remove empty df from dictionary
    dfs = {k:v for k, v in df_dicts.items() if not v.empty}

    return dfs


def trade_extracts(MARKET: str, save: bool=False) -> dict:
    """
    Generate position, trades and dividends `all_dicts` from ibkr history
    - Go to `Performance & Reports` -> `Custom Statements` -> `history`. 
    - `history` has `Open Positions`, `Option Exercises/Assignments`, `Trades` and `Dividends` selected
    - Choose to remove account number from the CSV file
    - Ensure `Hide Details for Positions, Trades and Client Fees Sections?` is a `NO`
    - Choose `csv` with `daily` and appropriate `Custom Date Range`. Max is 365 days. 
    
    - Recast the file name to `<MARKET>_<DATE_FROM>_<DATE_TO>.csv`. e.g. `SNP_20230102_20231229.csv`
    - Move the files to `<root>/data/raw` folder
    - Run the function

    - Date format in `Symbol` field is assumed to be `%d%m%y`
    """

    ROOT = from_root()
    RAWPATH = ROOT / 'data' / 'raw'
    filename =  MARKET.lower() + '_ib_reports.pkl'
    SAVEPATH = ROOT / 'data' / 'master' / filename

    # get the respective files
    paths = glob.glob(str(RAWPATH / '*.csv'))
    market_paths = [Path(p) for p in paths if Path(p).parts[-1].split('_')[0] == MARKET]

    all_dicts = []

    for market_path in market_paths:

        result = clean_an_ib_portal_report(market_path)

        all_dicts.append(result)

    positions = []
    trades =  []
    expass = []
    dividends = []

    for r in all_dicts:
        for k, v in r.items():
            match k:
                case 'positions':
                    positions.append(v)
                case 'trades':
                    trades.append(v)
                case 'expass':
                    expass.append(v)
                case 'dividends':
                    dividends.append(v)
                case _:
                    logger.error(f"Unknown DataFrame {v.head()}")

    xn = dict()

    if trades:
        df_trades  = pd.concat(trades, ignore_index=True)
        xn['trades'] = df_trades
    if positions:
        df_positions  = pd.concat(positions, ignore_index=True)
        xn['positions'] = df_positions
    if expass:
        df_expass  = pd.concat(expass, ignore_index=True)
        xn['expass'] = df_expass
    if dividends:
        df_dividends  = pd.concat(dividends, ignore_index=True)
        xn['dividends'] = df_dividends

    if save:
        pickle_me(xn, SAVEPATH)

    return xn


if __name__ == "__main__":
    pass
