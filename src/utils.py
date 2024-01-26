import asyncio
import datetime
import glob
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union, Tuple, List

import numpy as np
import pandas as pd
import pytz
import yaml
from from_root import from_root
from ib_insync import IB, Contract, MarketOrder, util
from loguru import logger
from tqdm.asyncio import tqdm
from scipy.integrate import quad

ROOT = from_root()
BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar}{r_bar}"

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
                file.write(None)

    # print("All files deleted successfully.")

def delete_all_pickles(MARKET: str):
    """Deletes all pickle files for the MARKET"""

    pickle_folder_path = ROOT / 'data' / MARKET.lower()
    file_pattern = glob.glob(str(pickle_folder_path / '*.pkl'))

    delete_files(file_pattern)

    return None

# Usage example
folder_path = "/path/to/folder"

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
    """Yield items from any nested iterable; see Reference."""
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


async def qualify_me(ib: IB, 
                     contracts: list,
                     desc: str = 'Qualifying contracts'):
    """Qualify contracts asynchronously"""

    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in contracts]

    await tqdm.gather(*tasks, desc=desc)

    result = [r for t in tasks for r in t.result()]

    return result


def make_dict_of_qualified_contracts(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

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
        # logger.info(f"Pickled underlying contracts to {file_name_with_path}")
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

async def get_an_option_chain(ib: IB, contract:Contract, MARKET: str):
    """Get Option Chains from IB"""

    chain = await ib.reqSecDefOptParamsAsync(
    underlyingSymbol=contract.symbol,
    futFopExchange="",
    underlyingSecType=contract.secType,
    underlyingConId=contract.conId,
    )

    chain = chain[-1] if isinstance(chain, list) else chain

    df = chain_to_df(chain, contract, MARKET)

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

    return df4


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

    """Computes price and IV of a contract.

    OUTPUT: dict{localsymbol, price, iv}
    
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


async def combine_a_chain_with_stdev(ib: IB, contract, MARKET: str, sleep: float=3, ) -> pd.DataFrame:
    """makes a dataframe from chain and stdev for a symbol"""

    price_iv = await get_price_iv(ib, contract, sleep)
    symbol, undPrice, iv = price_iv

    chain = await get_an_option_chain(ib, contract, MARKET)
    
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


async def make_chains(port: int, 
                      contracts: list,
                      MARKET: str,
                      chunk_size: int=25, 
                      sleep: float=7,
                      CID: int=0,
                      ) -> pd.DataFrame:
    """Makes chains for a list of contracts. 2m 11s for 186 contracts.\n
    ...for optimal off-market, chunk-size of 25 and sleep of 7 seconds."""

    with await IB().connectAsync(port=port, clientId=CID) as ib:

        chunks = tqdm(chunk_me(contracts, chunk_size), desc="Getting chains")
        dfs = []

        for cts in chunks:

            tasks = [asyncio.create_task(combine_a_chain_with_stdev(ib=ib, contract=c, MARKET=MARKET, sleep=sleep)) for c in cts]

            results = await asyncio.gather(*tasks)

            df = pd.concat(results, ignore_index=True)

            dfs.append(df)

        df_chains = pd.concat(dfs, ignore_index=True)

    return df_chains


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
                                      how_many: int) -> pd.DataFrame:
    
    """Adjust the standard deviation to DTE, penalizes DTES closer to zero"""

    # Get the extra SD adjusted to DTE
    xtra_sd = 1-(df_chains.dte/100)

    # Build the series for revised SD
    sd_revised = STDMULT + xtra_sd if STDMULT > 0 else STDMULT - xtra_sd
    # Identify the closest standerd devs to the revised SD

    df_ch = df_chains.assign(sd_revised=sd_revised)
    closest_sdevs = df_ch.groupby(['symbol', 'dte'])[['sdev', 'sd_revised']]\
        .apply(lambda x: get_closest_values(x.sdev, 
                                            x.sd_revised.min(), 
                                            how_many))
    closest_sdevs.name = 'sdev1' 

    # Join the closest chains to the closest revised SD
    df_ch1 = df_ch.set_index(['symbol', 'dte']).join(closest_sdevs)

    # Get the target chains
    df_ch2 = df_ch1[df_ch1.apply(lambda x: x.sdev in x.sdev1, axis=1)] \
                        .reset_index()
    
    return df_ch2


async def make_qualified_opts(port:int, 
                    df_chains: pd.DataFrame, 
                    MARKET: str,
                    STDMULT: int,
                    how_many: int,
                    CID: int=0,
                    desc: str='Qualifying Options'
                    ) -> pd.DataFrame:
    
    """Make naked puts from chains, based on PUTSTDMULT"""

    df_ch2 = target_options_with_adjusted_sdev(df_chains, STDMULT, how_many)

    df_target = df_ch2[['symbol', 'strike', 'expiry', 'right',]].reset_index()

    options_list = make_target_option_contracts(df_target, MARKET=MARKET)

    # qualify target options
    with await IB().connectAsync(port=port, clientId=CID) as ib:
        options = await qualify_me(ib, options_list, desc=desc)

    # generate target puts
    df_puts = util.df(options).iloc[:, 2:6].\
        rename(columns={"lastTradeDateOrContractMonth": "expiry"}).\
            assign(qualified_opts=options)
    
    cols = df_puts.columns[:-1].to_list()

    # weed out other chain options
    df_output = df_chains.set_index(cols).join(df_puts.set_index(cols), on=cols)
    df_out = df_output.dropna().drop('localSymbol', axis=1).reset_index()

    return df_out

# GET OPTION PRICES FOR NAKED PUTS
# --------------------------------

async def get_opt_prices(ib:IB,
                         opt_contracts: list, 
                         chunk_size: int=44):
    """Gets option prices"""

    results = []

    pbar = tqdm(total=len(opt_contracts),
                desc="Getting option prices:",
                bar_format = BAR_FORMAT,
                ncols=80,
                leave=True,
            )

    chunks = chunk_me(opt_contracts, chunk_size)

    for cts in chunks:

        tasks = [asyncio.create_task(get_tick_data(ib, contract), name= contract.localSymbol) 
                for contract in cts]
    
        ticks = await asyncio.gather(*tasks)

        results.append(ticks)
        pbar.update(len(cts))

    flat_results =list(flatten(results))
    pbar.refresh()

    return flat_results


async def get_opt_price_ivs(port: int, 
                            df_qualified_opts: pd.DataFrame,
                            CID: int=0) -> pd.DataFrame:
    """gets option prices and IVs"""

    opt_contracts = df_qualified_opts.qualified_opts.to_list()

    with await IB().connectAsync(port=port, clientId=CID) as ib:

        opt_prices = await get_opt_prices(ib, opt_contracts)

    df_opt_prices = util.df(opt_prices)
    opt_cols = ['contract', 'time', 'bidSize', 'askSize', 'lastSize', 
    'volume', 'high', 'low', 'bid', 'ask', 'last', 'close']

    df_opts = df_opt_prices[opt_cols]
    duplicated_columns = [col for col in df_qualified_opts.columns if col in df_opts.columns]

    df = df_qualified_opts.join(df_opts.drop(duplicated_columns, axis=1))

    greek_cols = [ 'undPrice', 'impliedVol', 'delta', 'gamma',
    'vega', 'theta', 'optPrice']

    try:
        model_df= util.df(list(util.df(opt_prices).modelGreeks.values))[greek_cols]
    except TypeError:
        pass # ignore adding Greeks
    else:
        ask_df = util.df(list(util.df(opt_prices).askGreeks.values))[greek_cols]
        bid_df = util.df(list(util.df(opt_prices).bidGreeks.values))[greek_cols]

        df = df.join(model_df, lsuffix='Model').\
                    join(ask_df, lsuffix='Ask').\
                        join(bid_df, lsuffix='Bid')
    
    # puts optPrice column, if not in df
    if not 'optPrice' in list(df):
        df = df.assign(optPrice = df[['bid', 'ask', 'last', 'close']].max(axis=1))

    return df

### GET MARGINS AND COMMISSIONS
#------------------------------

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
                       lot_path: Path=None):
    
    """Gets a margin"""
    lot_size = 1 # Default for SNP

    if lot_path: # For NSE
        lot_size = get_pickle(lot_path).get(contract.symbol, None)

    order = MarketOrder('SELL', lot_size)

    def onError(reqId, errorCode, errorString, contract):
        logger.error(f"{contract.localSymbol} with reqId: {reqId} has errorCode: {errorCode} error: {errorString}")

    ib.errorEvent += onError
    wif = await ib.whatIfOrderAsync(contract, order)
    ib.errorEvent -= onError
    logger.remove()

    try:
        output = clean_a_margin(wif, contract.conId)
    except KeyError:
        output = {contract.conId: {
                  'margin': None,
                  'comm': None}}
        
        logger.error(f"{contract.localSymbol} has no margin and commission")

    output[contract.conId]['lot_size'] = lot_size
    return output


async def get_margins(port: int, 
                      df_nakeds: pd.DataFrame, 
                      lot_path: Path=None, 
                      chunk_size: int=100,
                      CID: int=0):

    opt_contracts = df_nakeds.qualified_opts.to_list()

    results = list()

    pbar = tqdm(total=len(opt_contracts),
                    desc="Getting margins:",
                    bar_format = BAR_FORMAT,
                    ncols=80,
                    leave=True,
                )

    df_nakeds = df_nakeds.assign(conId=[c.conId for c in opt_contracts]).\
                set_index('conId')

    chunks = chunk_me(opt_contracts, chunk_size)

    with await IB().connectAsync(port=port, clientId=CID) as ib:
        
        for cts in chunks:

            tasks = [asyncio.create_task(get_a_margin(ib, contract, lot_path), name= contract.localSymbol) 
                    for contract in cts]

            margin = await asyncio.gather(*tasks)

            results += margin
            pbar.update(len(cts))
            pbar.refresh()

    flat_results ={k: v for r in results for k, v in r.items()}
    df_mgncomm = pd.DataFrame(flat_results).T
    df_out = df_nakeds.join(df_mgncomm).reset_index()

    pbar.close()

    return df_out

    
def create_target_opts(market: str) -> pd.DataFrame:

    """Final naked target options with expected price"""
    
    MARKET = market.upper()
    DATAPATH = ROOT / 'data' / MARKET

    _vars = Vars(MARKET)
    MINEXPROM = _vars.MINEXPROM
    PREC = _vars.PREC
    MINOPTSELLPRICE = _vars.MINOPTSELLPRICE        

    df_opt_prices = get_pickle(DATAPATH / 'df_opt_prices.pkl')
    df_opt_margins = get_pickle(DATAPATH / 'df_opt_margins.pkl')
            

    cols = [x for x in list(df_opt_margins) if x not in list(df_opt_prices)]
    df_naked_targets = pd.concat([df_opt_prices, df_opt_margins[cols]], axis=1)

    # Get precise expected prices
    xp = ((MINEXPROM*df_naked_targets.dte/365*df_naked_targets.margin) +
            df_naked_targets.comm)/df_naked_targets.lot_size

    if MARKET == 'SNP':
        xp = xp/100 # correction for lot_size

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

    # for SNP lot_size should be multiple of 100 for correct rom
    rom=df.expPrice*df.lot_size/df.margin*365/df.dte

    if MARKET == 'SNP':
        rom = rom * 100

    df = df.assign(prop = df.sdev.apply(get_prob),
                   rom=rom)

    return df


def prepare_to_sow(market: str) -> pd.DataFrame():
    """Prepares the naked sow dataframe"""

    MARKET = market.upper()
    _vars = Vars(MARKET)
    PORT = _vars.PORT

    df = create_target_opts(market=MARKET)

    # Get the portfolio and open orders
    with IB().connect(port=PORT) as ib:
        ib.reqAllOpenOrders() 
        df_pf = quick_pf(ib)

    # Remove targets which are already in the portfolio
    if isinstance(df_pf, pd.DataFrame):
        df = df[~df.symbol.isin(set(df_pf.symbol))]

    # Remove symbols with open orders
    trades = ib.trades()
    if trades:
        all_trades_df = (util.df(t.contract for t in trades).join(
            util.df(t.orderStatus
                    for t in trades)).join(util.df(t.order for t in trades),
                                            lsuffix="_"))

        all_trades_df.rename({"lastTradeDateOrContractMonth": "expiry"},
                                axis="columns",
                                inplace=True)
        trades_cols = ["conId", "symbol", "secType", "expiry",
            "strike", "right", "orderId", "permId", "action",
            "totalQuantity", "lmtPrice", "status",]

        dfo = all_trades_df[trades_cols]
        df_openords = dfo[all_trades_df.status.isin(_vars.ACTIVE_STATUS)]

    else:
        df_openords = pd.DataFrame([{'symbol': None}])
        

    if ~df_openords.empty:  # There are some open orders
        df = df[~df.symbol.isin(set(df_openords.symbol))]

    # Keep options with margin and expected price only
    margins_only = ~df.margin.isnull()
    with_expPrice = ~df.expPrice.isnull()

    cleaned = margins_only & with_expPrice

    df = df[cleaned]

    return df


# ORDER PLACING WITH PORTFOLIO CHECKS
# +++++++++++++++++++++++++++++++++++

def place_orders(ib: IB, cos: Union[Tuple, List], blk_size: int = 25) -> List:
    """!!!CAUTION!!!: This places orders in the system
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
            ib.sleep(0.75)

    return trades

def quick_pf(ib) -> Union[None, pd.DataFrame]:
    """Gets the portfolio dataframe"""
    pf = ib.portfolio()  # returns an empty [] if there is nothing in the portfolio

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(columns=["contract", "account"]))
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
        df_pf = None

    return df_pf