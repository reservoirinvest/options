import asyncio
import glob
import logging
from pathlib import Path

import pandas as pd
import yaml
from from_root import from_root
from ib_insync import IB, Index, Stock, util
from loguru import logger
from nsepython import fnolist, nse_get_fno_lot_sizes

from utils import (Timer, Vars, create_target_opts, delete_all_pickles,
                   delete_files, get_margins, get_opt_price_ivs, get_pickle,
                   make_chains, make_dict_of_qualified_contracts,
                   make_qualified_opts, pickle_with_age_check, qualify_me)

ROOT = from_root() # Setting the root directory of the program
MARKET = 'NSE'

# Set variables
_vars = Vars(MARKET)
PORT = _vars.PORT
CID = _vars.CID
CALLSTDMULT = _vars.CALLSTDMULT
PUTSTDMULT = _vars.PUTSTDMULT
MINEXPROM = _vars.MINEXPROM
MINOPTSELLPRICE = _vars.MINOPTSELLPRICE
PREC = _vars.PREC

# Set paths for nse pickles
unds_path = ROOT / 'data' / MARKET / 'unds.pkl'
chains_path = ROOT / 'data' / MARKET / 'df_chains.pkl'
lots_path = ROOT / 'data' / MARKET / 'lots.pkl'

qualified_puts_path = ROOT / 'data' / MARKET / 'df_qualified_puts.pkl'
qualified_calls_path = ROOT / 'data' / MARKET / 'df_qualified_calls.pkl'

opt_prices_path = ROOT / 'data' / MARKET / 'df_opt_prices.pkl'
opt_margins_path = ROOT / 'data' / MARKET / 'df_opt_margins.pkl'

naked_targets_path = ROOT / 'data' / MARKET / 'df_naked_targets.pkl'

temp_path = ROOT / 'data' / MARKET / 'ztemp.pkl'

def nse2ib(nselist: list, path_to_yaml_file: Path) -> list:
    """Convert NSE symbols to IB friendly ones"""

    # get substitutions from YAML file
    with open(path_to_yaml_file, 'r') as f:
        subs = yaml.load(f, Loader=yaml.FullLoader)
    
    list_without_percent_sign = list(map(subs.get, nselist, nselist))

    # fix length to 9 characters
    ib_fnos = [s[:9] for s in list_without_percent_sign]

    return ib_fnos


def make_unqualified_nse_underlyings(symbols: list) -> list:
    """Makes raw underlying contracts for NSE"""

    contracts = [Index(symbol, 'NSE', 'INR') 
           if 'NIFTY' in symbol 
                else Stock(symbol, 'NSE', 'INR') 
            for symbol in symbols]
    
    return contracts


async def assemble_nse_underlyings(port: int=_vars.PORT) -> dict:
    """Assembles a dictionary of NSE underlying contracts"""

    with await IB().connectAsync(port=port) as ib:

        # get FNO list
        fnos = fnolist()

        # remove blacklisted symbols - like NIFTYIT that doesn't have options
        nselist = [n for n in fnos if n not in Vars('NSE').BLACKLIST]

        # clean to get IB FNOs
        nse2ib_yml_path = ROOT / 'data' / 'master' / 'nse2ibkr.yml'
        ib_fnos = nse2ib(nselist, nse2ib_yml_path)

        # make raw underlying fnos
        raw_nse_contracts = make_unqualified_nse_underlyings(ib_fnos)

        # qualify underlyings
        qualified_unds = await qualify_me(ib, raw_nse_contracts, desc='Qualifying Unds')

        unds_dict = make_dict_of_qualified_contracts(qualified_unds)

        return unds_dict


def make_nse_lots() -> dict:
    """Generates lots for nse on cleansed IB symbols"""

    lots = nse_get_fno_lot_sizes()
    
    clean_lots = {k: v for k, v 
                  in lots.items() 
                  if k not in _vars.BLACKLIST}
    
    # splits dict
    symbol_list, lot_list = zip(*clean_lots.items()) 
    
    nse2ib_yml_path = ROOT / 'data' / 'master' / 'nse2ibkr.yml'
    output = {k: v for k, v 
              in zip(nse2ib(symbol_list,nse2ib_yml_path), lot_list)}
    
    return output
    

def build_base(puts_only: bool = True):
    """Freshly build the base and pickle"""
    
    # Assemble underlyings
    unds = asyncio.run(assemble_nse_underlyings(PORT))        
    pickle_with_age_check(unds, unds_path, 0)

    # Make chains for underlyings and limit the dtes
    df_chains = asyncio.run(make_chains(port=PORT, MARKET=MARKET,
                                        contracts=list(unds.values())))
    df_chains = df_chains[df_chains.dte <= _vars.MAXDTE].reset_index(drop=True)
    pickle_with_age_check(df_chains, chains_path, 0)

    # Qualified put and options generated from the chains
    df_qualified_puts = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=PUTSTDMULT,
                            how_many=-1,
                            desc="Qualifying Puts"))     
    pickle_with_age_check(df_qualified_puts, qualified_puts_path, 0)

    df_qualified_calls = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=CALLSTDMULT,
                            how_many=1,
                            desc="Qualifying Calls"))     
    pickle_with_age_check(df_qualified_calls, qualified_calls_path, 0)

    if puts_only:
        df_all_qualified_options = df_qualified_puts
    else:
        df_all_qualified_options = pd.concat([df_qualified_calls, 
                                          df_qualified_puts], 
                                          ignore_index=True)

    # Get the option prices
    df_opt_prices = asyncio.run(get_opt_price_ivs(PORT, df_all_qualified_options))
    pickle_with_age_check(df_opt_prices, opt_prices_path, 0)

    # Get the lots for nse
    lots = make_nse_lots()        
    pickle_with_age_check(lots, lots_path, 0)

    # Get the option margins
    df_opt_margins = asyncio.run(get_margins(PORT, df_all_qualified_options, lots_path))
    pickle_with_age_check(df_opt_margins, opt_margins_path, 0)

    # Get alll the options
    df_naked_targets = create_target_opts(market=MARKET)
    
    pickle_with_age_check(df_naked_targets, naked_targets_path, 0)

    return None


def nse_nakeds_from_pickles(MARKET:str = 'NSE') -> pd.DataFrame:
    """Generates nakeds from existing margin and price pickles"""

    opt_prices_path = ROOT / 'data' / MARKET / 'df_opt_prices.pkl'
    opt_margins_path = ROOT / 'data' / MARKET / 'df_opt_margins.pkl'

    df_opt_prices = get_pickle(opt_prices_path)
    df_opt_margins = get_pickle(opt_margins_path)

    df_naked_targets = create_target_opts(df_opt_prices, 
                                    df_opt_margins, 
                                    _vars.MINEXPROM)
    
    return df_naked_targets


if __name__ == "__main__":    

    program_timer = Timer(f"{MARKET} base building")
    program_timer.start()

    # Delete log files
    
    log_folder_path = ROOT / 'log' / str(MARKET.lower()+"*.log")
    file_pattern = glob.glob(str(log_folder_path))

    delete_files(file_pattern)

    # Set the logger with logpath
    IBI_LOGPATH = ROOT / 'log' / f'{MARKET.lower()}_ib.log'
    LOGURU_PATH = ROOT / 'log' / f'{MARKET.lower()}_app.log'


    util.logToFile(IBI_LOGPATH, level=logging.ERROR)
    logger.add(LOGURU_PATH, rotation='10 MB', compression='zip', mode='w')

    GET_FROM_PICKLES = False

    if GET_FROM_PICKLES:
        df_opt_prices = get_pickle(opt_prices_path)
        df_opt_margins = get_pickle(opt_margins_path)

        df_naked_targets = create_target_opts(df_opt_prices, 
                                     df_opt_margins)

    else:
        delete_all_pickles(MARKET)
        build_base(puts_only = False)

    logger.info(program_timer.stop())

    

    
