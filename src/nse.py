import asyncio
import logging
from pathlib import Path

import pandas as pd
import yaml
from from_root import from_root
from ib_insync import IB, Index, Stock, util
from loguru import logger
from nsepython import fnolist, nse_get_fno_lot_sizes
from utils import (Timer, Vars, delete_files, get_margins, get_opt_price_ivs,
                   get_pickle, get_prec, make_chains,
                   make_dict_of_qualified_contracts, make_qualified_opts,
                   pickle_with_age_check, qualify_me)

ROOT = from_root() # Setting the root directory of the program
MARKET = 'NSE'

# Set variables
_vars = Vars(MARKET)
PORT = _vars.PORT
CALLSTDMULT = _vars.CALLSTDMULT
PUTSTDMULT = _vars.PUTSTDMULT
MINEXPROM = _vars.MINEXPROM

# Set paths for nse pickles
unds_path = ROOT / 'data' / MARKET / 'unds.pkl'
chains_path = ROOT / 'data' / MARKET / 'df_chains.pkl'
lots_path = ROOT / 'data' / MARKET / 'lots.pkl'

qualified_puts_path = ROOT / 'data' / MARKET / 'df_qualified_puts.pkl'
qualified_calls_path = ROOT / 'data' / MARKET / 'df_qualified_calls.pkl'

opt_prices_path = ROOT / 'data' / MARKET / 'df_opt_prices.pkl'
opt_margins_path = ROOT / 'data' / MARKET / 'df_opt_margins.pkl'

all_opts_path = ROOT / 'data' / MARKET / 'df_all_opts.pkl'

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


# Combine all target option margins and prices
def create_target_opts(df_opt_margins: pd.DataFrame,
                        df_opt_prices: pd.DataFrame,
                        MINEXPROM: float):
    
    """Final naked target options with expected price"""

    cols = [x for x in list(df_opt_margins) if x not in list(df_opt_prices)]
    df_all_opts = pd.concat([df_opt_prices, df_opt_margins[cols]], axis=1)

    # Get precise expected prices
    df_all_opts = df_all_opts.assign(expPrice = df_all_opts.apply(lambda x: 
                                    get_prec(((MINEXPROM*x.dte/365*x.margin)+x.comm)
                                            /x.lot_size, 0.05), 
                                                axis=1))

    return df_all_opts
    

def build_base():
    """Freshly build the base and pickle"""
    
    # Assemble underlyings
    unds = asyncio.run(assemble_nse_underlyings(PORT))        
    pickle_with_age_check(unds, unds_path, 0)

    # Make chains for underlyings
    df_chains = asyncio.run(make_chains(port=PORT, MARKET=MARKET,
                                        contracts=list(unds.values())))
    pickle_with_age_check(df_chains, chains_path, 0)

    # Qualified put and options generated from the chains
    df_qualified_puts = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=PUTSTDMULT,
                            how_many=-2))     
    pickle_with_age_check(df_qualified_puts, qualified_puts_path, 0)

    df_qualified_calls = asyncio.run(make_qualified_opts(PORT, 
                            df_chains, 
                            MARKET=MARKET,
                            STDMULT=CALLSTDMULT,
                            how_many=2))     
    pickle_with_age_check(df_qualified_calls, qualified_calls_path, 0)

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
    df_all_opts = create_target_opts(df_opt_prices, 
                                     df_opt_margins, 
                                     _vars.MINEXPROM)
    
    pickle_with_age_check(df_all_opts, all_opts_path, 0)

    return None

if __name__ == "__main__":    

    program_timer = Timer("Base")
    program_timer.start()

    # Delete log files
    folder_path = ROOT / 'log'
    file_pattern = MARKET.lower()+'*.log'
    file_list = [folder_path.joinpath(folder_path, file_name) for file_name in file_pattern]
    delete_files(file_list)

    # Set the logger with logpath
    IBI_LOGPATH = ROOT / 'log' / 'nse_ib.log'
    LOGURU_PATH = ROOT / 'log' / 'nse_app.log'

    util.logToFile(IBI_LOGPATH, level=logging.ERROR)
    logger.add(LOGURU_PATH, rotation='10 MB', compression='zip', mode='w')

    GET_FROM_PICKLES = False

    if GET_FROM_PICKLES:
        unds = get_pickle(unds_path)
        df_chains = get_pickle(chains_path)
        lots = get_pickle(lots_path)
        df_qualified_puts = get_pickle(qualified_puts_path)

    else:
        build_base()

    logger.info(program_timer.stop())

    

    
