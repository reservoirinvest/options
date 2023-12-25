import asyncio
from pathlib import Path

import yaml
from from_root import from_root
from ib_insync import IB, Index, Stock
from loguru import logger
from nsepython import fnolist, nse_get_fno_lot_sizes
from utils import (Vars, make_chains, make_dict_of_qualified_contracts,
                   make_naked_puts, pickle_with_age_check, qualify_me, get_pickle)

ROOT = from_root() # Setting the root directory of the program
MARKET = 'NSE'

_vars = Vars(MARKET)
PORT = _vars.PORT

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


async def assemble_nse_underlyings(ib: IB) -> dict:
    """Assembles a dictionary of NSE underlying contracts"""

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
    qualified_unds = await qualify_me(ib, raw_nse_contracts)

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


if __name__ == "__main__":

    async def assemble_unds():
        with await IB().connectAsync(port=_vars.PORT) as ib:
            unds = await assemble_nse_underlyings(ib)
            return unds
    unds = asyncio.run(assemble_unds())
    unds_path = ROOT / 'data' / 'nse' / 'unds.pkl'
    pickle_with_age_check(unds, unds_path)

    async def assemble_chains():
        with await IB().connectAsync(port=_vars.PORT) as ib:
            chains = await make_chains(ib, list(unds.values()))
        return chains
    df_chains = asyncio.run(assemble_chains())
    chains_path = ROOT / 'data' / MARKET / 'df_chains.pkl'
    pickle_with_age_check(df_chains, chains_path)

    lots = make_nse_lots()
    lots_path = ROOT / 'data' / MARKET / 'lots.pkl'
    pickle_with_age_check(lots, lots_path)

    async def assemble_naked_puts():
        with await IB().connectAsync(port=_vars.PORT) as ib:
            nakeds = await make_naked_puts(ib, df_chains, MARKET='nse', how_many=-2)
        return nakeds
    df_nakeds = asyncio.run(assemble_naked_puts())
    nakeds_path = ROOT / 'data' / MARKET / 'df_nakeds.pkl'
    pickle_with_age_check(df_nakeds, nakeds_path)


    

    
