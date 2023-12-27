import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from from_root import from_root
from ib_insync import IB, Index, Stock
from loguru import logger

from utils import (Timer, Vars, delete_files, get_margins, get_opt_price_ivs,
                   get_pickle, get_prec, make_chains,
                   make_dict_of_qualified_contracts, make_qualified_opts,
                   pickle_with_age_check, qualify_me)

ROOT = from_root() # Setting the root directory of the program
MARKET = 'SNP'

# Set variables
_vars = Vars(MARKET)
PORT = _vars.PORT
CALLSTDMULT = _vars.CALLSTDMULT
PUTSTDMULT = _vars.PUTSTDMULT
MINEXPROM = _vars.MINEXPROM


unds_path = ROOT / 'data' / MARKET / 'unds.pkl'
chains_path = ROOT / 'data' / MARKET / 'df_chains.pkl'
lots_path = ROOT / 'data' / MARKET / 'lots.pkl'

qualified_puts_path = ROOT / 'data' / MARKET / 'df_qualified_puts.pkl'
qualified_calls_path = ROOT / 'data' / MARKET / 'df_qualified_calls.pkl'

opt_prices_path = ROOT / 'data' / MARKET / 'df_opt_prices.pkl'
opt_margins_path = ROOT / 'data' / MARKET / 'df_opt_margins.pkl'

all_opts_path = ROOT / 'data' / MARKET / 'df_all_opts.pkl'

temp_path = ROOT / 'data' / MARKET / 'ztemp.pkl'

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


def add_indexes(df: pd.DataFrame, path_to_yaml_file: str) -> pd.DataFrame:
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


def split_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
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
        add_indexes(df_weekly_snps, indexes_path)
        .pipe(split_stocks_and_index)
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
    

async def assemble_snp_underlyings(port: int=_vars.PORT) -> dict:
    """Assembles a dictionary of SNP underlying contracts"""

    indexes_path = ROOT / 'data' / 'master' / 'snp_indexes.yml'
    
    df = make_snp_weeklies(indexes_path) \
         .pipe(make_unqualified_snp_underlyings)
    
    contracts = df.contract.to_list()

    with await IB().connectAsync(port=port) as ib:
    
        qualified_contracts = await qualify_me(ib, contracts, desc="Qualifying Unds")

    underlying_contracts = make_dict_of_qualified_contracts(qualified_contracts)

    return underlying_contracts

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
    unds = asyncio.run(assemble_snp_underlyings(PORT))        
    pickle_with_age_check(unds, unds_path, 0)

    # Make chains for underlyings
    df_chains = asyncio.run(make_chains(port=PORT,
                                        MARKET=MARKET, 
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

    # Get the option margins
    df_opt_margins = asyncio.run(get_margins(PORT, df_all_qualified_options))
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

    GET_FROM_PICKLES = False

    if GET_FROM_PICKLES:
        unds = get_pickle(unds_path)
        df_chains = get_pickle(chains_path)
        lots = get_pickle(lots_path)
        df_qualified_puts = get_pickle(qualified_puts_path)

    else:
        build_base()

    logger.info(program_timer.stop())

    