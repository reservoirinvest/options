import asyncio
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml
from from_root import from_root
from ib_insync import IB, Index, Stock
from loguru import logger
from tqdm.asyncio import tqdm
from utils import get_file_age

ROOT = from_root() # Setting the root directory of the program


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

    # more_df = pd.DataFrame(list(kv_pairs.items()), columns=['symbol', 'desc'])

    df_all = pd.concat([df, more_df], ignore_index=True)
    
    return df_all


def split_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """differentiates stocks and index"""
    
    df = df.assign(secType=np.where(df.desc.str.contains('Index'), 'IND', 'STK'))

    return df


def make_snp_weeklies(indexes_path: Path):
    """
    Makes snp weeklies with indexes
    """

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


def build_underlying_contracts(df: pd.DataFrame) -> pd.DataFrame:
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


async def qualify_unds(contracts: list):
    """Qualify underlying contracts asynchronously"""

    with await IB().connectAsync(port=1300) as ib:

        tasks = [ib.qualifyContractsAsync(c) for c in contracts]

        results = [await task_ 
                   for task_ 
                   in tqdm.as_completed(tasks, total=len(tasks), desc='Qualifying Unds')]

        return results
    

def make_dict_of_underlyings(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

    contracts_dict = {c[0].symbol: c[0] for c in qualified_contracts if c}

    return contracts_dict


def assemble_underlying_contracts() -> dict:
    """Assembles a dictionary of underlying contracts"""

    indexes_path = ROOT / 'data' / 'master' / 'snp_indexes.yml'
    
    df = make_snp_weeklies(indexes_path) \
         .pipe(build_underlying_contracts)
    
    contracts = df.contract.to_list()
    
    qualified_contracts = asyncio.run(qualify_unds(contracts))

    underlying_contracts = make_dict_of_underlyings(qualified_contracts)

    return underlying_contracts

def pickle_unds(und_contracts: dict, file_name_with_path: Path, minimum_age_in_days: int=1):

    existing_file_age = get_file_age(file_name_with_path)

    if existing_file_age is None: # No file exists
        pickle_me = True
    elif existing_file_age.days > minimum_age_in_days:
        pickle_me = True
    else:
        pickle_me = False

    if pickle_me:
        with open(str(file_name_with_path), 'wb') as handle:
            pickle.dump(und_contracts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Pickled underlying contracts")
    else:
        logger.info(f"Not pickled as existing file age {existing_file_age.days} is < {minimum_age_in_days}")


if __name__ == "__main__":

   und_contracts = assemble_underlying_contracts()

   print(und_contracts)

   pickle_unds(und_contracts, ROOT / 'data' / 'unds.pkl')


    