from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ib_insync import IB
from loguru import logger


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

    more_df = pd.DataFrame(list(kv_pairs.items()), columns=['symbol', 'desc'])

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



if __name__ == "__main__":

    from utils import get_project_root

    root = get_project_root()

    indexes_path = root / 'data' / 'master' / 'snp_indexes.yml'
    
    df = make_snp_weeklies(indexes_path)
    
    logger.info(df)


    