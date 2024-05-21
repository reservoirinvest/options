# ! OLD FUNCTIONS FOR REFERENCE ONLY !!!

import re
from collections.abc import Iterable

import yaml
from from_root import from_root
from loguru import logger

from utils import (Vars, get_nse_payload, get_pickle, nse2ib, pickle_me,
                   zerodha_lots_expiries)

ROOT = from_root()

def make_name(obj) -> str:
    """
    Builds name for option objects.
    Checks for expiry.
    If obj is a list names with first and last element

    Input: contract or a list / set of contracts
    """

    # make object to list
    if isinstance(obj, Iterable):
        li = list(obj)
    else:
        li = [obj]

    # check length
    if len(li) > 1:
        
        # check if expiry exists
        if li[0].lastTradeDateOrContractMonth:

            name = \
                li[0].symbol + \
                li[0].lastTradeDateOrContractMonth[-4:] + \
                li[0].right + \
                str(li[0].strike) + \
                "->" + \
                li[-1].symbol + \
                li[-1].lastTradeDateOrContractMonth[-4:] + \
                li[-1].right + \
                str(li[-1].strike)
            
        else:

            name = \
                li[0].symbol + \
                "->" + \
                li[-1].symbol
    
    else: # only one contract

        # check if expiry exists
        if li[0].lastTradeDateOrContractMonth:

            name = li[0].symbol + \
                li[0].lastTradeDateOrContractMonth[-4:] + \
                li[0].right + \
                str(li[0].strike)
        
        else:

            name = li[0].symbol

    return name


def get_lots_from_nse(save: bool=True) -> dict:
    """
    Gets lots from nse site without any symbol cleansing
    """

    MKT_LOTS_URL = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'

    MARKET = 'NSE'
    DATAPATH = ROOT / 'data' / MARKET.lower()

    lots_path = DATAPATH / 'lots.pkl'

    response = get_nse_payload(MKT_LOTS_URL).text

    res_dict = {} # unclean symbol results dictionary

    try:
        for line in response.split('\n'):
            if line != '' and re.search(',', line) and (line.casefold().find('symbol') == -1):
                (code, name) = [x.strip() for x in line.split(',')[1:3]]
                res_dict[code] = int(name)
    except ValueError:
        logger.error("Fresh lots not available in 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'")
        res_dict = get_pickle(lots_path, print_msg=True)
        save = False

    if save:
        pickle_me(res_dict, lots_path)
    
    return res_dict


def get_nse_native_fno_list() -> list:
    """Gets a dictionary of native nse symbols from nse.com"""

    res_dict = get_lots_from_nse()

    nselist = [k for k, _ in res_dict.items()]

    return nselist


def make_nse_lots() -> dict:

    """Makes lots for NSE on cleansed IB symbols without BLACKLIST"""

    MARKET = 'NSE'
    DATAPATH = ROOT / 'data' / MARKET.lower()

    lots_path = DATAPATH / 'lots.pkl'

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

    return clean_lots


def old_make_nse_lots() -> dict:
    """Generates lots for nse on cleansed IB symbols"""

    BLACKLIST = Vars('NSE').BLACKLIST

    df = zerodha_lots_expiries()
    lots = df.set_index('symbol').lot.to_dict()
    # lots = nse_get_fno_lot_sizes()
    
    clean_lots = {k: v for k, v 
                  in lots.items() 
                  if k not in BLACKLIST}

    clean_lots['NIFTY'] = 50 # Sometimes nse_fno_lots_sizes gives 0!
    
    # splits dict
    symbol_list, lot_list = zip(*clean_lots.items()) 
    
    nse2ib_yml_path = ROOT / 'data' / 'master' / 'nse2ibkr.yml'
    output = {k: v for k, v 
              in zip(nse2ib(symbol_list), lot_list)}
    
    return output

