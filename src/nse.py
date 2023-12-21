import asyncio
from pathlib import Path

import yaml
from from_root import from_root
from ib_insync import Index, Stock
from nsepython import fnolist

from utils import Vars, qualify_unds, make_dict_of_underlyings

ROOT = from_root() # Setting the root directory of the program
PORT = Vars('NSE').PORT

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


def assemble_nse_underlyings() -> dict:
    """Assembles a dictionary of NSE underlying contracts"""

    # get FNO list
    nselist = fnolist()

    # clean to get IB FNOs
    nse2ib_yml_path = ROOT / 'data' / 'master' / 'nse2ibkr.yml'
    ib_fnos = nse2ib(nselist, nse2ib_yml_path)

    # make raw underlying fnos
    raw_nse_contracts = make_unqualified_nse_underlyings(ib_fnos)

    # qualify underlyings
    qualified_unds = asyncio.run(qualify_unds(raw_nse_contracts, port=PORT))

    unds_dict = make_dict_of_underlyings(qualified_unds)

    return unds_dict



if __name__ == "__main__":
    pass