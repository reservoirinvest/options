import datetime
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from from_root import from_root
from ib_insync import IB
from loguru import logger
from tqdm.asyncio import tqdm

ROOT = from_root()

@dataclass
class Timediff:
    """Stores time difference"""
    days: int
    hours: int
    minutes: int
    seconds: float

class Vars:
    """Variables from var.yml"""
    def __init__(self, MARKET: str) -> None:

        self.MARKET = MARKET

        with open(ROOT / 'config' / 'var.yml', "rb") as f:
            data = yaml.safe_load(f)

        for k, v in data["COMMON"].items():
            setattr(self, k, v)

        for k, v in data[MARKET].items():
            setattr(self, k, v)


def get_file_age(file_path: Path):
    """Gets age of a file"""

    time_now = datetime.datetime.now()

    try:

        file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    except FileNotFoundError as e:

        logger.info(f"{str(file_path)} file is not found")
        
        file_age = None

    else:

        # convert time difference to days, hours, minutes, secs
        td = (time_now - file_time)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        seconds += td.microseconds / 1e6

        file_age = Timediff(*(days, hours, minutes, seconds))

    return file_age


async def qualify_unds(contracts: list, port: int=1300):
    """Qualify underlying contracts asynchronously"""

    with await IB().connectAsync(port=port) as ib:

        tasks = [ib.qualifyContractsAsync(c) for c in contracts]

        results = [await task_ 
                   for task_ 
                   in tqdm.as_completed(tasks, total=len(tasks), desc='Qualifying Unds')]

        return results


def make_dict_of_underlyings(qualified_contracts: list) -> dict:
    """Makes a dictionary of underlying contracts"""

    contracts_dict = {c[0].symbol: c[0] for c in qualified_contracts if c}

    return contracts_dict



def chunk_me(data: list, size: int=44) -> list:
    """cuts the list into chunks"""

    if type(data) is not list:
        logger.error(f"Data type needs to be a `list`, not {type(data)}")
        output = None
    else:
        output = [data[x: x+size] for x in range(0, len(data), size)]

    return output


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
        logger.info(f"Pickled underlying contracts to {file_name_with_path}")
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

def make_name(obj) -> str:
    """
    Builds name for the object.
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


if __name__ == "__main__":
    print(Vars('NSE'))




