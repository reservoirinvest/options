from loguru import logger
from collections.abc import Iterable


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


def chunk_me(data: list, size: int=25) -> list:
    """cuts the list into chunks"""

    if type(data) is not list:
        logger.error(f"Data type needs to be a `list`, not {type(data)}")
        output = None
    else:
        output = [data[x: x+size] for x in range(0, len(data), size)]

    return output