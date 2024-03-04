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

# ! OLD FUNCTIONS FOR REFERENCE ONLY !!!

def old_make_nse_lots() -> dict:
    """Generates lots for nse on cleansed IB symbols"""

    BLACKLIST = Vars('NSE').BLACKLIST
    lots = nse_get_fno_lot_sizes()
    
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