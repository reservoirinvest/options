# run the chosen program

import pandas as pd

from utils import (build_base_and_pickle, build_base_without_pickling,
                   get_sows_from_pickles, make_a_choice)

if __name__ == "__main__":

    # CHOOSE A MARKET
    # ===============

    msg_header = "Choose a number for MARKET:\n"
    choice_list = {"NSE", "SNP"}

    market_choice = make_a_choice(choice_list, msg_header)
    selection = next(iter(market_choice.keys()))
    MARKET = next(iter(market_choice.values()))


    # CHOOSE AN ACTION
    # ================

    action_choice_list = ["Sow with base from scratch and pickle",
                          "Sow from pickle",
                          "Build base from scratch but don't save pickles"]

    action_choice = make_a_choice(action_choice_list)

    my_choice = next(iter(action_choice.keys()))

    match my_choice:

        case 1:
            df = build_base_and_pickle(MARKET)   

        case 2:
            df = get_sows_from_pickles(MARKET)
            
        case 3: # Build base but DON'T SAVE PICKLES
            df = build_base_without_pickling(MARKET)

        case _: # Unknown case
            print(f"Unknown choice...\n")
            df = pd.DataFrame([])
            
    print(df)