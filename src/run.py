# run the chosen program

import pandas as pd

from utils import (build_base_and_pickle, build_base_without_pickling,
                   get_sows_from_pickles, make_a_choice)

if __name__ == "__main__":

    # CHOOSE A MARKET
    # ===============

    msg_header = "Choose a number for MARKET:\n"
    choice_list = ["NSE", "SNP"]

    market_choice = make_a_choice(choice_list, msg_header)
    selection = next(iter(market_choice.keys()))
    MARKET = next(iter(market_choice.values()))

    # PAPER OR LIVE
    # =============
    msg_header = "Do you want to go with LIVE or PAPER trades ? :\n"
    choice_list = ["LIVE", "PAPER"]

    port_choice = make_a_choice(choice_list, msg_header)
    selection = next(iter(port_choice.keys()))

    port = next(iter(port_choice.values()))
    
    if port == "PAPER":
        PAPER = True
    else:
        PAPER = False

    # CHOOSE AN ACTION
    # ================

    action_choice_list = ["Sow with base from scratch and pickle",
                          "Sow from pickle",
                          "Build base from scratch but don't save pickles"]

    action_choice = make_a_choice(action_choice_list)

    my_choice = next(iter(action_choice.keys()))

    match my_choice:

        case 1:
            df = build_base_and_pickle(MARKET, PAPER)   

        case 2:
            df = get_sows_from_pickles(MARKET, PAPER)
            
        case 3: # Build base but DON'T SAVE PICKLES
            df = build_base_without_pickling(MARKET, PAPER)

        case _: # Unknown case
            print(f"Unknown choice...\n")
            df = pd.DataFrame([])
            
    print(df)