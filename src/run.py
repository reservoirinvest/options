# run the chosen program

import pandas as pd

from utils import (build_base_without_pickling,
                   get_sows_from_pickles, make_a_choice, sow_me, Timer)

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

    action_choice_list = ["Cancel API open orders. Sow orders from scratch.",
                          "Sow orders from pickle. Cancel API open orders. Don't save sows.",
                          "Build sow df from pickle. Don't order.",
                          "Build sow df from scratch. Don't order or save pickles."]

    action_choice = make_a_choice(action_choice_list)

    my_choice = next(iter(action_choice.keys()))
    my_choice_text = next(iter(action_choice.values()))
    print(f"\nYou chose to `{my_choice_text}` for {port} {MARKET}...")

    match my_choice:

        case 1:
            timer = Timer("Sow from scratch")
            timer.start()
            out = sow_me(MARKET=MARKET, build_from_scratch=True, save_sow=True)
            timer.stop()

        case 2:
            out = sow_me(MARKET=MARKET, build_from_scratch=False, save_sow=False)

        case 3:
            out = get_sows_from_pickles(MARKET, PAPER)
            
        case 4: # Build base but DON'T SAVE PICKLES
            out = build_base_without_pickling(MARKET, PAPER)

        case _: # Unknown case
            print(f"Unknown choice...\n")
            out = pd.DataFrame([])

    # print(out)