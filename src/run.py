# run the chosen program

import pandas as pd

from utils import (Timer, build_base_without_pickling, get_sows_from_pickles,
                   make_a_choice, sow_me, trade_extracts)

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
    msg_header = "\nDo you want to go with LIVE or PAPER trades ? :\n"
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
                          "Cancel API open orders. Sow orders from PICKLE and save sows.",
                          "Sow orders from pickle. Cancel API open orders. Don't save sows.",
                          "Build sow df from pickle. Don't order.",
                          "Build sow df from scratch. Save base pickles. But don't order or save sows.",
                          "Generate reports from IBKR portal statements and save."]

    action_header = "\nChoose a number for the action to be performed:\n"
    action_choice = make_a_choice(action_choice_list, action_header)

    my_choice = next(iter(action_choice.keys()))
    my_choice_text = next(iter(action_choice.values()))
    print(f"\nYou chose to `{my_choice_text}` for {port} {MARKET}...")

    match my_choice:

        case 0: # EXit
            out = None

        case 1: # Cancel API open orders. Sow orders from scratch.
            timer = Timer("Sow from scratch")
            timer.start()
            out = sow_me(MARKET=MARKET, build_from_scratch=True, save_sow=True)
            timer.stop()

        case 2: # Cancel API open orders. Sow orders from PICKLE and save sows.
            out = sow_me(MARKET=MARKET, build_from_scratch=False, save_sow=True)

        case 3:# Sow orders from pickle. Cancel API open orders. Don't save sows.
            out = sow_me(MARKET=MARKET, build_from_scratch=False, save_sow=False)

        case 4: # Build sow df from pickle. Don't order.
            out = get_sows_from_pickles(MARKET, PAPER)
            
        case 5: # Build sow df from scratch. Save base pickles. But don't order or save sows
            out = build_base_without_pickling(MARKET, PAPER)

        case 6: # Generate reports from IBKR portal statements and save.
            out = trade_extracts(MARKET, True)

        case _: # Unknown case
            print(f"Unknown choice...\n")
            out = pd.DataFrame([])

    # print(out)