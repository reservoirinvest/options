# Program left at:

## Generating the right margins - independent of IBKR - for nse
 - `tests\znse_expiries.ipynb`. To get option expiry dates for nse.
    - function to take care of Thursday holidays
    - If `NIFTY` in symbol, should take this week's Thursday date
    - ...else take the last Thursday of the current month

- `tests\scrape_nse_margins.ipynb`. To complete SAMCO margins with the following defaults for a symbol
    - Expiry Date: Take nse expiry date functions from above
    - Price: Take the current market price. (May need a new nse function)
    - Qty: extract from lots

 - `/tests/one_symbol.ipynb` - to integrate lots and mutlipliers to get_unds_with_prices() function
    
 - `/tests/zrule_of_25.ipynb` - to complete offline margin calculations, first for SNP and then for NSE


# Functions for the following flow:
- [x] generate chains for unds and pickle
- [x] generate options for strikes and expiries for each und
- [x] get price and margin for edge options for each und
- [x] determine expected price from yml defaults
- [x] place naked orders for new `sow`


- [x] Build RoM for naked_targets
- [x] Automate `sow`
- [x] Build `run.py` with a simple CLI
- [x] Make functions for `cancel_api_ords`
- [x] Make function for `cancel_ords` in which you can give the specific orders.    
    -- replace sow_me's `cancel_api_ords` with `cancel_ords` only for equity position open orders.
- [x] Make function to extract info from IBKR portal reports
- [ ] Make rule of 25 for `computed margins` of option chain
- [ ] Make a function to limit number of naked orders per symbol with `MAXNAKEDORD`.
- [ ] Generate symbol `state`s (see below)

- [ ] Place closing orders for `unreaped` options
- [ ] Place opening `cover` orders for equity positions\

- [ ] Make function to consolidate successful `sows` 

# For SNP

- [x] Keep PUTS ONLY for SNP nakeds
- [ ] Every sowed option should have a closing order that reaps
- [ ] Every long stock should have a covered call in next week's expiry and protective put in 3 months
- [ ] Every short stock should have a covered put in next week's expiry and protective call in 3 months
- [ ] Every symbol should have its own standard deviations for naked sows, covers and protections

## Margin Calculation

Initial Margin
* 30% * Market Value of Stock, if Stock Value > $16.67 per share  
* \$5.00 per share, if Stock Value < \$16.67 and > $5.00   
* 100% of Market Value of Stock, if Stock Value < $5.00   
* \$2.50 per share, if Stock Value <= $2.50   

Maintenance Margin is the same as Initial Margin   

Check https://www.interactivebrokers.com/en/trading/margin-stocks.php

# For NSE
- [ ] Every sowed option should have a closing order that reaps

## Margin Calculation

All securities are classified into three groups for the purpose of VaR margin 

For the securities listed in Group I, scrip wise daily volatility calculated using the exponentially weighted moving average methodology is applied to daily returns. The scrip wise daily VaR is 6 times the volatility so calculated subject to a minimum of 9%. 
For the securities listed in Group II, scrip wise daily VaR is 6 times the volatility so calculated subject to a minimum of 21.5%. 
For the securities listed in Group III the VaR margin is 50% if traded at least once per week on any stock exchange; 75% otherwise. In case of Group III the securities shall be monitored on a weekly basis, and the VaR margin rates shall be increased to 75% if the security has not traded for a week. In case the VaR margin rate is 75% and the security trades during the day, the VaR margin rate shall be revised to 50% from start of next trading day 

### Notes:
URL for fnolist - https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O
URL for Group Type: https://nsearchives.nseindia.com/content/nsccl/C_CATG.T01
Check this link out for how to do it: https://stackoverflow.com/a/39213616

# Symbol `States` with colours.

- 'tbd' : Unknown status of the position or order. [grey]

- 'reaped' : An option position with a closing order. [purple]
- 'unreaped' : A naked call or put option that doesn't have an open order to reap. [light-yellow]

- 'uncovered': A (long/short) stock with no covered (call/put) buy orders [yellow]

- 'unsowed': No orders sown. No existing position. [white]
- 'orphaned' : A long call or put option (positive) position that doesn't have an underlying position. [blue]

- 'perfect': Ticker position present that is both covered and protected (or) Option position that is with a reap order. [green]
  

- 'unprotected': Ticker position present but with no protective call or put option and without any option open orders [light-red]
- 'imperfect': Ticker position present that has no cover or protection options and without any option open orders [light-brown]
  
- 'sowing' : Naked Option orders present to be sowed. [blue]
- 'covering' : Ticker position is protected with option but not covered and has option open orders to cover [cream]
- 'protecting' : Ticker position is covered with option but not protected and has option open orders to protect [pink]
- 'perfecting' : Ticker position with open orders for covering and protecting, that are not in position yet [light-green]



# NOTES
1. Option expiries are wonky in SNP. Use `US/Central Standard` Time to fix the wonky CBOE option expiry dates.  

# Good reads
1. Check out [Covered calls writing with protective put](https://www.thebluecollarinvestor.com/covered-call-writing-with-protective-puts-a-proposed-strategy/) article