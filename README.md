# To-do
- [x] generate chains for unds and pickle
- [x] generate options for strikes and expiries for each und
- [x] get price and margin for edge options for each und
- [x] determine expected price
- [x] place naked orders for new `sow`


- [x] Build RoM for naked_targets
- [x] Automate `sow`
- [x] Build `run.py` with a simple CLI
- [x] Make functions for `cancel_api_ords`
- [x] Make function for `cancel_ords` in which you can give the specific orders.    
    -- replace sow_me's `cancel_api_ords` with `cancel_ords` only for equity position open orders.
- [ ] Make function for getting status of a symbol (See `Symbol Statuses` below)
- [ ] place closing orders for existing `reap`

# For SNP

- [x] Keep PUTS ONLY for SNP nakeds
- [ ] Every sowed option should have a closing order that reaps
- [ ] Every long stock should have a covered call in next week's expiry and protective put in 3 months
- [ ] Every short stock should have a covered put in next week's expiry and protective call in 3 months
- [ ] Every symbol should have its own standard deviations for naked sows, covers and protections

# For NSE
- [ ] Every sowed option should have a closing order that reaps

# Symbol `States` with colours.

- 'unsowed': No orders sown. No existing position. [white]
- 'orphaned' : A long call or put option (positive) position that doesn't have an underlying position. [grey]

- 'perfect': Ticker position present that is both covered and protected (or) Option position that is with a reap order. [green]
  
- 'uncovered': Ticker position present but with no covered call or put option and without any option open orders supporting it [yellow]
- 'unprotected': Ticker position present but with no protective call or put option and without any option open orders [light-red]
- 'imperfect': Ticker position present that has no cover or protection options and without any option open orders [purple]
  
- 'sowing' : Naked Option orders present to be sowed. [blue]
- 'covering' : Ticker position is protected with option but not covered and has option open orders to cover [cream]
- 'protecting' : Ticker position is covered with option but not protected and has option open orders to protect [pink]
- 'perfecting' : Ticker position with open orders for covering and protecting, that are not in position yet [light-green]

- 'unreaped' : A naked call or put option that doesn't have an open order to reap. [light-yellow]

# NOTES
1. Option expiries are wonky in SNP. Use `US/Central Standard` Time to fix the wonky CBOE option expiry dates.  

# Good reads
1. Check out [Covered calls writing with protective put](https://www.thebluecollarinvestor.com/covered-call-writing-with-protective-puts-a-proposed-strategy/) article