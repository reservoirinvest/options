# To-do
- [x] generate chains for unds and pickle
- [x] generate options for strikes and expiries for each und
- [x] get price and margin for edge options for each und
- [x] determine expected price
- [x] place naked orders for new `sow`


- [x] Build RoM for naked_targets
- [ ] Automate `sow`
- [ ] Make function for getting status of a symbol (See `Symbol Statuses` below)
- [ ] Make functions for `cancel_api_ords` and `cancel_all_ords`
- [ ] place closing orders for existing `reap`

# For SNP

- [x] Keep PUTS ONLY for SNP nakeds
- [ ] Every sowed option should have a closing order reaps
- [ ] Every long stock should have a covered call in next month's expiry and protective put in 3 months
- [ ] Every short stock should have a covered put in next month's expiry and protective call in 3 months
- [ ] Every symbol should have its own standard deviations for naked sows, covers and protections

# For NSE
- [ ] Every sowed option should have a closing order that reaps

# Symbol `States`.

- 'unsowed': No orders to sow. No existing position.
- 'orphaned' : A call or put buy option that doesn't have an underlying position.

- 'perfect': Position present that is both covered and protected.
  
- 'uncovered': Position present but with no covered call or put option and without any open orders
- 'unprotected': Position present but with no protective call or put option and without any open orders
- 'imperfect': Position present that has no cover or protection options and without any open orders
  
- 'sowing' : Orders present to be sowed.
- 'covering' : Position protected with option but not covered and has open orders to cover
- 'protecting' : Position covered with option but not protected and has open orders to protect
- 'perfecting' : Position with open orders for covering and protecting

- 'unreaped' : A naked call or put option that doesn't have an open order to reap.
- 'reaping' : A naked call or put that has an open order to reap

