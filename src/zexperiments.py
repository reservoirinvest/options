from utils import sow_me, prepare_to_sow
import asyncio
from ib_insync import IB

MARKET = 'SNP'

# out = asyncio.run(cancel_all_api_orders('SNP'))
# out = prepare_to_sow(MARKET, save_sow=False, cancel_all_open_ords=True) # Works!!
out = sow_me(MARKET)
print(out)