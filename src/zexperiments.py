# !-- Experiments

import asyncio

from utils import chunk_me
from tqdm.asyncio import tqdm
from io import StringIO

async def create_a_qualify_task(ib, contract):

    task = asyncio.create_task(ib.qualifyContractsAsync(contract), name=contract.symbol)

    return task


async def chunk_tasks(ib, contracts: list, timeout: int=44):
    
    tasks = [create_a_qualify_task(ib, contract) for contract in contracts]

    task_chunks = chunk_me(tasks, 44)

    return task_chunks
 

async def gather_results(task_chunks: list):

    all_tasks = []

    for tasks in task_chunks:

        all_tasks.append(await tqdm.gather(*tasks))

    return all_tasks


import requests
mode ='local'

if(mode=='local'):
    def nsefetch(payload):
        output = requests.get(payload,headers=headers).json()
        return output

headers = {
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    'Sec-Fetch-User': '?1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
}

def fnolist():
    # df = pd.read_csv("https://www1.nseindia.com/content/fo/fo_mktlots.csv")
    # return [x.strip(' ') for x in df.drop(df.index[3]).iloc[:,1].to_list()]

    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O')

    nselist=['NIFTY','NIFTYIT','BANKNIFTY']

    i=0
    for x in range(i, len(positions['data'])):
        nselist=nselist+[positions['data'][x]['symbol']]

    return nselist


import requests
import pandas as pd

import pandas as pd
storage_options = {'User-Agent': 'Mozilla/5.0'}

url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"
df = pd.read_table(url, storage_options=storage_options)