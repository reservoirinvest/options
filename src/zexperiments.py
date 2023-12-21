# !-- Experiments

import asyncio

from utils import chunk_me
from tqdm.asyncio import tqdm

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