# Data class templates

from dataclasses import dataclass
import pandas as pd
import pickle
from typing import Optional
import datetime
from ib_async import Order

def empty_the_df(df):
    """Empty the dataclass df"""
    empty_df = pd.DataFrame([df.__dict__]).iloc[0:0]
    return empty_df


@dataclass
class OpenOrder:
    """
    Open order template with Dummy data. Use:\n
    `df = OpenOrder().empty()`   
    """
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime.datetime = datetime.datetime.now()
    strike: float = 0.0
    right: str = "?" # Will be 'P' for Put, 'C' for Call
    orderId: int = 0
    order: Order = None
    permId: int = 0
    action: str = "SELL"  # 'BUY' | 'SELL'
    totalQuantity: float = 0.0
    lmtPrice: float = 0.0
    status: str = None

    def empty(self):
        return empty_the_df(self)
    

@dataclass
class Portfolio:
    """
    Portfolio template with Dummy data. Use:\n
    `df = OpenOrder().empty()`   
    """    
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime.datetime = datetime.datetime.now()
    strike: float = 0.0
    right: str = "?" # Will be 'P' for Put, 'C' for Call
    position: float = 0.0
    mktPrice: float = 0.0
    mktVal: float = 0.0	
    avgCost: float = 0.0	
    unPnL: float = 0.0	
    rePnL: float = 0.0

    def empty(self):
        return empty_the_df(self)

@dataclass
class Ticker:   #!!! Unfinished !!!
    symbol: str
    underlying_price: Optional[float] = None
    historical_data: Optional[pd.DataFrame] = None
    option_chains: Optional[dict] = None
    volatility: Optional[float] = None

    def fetch_underlying_price(self):
        # Fetch underlying price using some data source (e.g., API)
        # Here, we are just simulating some sample underlying price
        self.underlying_price = 125  # Sample underlying price

    def fetch_historical_data(self, start_date, end_date, source='api'):
        if source == 'api':
            # Fetch historical data from API
            # Implement your API fetching logic here
            # Here, we are just creating a sample DataFrame
            self.historical_data = pd.DataFrame({
                'Date': pd.date_range(start=start_date, end=end_date),
                'Close': [100, 105, 110, 115, 120]  # Sample data
            })
        elif source == 'file':
            # Load historical data from a pickled file
            file_path = f'{self.symbol}_historical_data.pkl'
            with open(file_path, 'rb') as f:
                self.historical_data = pickle.load(f)
        else:
            raise ValueError("Invalid source. Use 'api' or 'file'.")

    def fetch_option_chains(self):
        # Fetch option chains using some data source (e.g., API)
        # Here, we are just simulating some sample option data
        self.option_chains = {}  # Placeholder for option chains

    def calculate_volatility(self):
        # Calculate volatility using historical data
        # Here, we are just simulating some sample volatility
        self.volatility = 0.15  # Sample volatility



