from pydantic import BaseModel
from typing import Optional, List


class Quote(BaseModel):
    symbol: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None


class ScreenerFilter(BaseModel):
    symbols: List[str]
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    min_change_percent: Optional[float] = None
    max_change_percent: Optional[float] = None
