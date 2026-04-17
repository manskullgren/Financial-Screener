import asyncio
from typing import Dict, List, Optional

import httpx

from .models import Quote, ScreenerFilter

_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


class FinancialScreener:
    def __init__(self, http_timeout: float = 5.0):
        self._timeout = http_timeout

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
                resp = await client.get(f"{_YF_BASE}/{symbol}")
                resp.raise_for_status()
                data = resp.json()
                meta = data["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice")
                prev = meta.get("previousClose") or meta.get("chartPreviousClose")
                change = (price - prev) if price is not None and prev else None
                change_pct = (change / prev * 100) if change is not None and prev else None
                return Quote(
                    symbol=symbol.upper(),
                    price=price,
                    change=round(change, 4) if change is not None else None,
                    change_percent=round(change_pct, 4) if change_pct is not None else None,
                    volume=meta.get("regularMarketVolume"),
                    previous_close=prev,
                    market_cap=meta.get("marketCap"),
                )
        except Exception:
            return None

    async def get_quotes(self, symbols: List[str]) -> List[Quote]:
        results = await asyncio.gather(*[self.get_quote(s) for s in symbols])
        return [q for q in results if q is not None]

    async def screen(self, criteria: ScreenerFilter) -> List[Quote]:
        quotes = await self.get_quotes(criteria.symbols)
        filtered: List[Quote] = []
        for q in quotes:
            if criteria.min_price is not None and (q.price is None or q.price < criteria.min_price):
                continue
            if criteria.max_price is not None and (q.price is None or q.price > criteria.max_price):
                continue
            if criteria.min_volume is not None and (q.volume is None or q.volume < criteria.min_volume):
                continue
            if criteria.min_change_percent is not None and (
                q.change_percent is None or q.change_percent < criteria.min_change_percent
            ):
                continue
            if criteria.max_change_percent is not None and (
                q.change_percent is None or q.change_percent > criteria.max_change_percent
            ):
                continue
            filtered.append(q)
        return filtered
