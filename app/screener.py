import asyncio
import time
from typing import Dict, List, Optional

import httpx

from .models import Quote, ScreenerFilter

_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


class FinancialScreener:
    def __init__(self, http_timeout: float = 5.0):
        self._timeout = http_timeout
        self._l1_cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _is_cache_fresh(self, cache_key: str, max_age: float) -> bool:
        if cache_key not in self._cache_timestamps:
            return False
        return (time.monotonic() - self._cache_timestamps[cache_key]) < max_age

    def _store_cache(self, cache_key: str, value: Dict) -> None:
        self._l1_cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.monotonic()

    async def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        raw_quotes = await asyncio.gather(*[self._fetch_single(s) for s in symbols])
        batch: Dict[str, Dict] = {}
        for symbol, raw in zip(symbols, raw_quotes):
            if raw is not None:
                cache_key = f"quote:{symbol}"
                self._store_cache(cache_key, raw)
                batch[symbol] = raw
        return batch

    async def _fetch_single(self, symbol: str) -> Optional[Dict]:
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
                return {
                    "symbol": symbol.upper(),
                    "price": price,
                    "change": round(change, 4) if change is not None else None,
                    "change_percent": round(change_pct, 4) if change_pct is not None else None,
                    "volume": meta.get("regularMarketVolume"),
                    "previous_close": prev,
                    "market_cap": meta.get("marketCap"),
                }
        except Exception:
            return None

    async def get_real_time_quote(self, symbols: List[str]) -> Dict:
        results = {}
        cache_hits = []
        cache_misses = []

        for symbol in symbols:
            cache_key = f"quote:{symbol}"
            if self._is_cache_fresh(cache_key, max_age=5):
                results[symbol] = self._l1_cache[cache_key]
                cache_hits.append(symbol)
            else:
                cache_misses.append(symbol)

        if cache_misses:
            try:
                batch_data = await self._fetch_batch_quotes(cache_misses)
                results.update(batch_data)
            except Exception:
                pass

        return results

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        data = await self.get_real_time_quote([symbol])
        raw = data.get(symbol)
        return Quote(**raw) if raw else None

    async def get_quotes(self, symbols: List[str]) -> List[Quote]:
        data = await self.get_real_time_quote(symbols)
        return [Quote(**v) for v in data.values()]

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
