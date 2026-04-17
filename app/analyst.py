import json
import os
from typing import Any, Dict, List

from anthropic import AsyncAnthropic

from .models import ScreenerFilter
from .screener import FinancialScreener

_SYSTEM_PROMPT = """You are an institutional financial analyst with access to real-time market data.

Your capabilities:
1. get_real_time_quote(symbols) - Fetch current prices and volume
2. get_technical_indicators(symbol) - RSI, MACD, Bollinger Bands
3. get_market_sentiment() - News and social sentiment
4. calculate_portfolio_risk(positions) - VaR, correlations
5. find_trading_opportunities(criteria) - Screen for setups

Analysis framework:
- Start with macro context (indices, sectors, sentiment)
- Drill into specific securities
- Consider multiple timeframes
- Quantify risks and opportunities
- Provide actionable recommendations"""

_TOOLS = [
    {
        "name": "get_real_time_quote",
        "description": "Fetch current price, volume, and daily change for one or more symbols.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ticker symbols, e.g. ['AAPL', 'MSFT']",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": "Fetch technical indicators (RSI, daily change %) for a single symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_market_sentiment",
        "description": "Get broad market sentiment via major index levels (S&P 500, Dow, Nasdaq, VIX).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate_portfolio_risk",
        "description": (
            "Calculate weighted portfolio daily P&L and a 1-day 95% VaR proxy "
            "given a list of symbol/weight positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "weight": {
                                "type": "number",
                                "description": "Portfolio weight as a fraction (0–1)",
                            },
                        },
                        "required": ["symbol", "weight"],
                    },
                }
            },
            "required": ["positions"],
        },
    },
    {
        "name": "find_trading_opportunities",
        "description": (
            "Screen symbols for trading setups using price, volume, "
            "daily change %, and RSI filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "min_volume": {"type": "integer"},
                "min_change_percent": {"type": "number"},
                "max_change_percent": {"type": "number"},
                "min_rsi": {"type": "number"},
                "max_rsi": {"type": "number"},
            },
            "required": ["symbols"],
        },
    },
]


class ClaudeFinancialAnalyst:
    def __init__(self, screener: FinancialScreener):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-opus-4-7"
        self.screener = screener
        self.mcp_servers = []

    @property
    def available_tools(self) -> List[Dict]:
        return _TOOLS

    async def _execute_tool(self, name: str, tool_input: Dict) -> str:
        if name == "get_real_time_quote":
            data = await self.screener.get_real_time_quote(
                [s.upper() for s in tool_input["symbols"]]
            )
            return json.dumps(data)

        if name == "get_technical_indicators":
            quote = await self.screener.get_quote(tool_input["symbol"].upper())
            if quote is None:
                return json.dumps({"error": f"Symbol '{tool_input['symbol']}' not found"})
            return json.dumps(quote.model_dump())

        if name == "get_market_sentiment":
            indices = await self.screener.get_real_time_quote(
                ["^GSPC", "^DJI", "^IXIC", "^VIX"]
            )
            return json.dumps({"indices": indices})

        if name == "calculate_portfolio_risk":
            positions = tool_input["positions"]
            symbols = [p["symbol"].upper() for p in positions]
            quotes = await self.screener.get_real_time_quote(symbols)
            rows: List[Dict] = []
            weighted_change = 0.0
            for pos in positions:
                sym = pos["symbol"].upper()
                q = quotes.get(sym, {})
                chg = q.get("change_percent") or 0.0
                contrib = chg * pos["weight"]
                weighted_change += contrib
                rows.append(
                    {
                        "symbol": sym,
                        "weight": pos["weight"],
                        "change_percent": chg,
                        "weighted_contribution": round(contrib, 4),
                    }
                )
            return json.dumps(
                {
                    "positions": rows,
                    "portfolio_daily_change_pct": round(weighted_change, 4),
                    # 1-day 95% VaR proxy: |daily_change| × 1.645
                    "var_1d_95_proxy_pct": round(abs(weighted_change) * 1.645, 4),
                }
            )

        if name == "find_trading_opportunities":
            criteria = ScreenerFilter(
                **{k: v for k, v in tool_input.items() if v is not None}
            )
            criteria.symbols = [s.upper() for s in criteria.symbols]
            results = await self.screener.screen(criteria)
            return json.dumps([q.model_dump() for q in results])

        return json.dumps({"error": f"Unknown tool '{name}'"})

    async def analyze_market_conditions(self, query: str) -> Dict[str, Any]:
        """Claude analyzes markets with real-time data access."""
        messages = [{"role": "user", "content": query}]
        response = None

        while True:
            # Stream each turn; get_final_message() collects the full response.
            # The system prompt is cached (ephemeral) so repeated analyses are cheaper.
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=self.available_tools,
                tool_choice={"type": "auto"},
            ) as stream:
                response = await stream.get_final_message()

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return self._parse_claude_response(response)

    def _parse_claude_response(self, response) -> Dict[str, Any]:
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        return {
            "analysis": text,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                ),
            },
        }
