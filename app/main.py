import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyst import ClaudeFinancialAnalyst
from .connection_manager import ConnectionManager
from .models import Quote, ScreenerFilter
from .screener import FinancialScreener

manager = ConnectionManager()
screener = FinancialScreener()
analyst = ClaudeFinancialAnalyst(screener)

QUOTE_INTERVAL = 5  # seconds between broadcast cycles


async def _quote_broadcaster() -> None:
    while True:
        symbols = manager.get_active_symbols()
        if symbols:
            quotes = await screener.get_quotes(symbols)
            for quote in quotes:
                await manager.broadcast_to_symbol_subscribers(
                    quote.symbol,
                    {"type": "quote", "symbol": quote.symbol, "data": quote.model_dump()},
                )
        await asyncio.sleep(QUOTE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_quote_broadcaster())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Financial Screener", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/quotes/{symbol}", response_model=Quote)
async def get_quote(symbol: str):
    quote = await screener.get_quote(symbol.upper())
    if quote is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
    return quote


@app.post("/screen", response_model=List[Quote])
async def screen_stocks(criteria: ScreenerFilter):
    criteria.symbols = [s.upper() for s in criteria.symbols]
    return await screener.screen(criteria)


@app.get("/stats")
async def stats():
    return manager.get_stats()


@app.post("/analyze")
async def analyze(body: dict):
    query = body.get("query", "")
    if not query:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="'query' is required")
    return await analyst.analyze_market_conditions(query)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(client_id, websocket)
    await websocket.send_json({"type": "connected", "client_id": client_id})
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "").lower()
            symbol = data.get("symbol", "").upper()

            if not symbol:
                await websocket.send_json({"type": "error", "message": "symbol is required"})
                continue

            if action == "subscribe":
                manager.subscribe(client_id, symbol)
                quote = await screener.get_quote(symbol)
                await websocket.send_json({"type": "subscribed", "symbol": symbol})
                if quote:
                    await websocket.send_json(
                        {"type": "quote", "symbol": symbol, "data": quote.model_dump()}
                    )
            elif action == "unsubscribe":
                manager.unsubscribe(client_id, symbol)
                await websocket.send_json({"type": "unsubscribed", "symbol": symbol})
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown action '{action}'"}
                )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
