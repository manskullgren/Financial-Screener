import asyncio
from datetime import datetime, timezone
from typing import Dict, Set

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        # Active WebSocket connections mapped by client ID
        self.active_connections: Dict[str, WebSocket] = {}

        # Symbol subscriptions for targeted data delivery
        self.subscriptions: Dict[str, Set[str]] = {}

        # Connection metadata for analytics and debugging
        self.client_metadata: Dict[str, Dict] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "subscriptions": [],
        }

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)
        self.client_metadata.pop(client_id, None)
        for subscribers in self.subscriptions.values():
            subscribers.discard(client_id)

    def subscribe(self, client_id: str, symbol: str) -> None:
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        self.subscriptions[symbol].add(client_id)
        if client_id in self.client_metadata:
            meta = self.client_metadata[client_id]
            if symbol not in meta["subscriptions"]:
                meta["subscriptions"].append(symbol)

    def unsubscribe(self, client_id: str, symbol: str) -> None:
        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(client_id)
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
        if client_id in self.client_metadata:
            subs = self.client_metadata[client_id]["subscriptions"]
            if symbol in subs:
                subs.remove(symbol)

    def get_active_symbols(self) -> list[str]:
        return [s for s, clients in self.subscriptions.items() if clients]

    def get_stats(self) -> Dict:
        return {
            "active_connections": len(self.active_connections),
            "active_symbols": len(self.get_active_symbols()),
            "clients": [
                {"client_id": cid, **meta}
                for cid, meta in self.client_metadata.items()
            ],
        }

    async def broadcast_to_symbol_subscribers(self, symbol: str, message: Dict) -> None:
        if symbol not in self.subscriptions:
            return

        disconnected_clients = []

        for client_id in self.subscriptions[symbol].copy():
            try:
                await asyncio.wait_for(
                    self.active_connections[client_id].send_json(message),
                    timeout=0.1,
                )
            except (asyncio.TimeoutError, Exception):
                disconnected_clients.append(client_id)

        for client_id in disconnected_clients:
            self.disconnect(client_id)
