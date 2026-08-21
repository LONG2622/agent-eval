# 8.13 Phase 3 - FastAPI Server
"""WebSocket endpoint and connection manager for live progress updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    async def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except (RuntimeError, ConnectionError, OSError):
                await self.disconnect(ws)


ws_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Echo for now; future: subscribe to specific job progress
            await ws.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)
