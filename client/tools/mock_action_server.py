from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import msgpack_numpy  # noqa: E402


class MockActionChunkServer:
    """Small websocket server compatible with WebsocketClientPolicy."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        action_horizon: int = 50,
        action_dim: int = 14,
        action_start: float = 0.0,
        action_scale: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.action_start = float(action_start)
        self.action_scale = float(action_scale)
        self.metadata = dict(metadata or {})
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def start_in_thread(self, *, timeout_s: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self.serve_forever, name="mock-action-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=float(timeout_s)):
            raise TimeoutError(f"mock action server did not start within {timeout_s}s")

    def close(self, *, timeout_s: float = 5.0) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=float(timeout_s))
            self._thread = None

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        async with _server.serve(self._handler, self.host, self.port, compression=None, max_size=None) as server:
            sockets = list(server.sockets or [])
            if sockets:
                self.port = int(sockets[0].getsockname()[1])
            logging.info("mock action server listening host=%s port=%s", self.host, self.port)
            self._ready.set()
            await self._stop_event.wait()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._server_metadata()))
        server_request_id = 0
        while True:
            request_id = None
            try:
                raw_request = await websocket.recv()
                server_request_id += 1
                unpack_start = time.monotonic()
                message = msgpack_numpy.unpackb(raw_request)
                if not isinstance(message, dict) or message.get("type") != "infer" or "payload" not in message:
                    raise ValueError("expected websocket inference envelope with type='infer' and payload")
                request_id = message.get("request_id", message.get("request_index", server_request_id))
                payload = message["payload"]
                if not isinstance(payload, dict):
                    raise ValueError(f"expected payload dict, got {type(payload).__name__}")
                unpack_ms = (time.monotonic() - unpack_start) * 1000.0

                sleep_s = _request_sleep_s(payload)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                infer_start = time.monotonic()
                result = self._build_result(payload, request_id=request_id)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                result["server_timing"] = {
                    "transport": "mock_websocket",
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "unpack_ms": unpack_ms,
                    "infer_ms": infer_ms,
                }
                response = packer.pack(
                    {
                        "type": "result",
                        "request_id": request_id,
                        "server_request_id": server_request_id,
                        "payload": result,
                    }
                )
                await websocket.send(response)
            except websockets.ConnectionClosed:
                break
            except Exception:
                tb = traceback.format_exc()
                await websocket.send(
                    packer.pack(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "server_request_id": server_request_id,
                            "traceback": tb,
                        }
                    )
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Mock action server error. Traceback included in previous frame.",
                )
                raise

    def _server_metadata(self) -> dict[str, Any]:
        metadata = {
            "server_type": "mock_action_chunk_server",
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
        }
        metadata.update(self.metadata)
        return metadata

    def _build_result(self, payload: dict[str, Any], *, request_id: Any = None) -> dict[str, Any]:
        if "mock_actions" in payload:
            actions = np.asarray(payload["mock_actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"mock_actions must have shape [H, D], got {actions.shape}")
        else:
            horizon = int(payload.get("action_horizon", self.action_horizon))
            action_dim = int(payload.get("action_dim", self.action_dim))
            action_start = float(payload.get("action_start", self.action_start))
            action_scale = float(payload.get("action_scale", self.action_scale))
            values = np.arange(horizon * action_dim, dtype=np.float32).reshape(horizon, action_dim)
            chunk_offset = max(0, _request_id_int(request_id) - 1) * horizon * action_dim
            actions = action_start + action_scale * (values + chunk_offset)
        return {
            "actions": actions.astype(np.float32, copy=False),
            "actions_model": actions.astype(np.float32, copy=True),
        }


def _request_id_int(request_id: Any) -> int:
    try:
        return int(request_id)
    except Exception:
        return 1


def _request_sleep_s(payload: dict[str, Any]) -> float:
    if payload.get("sleep_s") is not None:
        return max(0.0, float(payload["sleep_s"]))
    if payload.get("sleep_ms") is not None:
        return max(0.0, float(payload["sleep_ms"]) / 1000.0)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a websocket mock server that returns deterministic action chunks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--action-start", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    server = MockActionChunkServer(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        action_start=args.action_start,
        action_scale=args.action_scale,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
