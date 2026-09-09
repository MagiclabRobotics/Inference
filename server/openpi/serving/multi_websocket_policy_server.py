from __future__ import annotations

import asyncio
import dataclasses
import http
import itertools
import time
import traceback
from typing import Any

from loguru import logger
import numpy as np
from openpi.serving import server_logging
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import torch
import websockets.asyncio.server as _server
import websockets.frames


@dataclasses.dataclass
class _QueuedRequest:
    connection_id: int
    server_request_id: int
    request_id: int | None
    request_is_enveloped: bool
    raw_request_bytes: int | None
    recv_wait_ms: float
    unpack_ms: float
    obs: Any
    websocket: _server.ServerConnection
    send_lock: asyncio.Lock
    enqueue_monotonic: float
    remote: str


class MultiWebsocketPolicyServer:
    """Websocket policy server with a shared inference queue and multiple workers."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        request_log_every_n: int = 1,
        worker_count: int = 2,
        queue_size: int = 16,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._request_log_every_n = max(1, int(request_log_every_n))
        self._worker_count = max(1, int(worker_count))
        self._queue_size = max(1, int(queue_size))
        self._connection_counter = itertools.count(1)
        self._request_queue: asyncio.Queue[_QueuedRequest] | None = None
        self._packer = msgpack_numpy.Packer()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        self._request_queue = asyncio.Queue(maxsize=self._queue_size)
        workers = [asyncio.create_task(self._worker_loop(i + 1)) for i in range(self._worker_count)]
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                logger.info(
                    "multi policy server listening host={} port={} workers={} queue_size={}",
                    self._host,
                    self._port,
                    self._worker_count,
                    self._queue_size,
                )
                await server.serve_forever()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        if self._request_queue is None:
            raise RuntimeError("request queue is not initialized")
        connection_id = next(self._connection_counter)
        server_request_id = 0
        remote = _format_remote(websocket.remote_address)
        send_lock = asyncio.Lock()
        logger.info("multi connection opened conn={} remote={}", connection_id, remote)
        server_logging.log_server_event(
            "connection_open",
            transport="multi_websocket",
            connection_id=connection_id,
            remote=remote,
        )
        await websocket.send(self._packer.pack(self._metadata))

        while True:
            request_id = None
            request_is_enveloped = False
            try:
                start_time = time.monotonic()
                raw_request = await websocket.recv()
                recv_ms = (time.monotonic() - start_time) * 1000.0
                server_request_id += 1

                unpack_start = time.monotonic()
                message = msgpack_numpy.unpackb(raw_request)
                if isinstance(message, dict) and message.get("type") == "infer" and "payload" in message:
                    request_is_enveloped = True
                    request_id = _safe_int(message.get("request_id", message.get("request_index")))
                    obs = message["payload"]
                else:
                    request_id = server_request_id
                    obs = message
                unpack_ms = (time.monotonic() - unpack_start) * 1000.0

                queued = _QueuedRequest(
                    connection_id=connection_id,
                    server_request_id=server_request_id,
                    request_id=request_id,
                    request_is_enveloped=request_is_enveloped,
                    raw_request_bytes=_payload_size(raw_request),
                    recv_wait_ms=recv_ms,
                    unpack_ms=unpack_ms,
                    obs=obs,
                    websocket=websocket,
                    send_lock=send_lock,
                    enqueue_monotonic=time.monotonic(),
                    remote=remote,
                )
                await self._request_queue.put(queued)
                server_logging.log_server_event(
                    "inference_queued",
                    transport="multi_websocket",
                    connection_id=connection_id,
                    server_request_id=server_request_id,
                    request_id=request_id,
                    queue_depth=self._request_queue.qsize(),
                )
            except websockets.ConnectionClosed:
                logger.info("multi connection closed conn={} remote={} requests={}", connection_id, remote, server_request_id)
                server_logging.log_server_event(
                    "connection_close",
                    transport="multi_websocket",
                    connection_id=connection_id,
                    remote=remote,
                    requests=server_request_id,
                )
                break
            except Exception as exc:
                tb = traceback.format_exc()
                logger.exception(
                    "multi inference receive failed conn={} server_req={} request_id={} remote={}: {}",
                    connection_id,
                    server_request_id,
                    request_id,
                    remote,
                    exc,
                )
                await self._send_error(
                    websocket=websocket,
                    send_lock=send_lock,
                    request_is_enveloped=request_is_enveloped,
                    request_id=request_id,
                    server_request_id=server_request_id,
                    traceback_text=tb,
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _worker_loop(self, worker_id: int) -> None:
        if self._request_queue is None:
            raise RuntimeError("request queue is not initialized")
        while True:
            request = await self._request_queue.get()
            try:
                await self._process_request(worker_id, request)
            finally:
                self._request_queue.task_done()

    async def _process_request(self, worker_id: int, request: _QueuedRequest) -> None:
        queue_wait_ms = (time.monotonic() - request.enqueue_monotonic) * 1000.0
        infer_start = time.monotonic()
        try:
            action = await asyncio.to_thread(self._policy.infer, request.obs)
            policy_return_ms = (time.monotonic() - infer_start) * 1000.0
            cuda_sync_start = time.monotonic()
            cuda_synchronized = _synchronize_cuda()
            cuda_sync_ms = (time.monotonic() - cuda_sync_start) * 1000.0 if cuda_synchronized else None
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            action["server_timing"] = {
                "transport": "multi_websocket",
                "worker_id": worker_id,
                "request_id": request.request_id,
                "server_request_id": request.server_request_id,
                "queue_wait_ms": queue_wait_ms,
                "recv_wait_ms": request.recv_wait_ms,
                "unpack_ms": request.unpack_ms,
                "infer_ms": infer_ms,
                "policy_return_ms": policy_return_ms,
            }
            if cuda_sync_ms is not None:
                action["server_timing"]["cuda_sync_ms"] = cuda_sync_ms

            encode_start = time.monotonic()
            response_payload = (
                {
                    "type": "result",
                    "request_id": request.request_id,
                    "server_request_id": request.server_request_id,
                    "payload": action,
                }
                if request.request_is_enveloped
                else action
            )
            response = self._packer.pack(response_payload)
            pack_ms = (time.monotonic() - encode_start) * 1000.0
            send_start = time.monotonic()
            async with request.send_lock:
                await request.websocket.send(response)
            send_ms = (time.monotonic() - send_start) * 1000.0
            total_ms = (time.monotonic() - request.enqueue_monotonic) * 1000.0

            event = {
                "transport": "multi_websocket",
                "worker_id": worker_id,
                "connection_id": request.connection_id,
                "request_id": request.request_id,
                "server_request_id": request.server_request_id,
                "remote": request.remote,
                "queue_wait_ms": queue_wait_ms,
                "queue_depth": self._request_queue.qsize() if self._request_queue is not None else None,
                "request_bytes": request.raw_request_bytes,
                "response_bytes": len(response),
                "recv_wait_ms": request.recv_wait_ms,
                "unpack_ms": request.unpack_ms,
                "policy_return_ms": policy_return_ms,
                "cuda_sync_ms": cuda_sync_ms,
                "infer_ms": infer_ms,
                "pack_ms": pack_ms,
                "send_ms": send_ms,
                "total_ms": total_ms,
                "observation": _summarize_value(request.obs),
                "action": _summarize_value(action),
            }
            server_logging.log_server_event("inference", **event)
            if request.server_request_id % self._request_log_every_n == 0:
                logger.info(
                    "multi inference worker={} conn={} request_id={} queue={:.1f}ms infer={:.1f}ms "
                    "cuda_sync={} pack={:.1f}ms send={:.1f}ms total={:.1f}ms",
                    worker_id,
                    request.connection_id,
                    request.request_id,
                    queue_wait_ms,
                    infer_ms,
                    _format_optional_ms(cuda_sync_ms),
                    pack_ms,
                    send_ms,
                    total_ms,
                )
        except websockets.ConnectionClosed:
            logger.warning(
                "multi response dropped because connection closed worker={} conn={} request_id={}",
                worker_id,
                request.connection_id,
                request.request_id,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(
                "multi inference failed worker={} conn={} server_req={} request_id={}: {}",
                worker_id,
                request.connection_id,
                request.server_request_id,
                request.request_id,
                exc,
            )
            server_logging.log_server_event(
                "inference_error",
                transport="multi_websocket",
                worker_id=worker_id,
                connection_id=request.connection_id,
                request_id=request.request_id,
                server_request_id=request.server_request_id,
                error=str(exc),
                traceback=tb,
            )
            await self._send_error(
                websocket=request.websocket,
                send_lock=request.send_lock,
                request_is_enveloped=request.request_is_enveloped,
                request_id=request.request_id,
                server_request_id=request.server_request_id,
                traceback_text=tb,
            )

    async def _send_error(
        self,
        *,
        websocket: _server.ServerConnection,
        send_lock: asyncio.Lock,
        request_is_enveloped: bool,
        request_id: int | None,
        server_request_id: int,
        traceback_text: str,
    ) -> None:
        payload: Any
        if request_is_enveloped:
            payload = {
                "type": "error",
                "request_id": request_id,
                "server_request_id": server_request_id,
                "traceback": traceback_text,
            }
        else:
            payload = traceback_text
        async with send_lock:
            await websocket.send(self._packer.pack(payload) if not isinstance(payload, str) else payload)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    if not _is_websocket_upgrade(request):
        return connection.respond(
            http.HTTPStatus.UPGRADE_REQUIRED,
            "This endpoint only accepts WebSocket connections. Use /healthz for HTTP health checks.\n",
        )
    return None


def _is_websocket_upgrade(request: _server.Request) -> bool:
    return str(request.headers.get("Upgrade", "")).lower() == "websocket"


def _format_remote(remote_address: Any) -> str:
    if remote_address is None:
        return "unknown"
    if isinstance(remote_address, tuple):
        return ":".join(str(part) for part in remote_address)
    return str(remote_address)


def _payload_size(payload: Any) -> int | None:
    if isinstance(payload, bytes | bytearray | memoryview):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _synchronize_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    return True


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def _summarize_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _summarize_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        summary: dict[str, Any] = {"type": type(value).__name__, "len": len(value)}
        if len(value) <= 8:
            summary["items"] = [_summarize_value(item, depth=depth + 1) for item in value]
        return summary
    return value
