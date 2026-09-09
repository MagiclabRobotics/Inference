from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from typing_extensions import override
import websockets.asyncio.client as _client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


@dataclasses.dataclass
class _Request:
    request_id: int
    payload: Dict
    submit_monotonic: float


@dataclasses.dataclass
class _Result:
    request_id: int | None
    payload: Dict | None = None
    error: BaseException | None = None
    endpoint: str | None = None


class MultiWebsocketClientPolicy(_base_policy.BasePolicy):
    """Non-blocking websocket client that dispatches observations to multiple server workers.

    Each call to infer enqueues the newest observation with a monotonically increasing
    request_id, then returns the next completed response in arrival order. If a
    response arrives after a newer id has already been returned, it is discarded.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        endpoints: Optional[list[str]] = None,
        connections_per_endpoint: int = 2,
        max_in_flight: int = 8,
        result_timeout_s: float = 0.0,
        first_result_timeout_s: float = 0.0,
        connect_retry_s: float = 5.0,
    ) -> None:
        self._endpoints = _build_endpoints(host=host, port=port, endpoints=endpoints)
        self._api_key = api_key
        self._connections_per_endpoint = max(1, int(connections_per_endpoint))
        self._max_in_flight = max(1, int(max_in_flight))
        self._result_timeout_s = max(0.0, float(result_timeout_s))
        self._first_result_timeout_s = max(0.0, float(first_result_timeout_s))
        self._connect_retry_s = max(0.1, float(connect_retry_s))
        self._request_counter = itertools.count(1)
        self._last_delivered_id = 0
        self._server_metadata: Dict = {}
        self._metadata_ready = threading.Event()
        self._closed = threading.Event()
        self._result_queue: queue.Queue[_Result] = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="multi-websocket-client", daemon=True)
        self._thread.start()
        self._metadata_ready.wait()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._closed.is_set():
            raise RuntimeError("MultiWebsocketClientPolicy is closed")

        request_id = next(self._request_counter)
        submitted = self._submit_request(_Request(request_id, obs, time.monotonic()))
        if not submitted:
            logging.warning("multi websocket request queue full; dropping request_id=%s", request_id)

        timeout = self._first_result_timeout_s if self._last_delivered_id == 0 else self._result_timeout_s
        result = self._next_deliverable_result(timeout)
        return {} if result is None else result

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._request_queue: asyncio.Queue[_Request] = asyncio.Queue(maxsize=self._max_in_flight)
        tasks = [
            self._loop.create_task(self._connection_worker(endpoint, worker_id))
            for endpoint in self._endpoints
            for worker_id in range(1, self._connections_per_endpoint + 1)
        ]
        self._loop.create_task(self._metadata_watchdog())
        try:
            self._loop.run_forever()
        finally:
            for task in tasks:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self._loop.close()

    async def _metadata_watchdog(self) -> None:
        while not self._metadata_ready.is_set() and not self._closed.is_set():
            await asyncio.sleep(0.05)
        if not self._metadata_ready.is_set():
            self._metadata_ready.set()

    def _submit_request(self, request: _Request) -> bool:
        future = asyncio.run_coroutine_threadsafe(self._put_request(request), self._loop)
        try:
            return bool(future.result(timeout=0.05))
        except Exception:
            return False

    async def _put_request(self, request: _Request) -> bool:
        if self._request_queue.full():
            return False
        self._request_queue.put_nowait(request)
        return True

    def _next_deliverable_result(self, timeout_s: float) -> Dict | None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if timeout_s <= 0:
                    result = self._result_queue.get_nowait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    result = self._result_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            if result.error is not None:
                raise RuntimeError(f"Error in multi websocket inference server {result.endpoint}: {result.error}") from result.error
            if result.payload is None:
                continue
            response_id = _safe_int(result.request_id)
            if response_id is None:
                logging.warning("multi websocket response without request_id from endpoint=%s; dropping", result.endpoint)
                continue
            if response_id <= self._last_delivered_id:
                logging.warning(
                    "dropping stale multi websocket response request_id=%s last_delivered=%s endpoint=%s",
                    response_id,
                    self._last_delivered_id,
                    result.endpoint,
                )
                continue
            self._last_delivered_id = response_id
            return result.payload

    async def _connection_worker(self, endpoint: str, worker_id: int) -> None:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        packer = msgpack_numpy.Packer()
        while not self._closed.is_set():
            try:
                logging.info("Connecting multi websocket worker endpoint=%s worker=%s", endpoint, worker_id)
                async with _client.connect(
                    endpoint,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                ) as websocket:
                    metadata = msgpack_numpy.unpackb(await websocket.recv())
                    if not self._metadata_ready.is_set():
                        self._server_metadata = metadata if isinstance(metadata, dict) else {"metadata": metadata}
                        self._metadata_ready.set()
                    await self._serve_connection(endpoint, worker_id, websocket, packer)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.warning(
                    "multi websocket worker disconnected endpoint=%s worker=%s error=%s",
                    endpoint,
                    worker_id,
                    exc,
                )
                if not self._metadata_ready.is_set():
                    await asyncio.sleep(min(self._connect_retry_s, 0.5))
                else:
                    await asyncio.sleep(self._connect_retry_s)

    async def _serve_connection(
        self,
        endpoint: str,
        worker_id: int,
        websocket: _client.ClientConnection,
        packer: msgpack_numpy.Packer,
    ) -> None:
        while not self._closed.is_set():
            request = await self._request_queue.get()
            try:
                data = packer.pack(
                    {
                        "type": "infer",
                        "request_id": request.request_id,
                        "payload": request.payload,
                    }
                )
                await websocket.send(data)
                response = await websocket.recv()
                if isinstance(response, str):
                    raise RuntimeError(response)
                message = msgpack_numpy.unpackb(response)
                result = _decode_response(message, endpoint=endpoint)
                if result.payload is not None:
                    result.payload.setdefault("client_timing", {})
                    result.payload["client_timing"].update(
                        {
                            "request_id": result.request_id,
                            "endpoint": endpoint,
                            "worker_id": worker_id,
                            "submit_to_receive_ms": (time.monotonic() - request.submit_monotonic) * 1000.0,
                        }
                    )
                self._result_queue.put(result)
            except Exception as exc:
                self._result_queue.put(_Result(request.request_id, error=exc, endpoint=endpoint))
                raise
            finally:
                self._request_queue.task_done()


def _build_endpoints(host: str, port: Optional[int], endpoints: Optional[list[str]]) -> list[str]:
    raw_endpoints = endpoints if endpoints else [part.strip() for part in str(host).split(",") if part.strip()]
    if not raw_endpoints:
        raw_endpoints = ["localhost"]
    return [_normalize_endpoint(endpoint, port) for endpoint in raw_endpoints]


def _normalize_endpoint(endpoint: str, port: Optional[int]) -> str:
    if endpoint.startswith("ws://") or endpoint.startswith("wss://"):
        return endpoint
    if port is not None and ":" not in endpoint:
        return f"ws://{endpoint}:{int(port)}"
    return f"ws://{endpoint}"


def _decode_response(message: Any, *, endpoint: str) -> _Result:
    if isinstance(message, dict) and message.get("type") == "error":
        return _Result(
            _safe_int(message.get("request_id", message.get("request_index"))),
            error=RuntimeError(message.get("traceback", "")),
            endpoint=endpoint,
        )
    if isinstance(message, dict) and message.get("type") == "result" and "payload" in message:
        payload = message["payload"]
        if not isinstance(payload, dict):
            return _Result(
                _safe_int(message.get("request_id", message.get("request_index"))),
                error=RuntimeError(f"Expected inference result dict, got {type(payload).__name__}"),
                endpoint=endpoint,
            )
        return _Result(_safe_int(message.get("request_id", message.get("request_index"))), payload=payload, endpoint=endpoint)
    if isinstance(message, dict):
        timing = message.get("server_timing")
        request_id = timing.get("request_id", timing.get("request_index")) if isinstance(timing, dict) else None
        return _Result(_safe_int(request_id), payload=message, endpoint=endpoint)
    return _Result(None, error=RuntimeError(f"Expected inference result dict, got {type(message).__name__}"), endpoint=endpoint)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
