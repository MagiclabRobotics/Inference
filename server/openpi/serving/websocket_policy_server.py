"""通过 WebSocket 暴露 ``BasePolicy.infer``。

网络层只负责消息信封、MsgPack/NumPy 编解码和耗时记录，不包含模型预处理或
机器人控制逻辑。每个请求的业务 payload 都原样交给 Policy。
"""

import asyncio
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


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        request_log_every_n: int = 1,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._request_log_every_n = max(1, int(request_log_every_n))
        self._connection_counter = itertools.count(1)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("policy server listening host={} port={}", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """处理一个长连接上的连续推理请求。"""
        connection_id = next(self._connection_counter)
        remote = _format_remote(websocket.remote_address)
        logger.info("connection opened conn={} remote={}", connection_id, remote)
        server_logging.log_server_event("connection_open", connection_id=connection_id, remote=remote)
        packer = msgpack_numpy.Packer()

        # 握手后的第一帧固定为模型 metadata，客户端据此检查服务能力。
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        server_request_id = 0
        while True:
            request_id = None
            try:
                start_time = time.monotonic()
                raw_request = await websocket.recv()
                recv_ms = (time.monotonic() - start_time) * 1000
                server_request_id += 1
                unpack_start = time.monotonic()
                message = msgpack_numpy.unpackb(raw_request)
                if not isinstance(message, dict) or message.get("type") != "infer" or "payload" not in message:
                    raise ValueError(
                        "Expected websocket inference envelope with type='infer' and payload, "
                        f"got {type(message).__name__}"
                    )
                request_id = message.get("request_id", message.get("request_index", server_request_id))
                obs = message["payload"]
                unpack_ms = (time.monotonic() - unpack_start) * 1000

                # 网络层到此结束：输入 transform、模型采样和输出反归一化均由 Policy 完成。
                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                policy_return_ms = (time.monotonic() - infer_start) * 1000
                cuda_sync_start = time.monotonic()
                # GPU kernel 可能异步返回；同步后记录的 infer_ms 才覆盖真实执行时间。
                cuda_synchronized = _synchronize_cuda()
                cuda_sync_ms = (time.monotonic() - cuda_sync_start) * 1000 if cuda_synchronized else None
                infer_ms = (time.monotonic() - infer_start) * 1000

                action["server_timing"] = {
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "recv_wait_ms": recv_ms,
                    "unpack_ms": unpack_ms,
                    "infer_ms": infer_ms,
                    "policy_return_ms": policy_return_ms,
                }
                if cuda_sync_ms is not None:
                    action["server_timing"]["cuda_sync_ms"] = cuda_sync_ms
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                encode_start = time.monotonic()
                response_payload = {
                    # request_id 让客户端在并发/多连接模式下关联请求与响应。
                    "type": "result",
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "payload": action,
                }
                response = packer.pack(response_payload)
                pack_ms = (time.monotonic() - encode_start) * 1000
                send_start = time.monotonic()
                await websocket.send(response)
                send_ms = (time.monotonic() - send_start) * 1000
                prev_total_time = time.monotonic() - start_time

                event = {
                    "connection_id": connection_id,
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "remote": remote,
                    "request_bytes": _payload_size(raw_request),
                    "response_bytes": len(response),
                    "recv_wait_ms": recv_ms,
                    "unpack_ms": unpack_ms,
                    "policy_return_ms": policy_return_ms,
                    "cuda_sync_ms": cuda_sync_ms,
                    "infer_ms": infer_ms,
                    "pack_ms": pack_ms,
                    "send_ms": send_ms,
                    "total_ms": prev_total_time * 1000,
                    "observation": _summarize_value(obs),
                    "action": _summarize_value(action),
                }
                server_logging.log_server_event("inference", **event)
                if server_request_id % self._request_log_every_n == 0:
                    logger.info(
                        "inference conn={} request_id={} recv={:.1f}ms unpack={:.1f}ms infer={:.1f}ms "
                        "policy_return={:.1f}ms cuda_sync={} pack={:.1f}ms send={:.1f}ms total={:.1f}ms "
                        "req_bytes={} resp_bytes={}",
                        connection_id,
                        request_id,
                        recv_ms,
                        unpack_ms,
                        infer_ms,
                        policy_return_ms,
                        _format_optional_ms(cuda_sync_ms),
                        pack_ms,
                        send_ms,
                        prev_total_time * 1000,
                        event["request_bytes"],
                        event["response_bytes"],
                    )

            except websockets.ConnectionClosed:
                logger.info("connection closed conn={} remote={} requests={}", connection_id, remote, server_request_id)
                server_logging.log_server_event(
                    "connection_close",
                    connection_id=connection_id,
                    remote=remote,
                    requests=server_request_id,
                )
                break
            except Exception as exc:
                tb = traceback.format_exc()
                logger.exception(
                    "inference failed conn={} server_req={} request_id={} remote={}: {}",
                    connection_id,
                    server_request_id,
                    request_id,
                    remote,
                    exc,
                )
                server_logging.log_server_event(
                    "inference_error",
                    connection_id=connection_id,
                    request_id=request_id,
                    server_request_id=server_request_id,
                    remote=remote,
                    error=str(exc),
                    traceback=tb,
                )
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
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    if not _is_websocket_upgrade(request):
        return connection.respond(
            http.HTTPStatus.UPGRADE_REQUIRED,
            "This endpoint only accepts WebSocket connections. Use /healthz for HTTP health checks.\n",
        )
    # Continue with the normal request handling.
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
    if isinstance(value, str):
        return {"type": "str", "len": len(value), "preview": value[:120]}
    if isinstance(value, int | float | bool) or value is None:
        return value
    return type(value).__name__
