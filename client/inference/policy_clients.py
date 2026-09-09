from __future__ import annotations

from typing import Any


def create_policy_client(cfg: dict[str, Any]):
    transport = str(cfg.get("transport", "websocket")).replace("-", "_").lower()
    if transport == "shared_memory":
        from openpi_client import shared_memory_client_policy

        return shared_memory_client_policy.SharedMemoryClientPolicy(
            str(cfg.get("shared_memory_socket_path", "/tmp/openpi_policy.sock")),
            connect_timeout_s=cfg.get("connect_timeout_s"),
        )
    if transport in {"multi_websocket", "websocket_multi"}:
        from openpi_client import multi_websocket_client_policy

        endpoints = _as_string_list(cfg.get("endpoints") or cfg.get("servers"))
        return multi_websocket_client_policy.MultiWebsocketClientPolicy(
            str(cfg.get("host", "localhost")),
            int(cfg.get("port", 8000)),
            endpoints=endpoints,
            connections_per_endpoint=int(cfg.get("connections_per_endpoint", 2)),
            max_in_flight=int(cfg.get("max_in_flight", 8)),
            result_timeout_s=float(cfg.get("result_timeout_s", 0.0)),
            first_result_timeout_s=float(cfg.get("first_result_timeout_s", 0.0)),
            connect_retry_s=float(cfg.get("connect_retry_s", 5.0)),
        )
    if transport != "websocket":
        raise ValueError(f"Unsupported policy transport: {transport}")
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(
        str(cfg.get("host", "localhost")),
        int(cfg.get("port", 8000)),
    )


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]
