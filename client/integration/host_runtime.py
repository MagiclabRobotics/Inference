"""
非主线程宿主运行 InferenceRuntime。

``execute_actions=False`` 时不在本进程控臂，由采集 30Hz ``pop_action`` 取 ``stream_buffer``。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any


logger = logging.getLogger(__name__)


def run_embedded(runtime: Any, *, execute_actions: bool = True) -> None:
    """供推理服务后台线程驱动观测 + 推理；可选是否在宿主内执行 _control_loop。"""
    logger.info("server metadata: %s", runtime.policy.get_server_metadata())
    runtime._warmup_inference()
    runtime.threads = [
        threading.Thread(target=runtime._observation_thread, name="observation", daemon=True),
        threading.Thread(target=runtime._inference_thread, name="inference", daemon=True),
    ]
    for thread in runtime.threads:
        thread.start()
    mode = str(
        runtime.cfg.get("mode")
        or runtime.cfg.get("async_mode")
        or runtime.cfg.get("execution_mode")
        or "unknown"
    )
    logger.info(
        "embedded runtime started mode=%s execute_actions=%s publish_rate=%s inference_rate=%s",
        mode,
        execute_actions,
        runtime.cfg.get("publish_rate", 30),
        runtime.cfg.get("inference_rate", 3),
    )
    if execute_actions:
        runtime._control_loop()
        return
    while not runtime.shutdown.is_set():
        time.sleep(0.25)
