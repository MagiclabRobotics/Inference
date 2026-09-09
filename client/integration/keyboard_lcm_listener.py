"""Subscribe to collector keyboard_event LCM messages for dataset save_path."""

from __future__ import annotations

import logging
import threading
from typing import Callable

try:
    import lcm
except Exception:  # pragma: no cover - optional dependency
    lcm = None

from integration.lcm_types.keyboard.keyboard_command_t import keyboard_command_t


logger = logging.getLogger(__name__)

CommandType = {
    "None": 0,
    "Start_Record": 1,
    "Save_Record": 2,
    "Discard_Record": 3,
}
TagType = {
    "None": 0,
    "Manual_Remote_Control": 1,
    "Dagger_Inference": 2,
    "Stop": 3,
}


class KeyboardLcmListener:
    def __init__(
        self,
        *,
        on_command: Callable[[keyboard_command_t], None],
        channel: str = "keyboard_event",
        lcm_url: str = "",
        handle_timeout_ms: int = 50,
    ):
        if lcm is None:
            raise RuntimeError("lcm is required for keyboard_event subscription")
        self._on_command = on_command
        self._channel = str(channel)
        self._handle_timeout_ms = int(handle_timeout_ms)
        self._lcm = lcm.LCM(lcm_url) if lcm_url else lcm.LCM()
        self._subscription = self._lcm.subscribe(self._channel, self._handle_message)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        def _spin() -> None:
            while self._running:
                try:
                    self._lcm.handle_timeout(self._handle_timeout_ms)
                except Exception as exc:
                    logger.warning("keyboard LCM handle failed: %s", exc)

        self._thread = threading.Thread(target=_spin, name="keyboard_lcm_spin", daemon=True)
        self._thread.start()
        logger.info("keyboard LCM listener started channel=%s", self._channel)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._subscription = None
        self._lcm = None

    def _handle_message(self, _channel: str, data: bytes) -> None:
        try:
            msg = keyboard_command_t.decode(data)
        except Exception as exc:
            logger.warning("keyboard_command_t decode failed: %s", exc)
            return
        try:
            self._on_command(msg)
        except Exception as exc:
            logger.exception("keyboard command handler failed: %s", exc)
