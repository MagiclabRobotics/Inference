"""Single-key episode controls adapted from the SZK-3 Piper client."""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import tty


logger = logging.getLogger(__name__)


def _control_key(value: object, default: str) -> str:
    text = str(default if value is None else value).lower()
    if text == " " or text.strip() in {"space", "spacebar"}:
        return " "
    text = text.strip()
    if len(text) != 1 or not text.isascii() or not text.isprintable():
        raise ValueError(f"keyboard control key must be one ASCII character or 'space', got {value!r}")
    return text


class KeyboardEpisodeController:
    """Read episode keys while the runtime executes in the main thread."""

    def __init__(self, cfg: dict):
        self.start_key = _control_key(cfg.get("start_key"), "s")
        self.stop_key = _control_key(cfg.get("stop_key"), "space")
        self.quit_key = _control_key(cfg.get("quit_key"), "q")
        if len({self.start_key, self.stop_key, self.quit_key}) != 3:
            raise ValueError("keyboard start, stop and quit keys must be different")
        self._auto_start = not bool(cfg.get("start_paused", True))
        self._start_requested = threading.Event()
        self._quit_requested = threading.Event()
        self._reader_stop = threading.Event()
        self._runtime_lock = threading.Lock()
        self._runtime = None
        self._accept_start = False
        self._pending_stop = False
        self._fd: int | None = None
        self._terminal_attrs = None
        self._reader_thread: threading.Thread | None = None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested.is_set()

    def set_runtime(self, runtime) -> None:
        with self._runtime_lock:
            self._runtime = runtime
            stopping = self.quit_requested or self._pending_stop
        if runtime is not None and stopping:
            runtime.request_episode_stop()

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard episode control requires an interactive TTY")
        self._fd = sys.stdin.fileno()
        self._terminal_attrs = termios.tcgetattr(self._fd)
        try:
            tty.setcbreak(self._fd, termios.TCSANOW)
            self._reader_thread = threading.Thread(target=self._read_loop, name="episode-keyboard", daemon=True)
            self._reader_thread.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._reader_stop.set()
        self._start_requested.set()
        if self._reader_thread is not None:
            if self._reader_thread.ident is not None:
                self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._fd is not None and self._terminal_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._terminal_attrs)
            except termios.error:
                pass
        self._fd = None
        self._terminal_attrs = None

    def wait_for_start(self) -> bool:
        with self._runtime_lock:
            self._pending_stop = False
            self._accept_start = True
            if self._auto_start:
                self._auto_start = False
                self._start_requested.set()
        logger.info(
            "[episode control] READY: %r starts, %r stops and returns to init, %r quits",
            self.start_key, self.stop_key, self.quit_key,
        )
        try:
            while not self.quit_requested and not self._reader_stop.is_set():
                if self._start_requested.wait(timeout=0.1):
                    with self._runtime_lock:
                        if not self._start_requested.is_set():
                            continue
                        self._accept_start = False
                        self._start_requested.clear()
                        return not self.quit_requested and not self._reader_stop.is_set()
            return False
        finally:
            with self._runtime_lock:
                self._accept_start = False
                self._start_requested.clear()

    def _handle_key(self, key: str) -> None:
        key = key.lower()
        with self._runtime_lock:
            runtime = self._runtime
            if key == self.quit_key:
                self._quit_requested.set()
                self._start_requested.set()
            elif key == self.start_key:
                if self._accept_start and runtime is None:
                    self._start_requested.set()
                return
            elif key == self.stop_key:
                self._start_requested.clear()
                self._pending_stop = not self._accept_start
            else:
                return
        if runtime is not None:
            runtime.request_episode_stop()

    def _read_loop(self) -> None:
        assert self._fd is not None
        while not self._reader_stop.is_set() and not self.quit_requested:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                raw = os.read(self._fd, 1)
                if not raw:
                    self._handle_key(self.quit_key)
                    return
                self._handle_key(raw.decode(errors="ignore"))
            except (OSError, ValueError) as exc:
                logger.warning("keyboard input closed: %s", exc)
                self._handle_key(self.quit_key)
                return
