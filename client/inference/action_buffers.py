"""把低频 action chunk 转换成控制循环可逐步消费的动作流。

所有 Buffer 都向 Runtime 暴露同一组核心操作：

``integrate_new_chunk``
    接收一次模型推理产生的 ``[horizon, action_dim]`` 动作，并与当前时间轴对齐。
``pop_next_action``
    控制循环每个周期取出一步动作。

模型推理和控制循环位于不同线程，因此每个实现都用锁保护 chunk、时间步和历史动作。
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Any

import numpy as np

from traceable_action_buffer import TraceableActionChunk


def _switch_info(
    old_chunk_id: int,
    new_chunk_id: int,
    old_executed_steps: int,
    old_remaining_steps: int,
    dropped_new_chunk_steps: int,
    new_chunk_len: int,
) -> dict[str, int]:
    return {
        "old_chunk_id": int(old_chunk_id),
        "new_chunk_id": int(new_chunk_id),
        "old_chunk_executed_steps": int(old_executed_steps),
        "old_chunk_remaining_steps": int(old_remaining_steps),
        "dropped_new_chunk_steps": int(dropped_new_chunk_steps),
        "new_chunk_len": int(new_chunk_len),
    }


## TODO： refactor the two buffers to share more code, and support temporal smoothing in both buffers.

class NaiveAsyncBuffer:
    """直接切到最新 chunk，并跳过异步推理期间已经过时的动作。"""

    def __init__(self, chunk_size: int = 50, state_dim: int = 14, smooth_method: str = "raw"):
        self.chunk_size = int(chunk_size)
        self.state_dim = int(state_dim)
        self.smooth_method = str(smooth_method).lower()
        self.lock = threading.Lock()
        self.current_chunk: np.ndarray | None = None
        self.current_chunk_id = 0
        self.chunk_start_t = 0
        self.global_t = 0
        self.last_action: np.ndarray | None = None
        self.last_action_chunk_id = 0
        self.last_action_step_index = 0
        self.prev_action_chunk_model: np.ndarray | None = None

    def add_chunk(
        self,
        actions_chunk: np.ndarray,
        start_timestep: int | None = None,
        *,
        chunk_id: int | None = None,
        actions_model_chunk: np.ndarray | None = None,
    ) -> dict[str, int] | None:
        with self.lock:
            arr = np.asarray(actions_chunk, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            old_chunk_id = int(self.current_chunk_id)
            old_executed_steps = max(0, int(self.global_t - self.chunk_start_t))
            old_remaining_steps = self.pending_count_unlocked()
            if actions_model_chunk is not None:
                self.prev_action_chunk_model = np.asarray(actions_model_chunk, dtype=np.float32).copy()

            if start_timestep is not None:
                # 新预测基于较早的观测。控制时间轴已经向前推进的部分不能再次执行。
                skip_steps = max(0, self.global_t - int(start_timestep))
            else:
                skip_steps = 0
            skip_steps = min(skip_steps, len(arr) - 1)
            self.current_chunk = arr.copy()
            self.current_chunk_id = self.current_chunk_id + 1 if chunk_id is None else int(chunk_id)
            self.chunk_start_t = self.global_t - skip_steps
            return _switch_info(
                old_chunk_id,
                self.current_chunk_id,
                old_executed_steps,
                old_remaining_steps,
                skip_steps,
                len(arr),
            )

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int,
        min_m: int = 8,
        actions_model_chunk: np.ndarray | None = None,
        drop_n: int | None = None,
        drop_reference_remaining: int | None = None,
        chunk_id: int | None = None,
    ) -> dict[str, int] | None:
        del min_m
        with self.lock:
            arr = np.asarray(actions_chunk, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            old_chunk_id = int(self.current_chunk_id)
            old_executed_steps = max(0, int(self.global_t - self.chunk_start_t))
            old_remaining_steps = self.pending_count_unlocked()
            if drop_n is None:
                if drop_reference_remaining is not None:
                    resolved_drop_n = max(0, int(drop_reference_remaining) - old_remaining_steps)
                else:
                    resolved_drop_n = min(max(0, int(max_k)), max(0, old_remaining_steps - 1))
            else:
                resolved_drop_n = max(0, int(drop_n))
            if drop_reference_remaining is not None and resolved_drop_n >= len(arr):
                return None
            resolved_drop_n = min(resolved_drop_n, len(arr) - 1)
            if actions_model_chunk is not None:
                self.prev_action_chunk_model = np.asarray(actions_model_chunk, dtype=np.float32).copy()
            self.current_chunk = arr.copy()
            self.current_chunk_id = self.current_chunk_id + 1 if chunk_id is None else int(chunk_id)
            self.chunk_start_t = self.global_t - resolved_drop_n
            return _switch_info(
                old_chunk_id,
                self.current_chunk_id,
                old_executed_steps,
                old_remaining_steps,
                resolved_drop_n,
                len(arr),
            )

    def pop_next_action(self) -> dict[str, Any] | None:
        with self.lock:
            action = None
            chunk_step_index = self.last_action_step_index
            chunk_id = self.last_action_chunk_id
            if self.current_chunk is not None:
                chunk_step_index = self.global_t - self.chunk_start_t
                if chunk_step_index < 0:
                    chunk_step_index = 0
                if chunk_step_index < len(self.current_chunk):
                    action = self.current_chunk[chunk_step_index].copy()
                    chunk_id = int(self.current_chunk_id)
            if action is None and self.last_action is not None:
                action = self.last_action.copy()
            self.global_t += 1
            if action is None:
                return None
            self.last_action = np.asarray(action, dtype=float).copy()
            self.last_action_chunk_id = int(chunk_id)
            self.last_action_step_index = int(chunk_step_index)
            return {
                "action": self.last_action.copy(),
                "chunk_id": int(chunk_id),
                "chunk_step_index": int(chunk_step_index),
            }

    def has_prediction(self, timestep: int | None = None) -> bool:
        del timestep
        with self.lock:
            if self.current_chunk is None:
                return self.last_action is not None
            chunk_index = self.global_t - self.chunk_start_t
            return chunk_index < len(self.current_chunk) or self.last_action is not None

    def get_current_timestep(self) -> int:
        with self.lock:
            return int(self.global_t)

    def peek_future_action(self, delay_steps: int) -> np.ndarray | None:
        with self.lock:
            if self.current_chunk is None:
                return None if self.last_action is None else self.last_action.copy()
            idx = self.global_t - self.chunk_start_t + max(0, int(delay_steps) - 1)
            idx = min(max(0, idx), len(self.current_chunk) - 1)
            return self.current_chunk[idx].copy()

    def get_chunk_progress(self) -> dict[str, int]:
        with self.lock:
            return {
                "chunk_id": int(self.current_chunk_id),
                "executed_steps": max(0, int(self.global_t - self.chunk_start_t)),
                "remaining_steps": self.pending_count_unlocked(),
                "action_horizon": 0 if self.current_chunk is None else int(len(self.current_chunk)),
            }

    def get_prev_action_chunk_model(self) -> np.ndarray | None:
        with self.lock:
            if self.prev_action_chunk_model is None:
                return None
            return np.asarray(self.prev_action_chunk_model, dtype=np.float32).copy()

    def pending_count_unlocked(self) -> int:
        if self.current_chunk is None:
            return 0
        return max(0, int(len(self.current_chunk) - max(0, self.global_t - self.chunk_start_t)))

    def pending_count(self) -> int:
        with self.lock:
            return self.pending_count_unlocked()


class TemporalEnsemblingBuffer:
    """按全局时间步保存多次预测，并对同一时刻的预测做指数加权。"""

    def __init__(
        self,
        max_timesteps: int = 10000,
        chunk_size: int = 50,
        state_dim: int = 14,
        exp_weight_m: float = 0.01,
        smooth_method: str = "temporal_ensembling",
    ):
        self.max_timesteps = int(max_timesteps)
        self.chunk_size = int(chunk_size)
        self.state_dim = int(state_dim)
        self.exp_weight_m = float(exp_weight_m)
        self.smooth_method = str(smooth_method).lower()
        self.lock = threading.Lock()
        self.predictions: dict[int, list[tuple[int, int, int, np.ndarray]]] = {}
        self.current_t = 0
        self.inference_count = 0
        self.last_action: np.ndarray | None = None
        self.last_action_chunk_id = 0
        self.prev_action_chunk_model: np.ndarray | None = None

    def add_chunk(
        self,
        actions_chunk: np.ndarray,
        start_timestep: int | None = None,
        *,
        chunk_id: int | None = None,
        actions_model_chunk: np.ndarray | None = None,
    ) -> dict[str, int] | None:
        with self.lock:
            arr = np.asarray(actions_chunk, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            old_chunk_id = int(self.last_action_chunk_id)
            old_executed_steps = int(self.current_t)
            old_remaining_steps = self.pending_count_unlocked()
            if actions_model_chunk is not None:
                self.prev_action_chunk_model = np.asarray(actions_model_chunk, dtype=np.float32).copy()
            if start_timestep is None:
                start_timestep = self.current_t
            resolved_chunk_id = self.inference_count + 1 if chunk_id is None else int(chunk_id)
            inference_idx = self.inference_count
            self.inference_count += 1
            self.last_action_chunk_id = resolved_chunk_id
            for step_index, action in enumerate(arr):
                # 多个不同请求可能同时覆盖同一个未来 timestep；先全部保留，
                # 到 pop 时再融合，而不是在新 chunk 到达时覆盖旧预测。
                timestep = int(start_timestep) + step_index
                if timestep < 0 or timestep >= self.max_timesteps:
                    continue
                self.predictions.setdefault(timestep, []).append(
                    (inference_idx, resolved_chunk_id, step_index, action.copy())
                )
            self._cleanup_old_predictions_unlocked()
            return _switch_info(old_chunk_id, resolved_chunk_id, old_executed_steps, old_remaining_steps, 0, len(arr))

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int,
        min_m: int = 8,
        actions_model_chunk: np.ndarray | None = None,
        drop_n: int | None = None,
        drop_reference_remaining: int | None = None,
        chunk_id: int | None = None,
    ) -> dict[str, int] | None:
        del min_m
        arr = np.asarray(actions_chunk, dtype=float)
        if drop_n is None and drop_reference_remaining is not None:
            drop_n = max(0, int(drop_reference_remaining) - self.pending_count())
        if drop_n is None:
            drop_n = min(max(0, int(max_k)), max(0, len(arr) - 1))
        drop_n = max(0, int(drop_n))
        if drop_n >= len(arr):
            return None
        return self.add_chunk(
            arr[drop_n:],
            start_timestep=self.get_current_timestep(),
            chunk_id=chunk_id,
            actions_model_chunk=actions_model_chunk,
        )

    def _cleanup_old_predictions_unlocked(self) -> None:
        cleanup_threshold = max(0, self.current_t - 10)
        for timestep in list(self.predictions):
            if timestep < cleanup_threshold:
                del self.predictions[timestep]

    def _get_action_unlocked(self, timestep: int) -> tuple[np.ndarray | None, int, int]:
        predictions = self.predictions.get(int(timestep), [])
        if not predictions:
            if self.last_action is None:
                return None, self.last_action_chunk_id, int(timestep)
            return self.last_action.copy(), self.last_action_chunk_id, int(timestep)
        predictions_sorted = sorted(predictions, key=lambda item: item[0])
        actions = np.asarray([item[3] for item in predictions_sorted], dtype=float)
        chunk_id = int(predictions_sorted[-1][1])
        step_index = int(predictions_sorted[-1][2])
        if len(actions) == 1:
            action = actions[0].copy()
        else:
            # prediction 按产生先后排列；指数权重控制历史预测的影响。
            indices = np.arange(len(actions), dtype=float)
            exp_weights = np.exp(-self.exp_weight_m * indices)
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions * exp_weights[:, np.newaxis]).sum(axis=0)
        self.last_action = action.copy()
        self.last_action_chunk_id = chunk_id
        return action, chunk_id, step_index

    def pop_next_action(self) -> dict[str, Any] | None:
        with self.lock:
            action, chunk_id, step_index = self._get_action_unlocked(self.current_t)
            self.current_t += 1
            if action is None:
                return None
            return {"action": action.copy(), "chunk_id": int(chunk_id), "chunk_step_index": int(step_index)}

    def has_prediction(self, timestep: int | None = None) -> bool:
        with self.lock:
            if timestep is None:
                timestep = self.current_t
            return int(timestep) in self.predictions and len(self.predictions[int(timestep)]) > 0

    def get_current_timestep(self) -> int:
        with self.lock:
            return int(self.current_t)

    def peek_future_action(self, delay_steps: int) -> np.ndarray | None:
        with self.lock:
            timestep = self.current_t + max(0, int(delay_steps) - 1)
            action, _, _ = self._get_action_unlocked(timestep)
            return None if action is None else action.copy()

    def get_chunk_progress(self) -> dict[str, int]:
        with self.lock:
            return {
                "chunk_id": int(self.last_action_chunk_id),
                "executed_steps": int(self.current_t),
                "remaining_steps": self.pending_count_unlocked(),
                "action_horizon": int(self.chunk_size),
            }

    def get_prev_action_chunk_model(self) -> np.ndarray | None:
        with self.lock:
            if self.prev_action_chunk_model is None:
                return None
            return np.asarray(self.prev_action_chunk_model, dtype=np.float32).copy()

    def pending_count_unlocked(self) -> int:
        return sum(1 for timestep in self.predictions if timestep >= self.current_t)

    def pending_count(self) -> int:
        with self.lock:
            return self.pending_count_unlocked()


class StreamActionBuffer:
    """维护一个当前动作流，在新旧 chunk 的重叠区间做线性过渡。

    ``k`` 表示当前 chunk 已执行步数。新预测返回时，其前 ``min(k, max_k)`` 步
    已经落后于控制时间轴，需要先丢弃。剩余的新动作再与旧 chunk 的未执行部分
    按 ``old -> new`` 的线性权重融合，避免切换瞬间产生关节跳变。
    """

    def __init__(self, max_chunks: int = 10, state_dim: int = 14, smooth_method: str = "temporal"):
        self.max_chunks = int(max_chunks)
        self.state_dim = int(state_dim)
        self.smooth_method = str(smooth_method).lower()
        self.lock = threading.Lock()
        self.cur_chunk = deque()
        self.last_action: np.ndarray | None = None
        self.k = 0
        self.chunk_id = 0
        self.prev_action_horizon = 0
        self.prev_action_chunk_model: np.ndarray | None = None

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int,
        min_m: int = 8,
        actions_model_chunk: np.ndarray | None = None,
        drop_n: int | None = None,
        drop_reference_remaining: int | None = None,
        chunk_id: int | None = None,
    ) -> dict[str, int] | None:
        with self.lock:
            arr = np.asarray(actions_chunk, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            old_chunk_id = int(self.chunk_id)
            old_executed_steps = int(self.k)
            old_remaining_steps = int(len(self.cur_chunk))
            max_k = max(0, int(max_k))
            if drop_n is None:
                if drop_reference_remaining is not None:
                    resolved_drop_n = max(0, int(drop_reference_remaining) - old_remaining_steps)
                else:
                    # raw 模式直接换块；平滑模式按已执行步数补偿推理期间流逝的时间。
                    resolved_drop_n = 0 if self.smooth_method == "raw" else min(self.k, max_k)
            else:
                resolved_drop_n = max(0, int(drop_n))
            if resolved_drop_n >= len(arr):
                return None
            if actions_model_chunk is not None:
                self.prev_action_chunk_model = np.asarray(actions_model_chunk, dtype=np.float32).copy()
            new_list = [a.copy() for a in arr[resolved_drop_n:]]
            new_action_horizon = len(new_list)
            new_chunk_id = self.chunk_id + 1 if chunk_id is None else int(chunk_id)
            if self.smooth_method == "raw":
                # RTC/Legato 等模型级连续性算法通常使用 raw，避免客户端再次平滑。
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )

            min_m = max(1, int(min_m))
            if not self.cur_chunk and self.last_action is not None:
                # 旧 chunk 刚好耗尽时，用最后动作延展出最短过渡区，避免直接跳变。
                old_list = [self.last_action.copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
            if not old_list:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )
            if len(old_list) < min_m:
                tail = np.asarray(old_list[-1], dtype=float).copy()
                old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])

            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )
            if len(old_list) > len(new_list):
                old_list = old_list[: len(new_list)]
                overlap_len = len(new_list)

            w_old = (
                np.linspace(1.0, 0.0, overlap_len, dtype=float)
                if overlap_len > 1
                else np.array([1.0], dtype=float)
            )
            # 过渡开始完全沿用旧动作，随后逐步提高新动作权重。
            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=float)
                + (1.0 - w_old[i]) * np.asarray(new_list[i], dtype=float)
                for i in range(overlap_len)
            ]
            self.cur_chunk = deque(smoothed + new_list[overlap_len:], maxlen=None)
            self.k = 0
            self.prev_action_horizon = new_action_horizon
            self.chunk_id = new_chunk_id
            return self._switch_info(
                old_chunk_id,
                new_chunk_id,
                old_executed_steps,
                old_remaining_steps,
                resolved_drop_n,
                new_action_horizon,
            )

    def pop_next_action(self) -> dict[str, Any] | None:
        with self.lock:
            if not self.cur_chunk:
                return None
            chunk_step_index = int(self.k)
            chunk_id = int(self.chunk_id)
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            action = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return {
                "action": action,
                "chunk_id": chunk_id,
                "chunk_step_index": chunk_step_index,
            }

    def peek_future_action(self, delay_steps: int) -> np.ndarray | None:
        with self.lock:
            if self.cur_chunk:
                idx = min(max(0, int(delay_steps) - 1), len(self.cur_chunk) - 1)
                return np.asarray(self.cur_chunk[idx], dtype=float).copy()
            if self.last_action is not None:
                return self.last_action.copy()
            return None

    def get_chunk_progress(self) -> dict[str, int]:
        with self.lock:
            return {
                "chunk_id": int(self.chunk_id),
                "executed_steps": int(self.k),
                "remaining_steps": int(len(self.cur_chunk)),
                "action_horizon": int(self.prev_action_horizon),
            }

    def get_prev_action_chunk_model(self) -> np.ndarray | None:
        with self.lock:
            if self.prev_action_chunk_model is None:
                return None
            return np.asarray(self.prev_action_chunk_model, dtype=np.float32).copy()

    def pending_count(self) -> int:
        with self.lock:
            return len(self.cur_chunk)

    @staticmethod
    def _switch_info(
        old_chunk_id: int,
        new_chunk_id: int,
        old_executed_steps: int,
        old_remaining_steps: int,
        dropped_new_chunk_steps: int,
        new_chunk_len: int,
    ) -> dict[str, int]:
        return {
            "old_chunk_id": int(old_chunk_id),
            "new_chunk_id": int(new_chunk_id),
            "old_chunk_executed_steps": int(old_executed_steps),
            "old_chunk_remaining_steps": int(old_remaining_steps),
            "dropped_new_chunk_steps": int(dropped_new_chunk_steps),
            "new_chunk_len": int(new_chunk_len),
        }


class TraceableStreamActionBuffer:
    """StreamActionBuffer 的可追踪版本，额外记录每个动作值来自哪个 chunk。

    算法行为与普通 Stream Buffer 对齐，适合调试平滑结果；生产环境不需要来源
    追踪时可使用普通实现，减少对象和日志开销。
    """

    def __init__(
        self,
        max_chunks: int = 10,
        state_dim: int = 14,
        smooth_method: str = "temporal",
        ensemble_new_weight: float = 0.6,
        include_action_trace: bool = True,
    ):
        self.max_chunks = int(max_chunks)
        self.state_dim = int(state_dim)
        self.smooth_method = str(smooth_method).lower()
        self.ensemble_new_weight = float(ensemble_new_weight)
        self.include_action_trace = bool(include_action_trace)
        self.lock = threading.Lock()
        self.cur_chunk: TraceableActionChunk = TraceableActionChunk([], chunk_id=0)
        self.last_action = None
        self.k = 0
        self.chunk_id = 0
        self.prev_action_horizon = 0
        self.prev_action_chunk_model: np.ndarray | None = None

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int,
        min_m: int = 8,
        actions_model_chunk: np.ndarray | None = None,
        drop_n: int | None = None,
        drop_reference_remaining: int | None = None,
        chunk_id: int | None = None,
    ) -> dict[str, int] | None:
        with self.lock:
            arr = np.asarray(actions_chunk, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            old_chunk_id = int(self.chunk_id)
            old_executed_steps = int(self.k)
            old_remaining_steps = len(self.cur_chunk)

            max_k = max(0, int(max_k))
            if drop_n is None:
                if drop_reference_remaining is not None:
                    resolved_drop_n = max(0, int(drop_reference_remaining) - old_remaining_steps)
                else:
                    resolved_drop_n = 0 if self.smooth_method == "raw" else min(self.k, max_k)
            else:
                resolved_drop_n = max(0, int(drop_n))
            if resolved_drop_n >= len(arr):
                return None
            if actions_model_chunk is not None:
                self.prev_action_chunk_model = np.asarray(actions_model_chunk, dtype=np.float32).copy()

            new_chunk_id = self.chunk_id + 1 if chunk_id is None else int(chunk_id)
            new_chunk = TraceableActionChunk.from_array(
                arr[resolved_drop_n:],
                chunk_id=new_chunk_id,
                source_offset=resolved_drop_n,
            )
            new_action_horizon = len(new_chunk)

            if self.smooth_method == "raw":
                self.cur_chunk = new_chunk
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                self.cur_chunk.print_source_map()
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )

            min_m = max(1, int(min_m))
            if len(self.cur_chunk) == 0:
                old_chunk = None
                if self.last_action is not None:
                    old_chunk = TraceableActionChunk([self.last_action.copy() for _ in range(min_m)], chunk_id=old_chunk_id)
            else:
                old_chunk = self.cur_chunk

            if old_chunk is None or len(old_chunk) == 0:
                self.cur_chunk = new_chunk
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )

            if len(old_chunk) < min_m:
                tail = old_chunk[-1]
                old_chunk = TraceableActionChunk(
                    list(old_chunk) + [tail.copy() for _ in range(min_m - len(old_chunk))],
                    chunk_id=old_chunk.id,
                )
            if len(old_chunk) > len(new_chunk):
                old_chunk = old_chunk[: len(new_chunk)]

            overlap_len = min(len(old_chunk), len(new_chunk))
            if self.smooth_method != "temporal_ensembling":
                overlap_len = min(overlap_len, min_m)
            if overlap_len <= 0:
                self.cur_chunk = new_chunk
                self.k = 0
                self.prev_action_horizon = new_action_horizon
                self.chunk_id = new_chunk_id
                return self._switch_info(
                    old_chunk_id,
                    new_chunk_id,
                    old_executed_steps,
                    old_remaining_steps,
                    resolved_drop_n,
                    new_action_horizon,
                )

            if self.smooth_method == "temporal_ensembling":
                new_weight = min(1.0, max(0.0, self.ensemble_new_weight))
                w_old = np.full(overlap_len, 1.0 - new_weight, dtype=float)
            else:
                w_old = (
                    np.linspace(1.0, 0.0, overlap_len, dtype=float)
                    if overlap_len > 1
                    else np.array([1.0], dtype=float)
                )
            smoothed = TraceableActionChunk.weighted_sum(
                ((w_old, old_chunk[:overlap_len]), (1.0 - w_old, new_chunk[:overlap_len])),
                chunk_id=new_chunk_id,
            )
            self.cur_chunk = TraceableActionChunk(list(smoothed) + list(new_chunk[overlap_len:]), chunk_id=new_chunk_id)
            self.cur_chunk.print_source_map()
            self.k = 0
            self.prev_action_horizon = new_action_horizon
            self.chunk_id = new_chunk_id
            return self._switch_info(
                old_chunk_id,
                new_chunk_id,
                old_executed_steps,
                old_remaining_steps,
                resolved_drop_n,
                new_action_horizon,
            )

    def pop_next_action(self) -> dict[str, Any] | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            chunk_step_index = int(self.k)
            chunk_id = int(self.chunk_id)
            if len(self.cur_chunk) == 1:
                self.last_action = self.cur_chunk[0]
            action_trace = self.cur_chunk.popleft()
            self.k += 1
            result = {
                "action": action_trace,
                "chunk_id": chunk_id,
                "chunk_step_index": chunk_step_index,
            }
            if self.include_action_trace:
                result["action_trace"] = action_trace
            self.cur_chunk.print_source_map()
            return result

    def peek_future_action(self, delay_steps: int) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) > 0:
                idx = min(max(0, int(delay_steps) - 1), len(self.cur_chunk) - 1)
                return np.asarray(self.cur_chunk[idx].value, dtype=float).copy()
            if self.last_action is not None:
                return np.asarray(self.last_action.value, dtype=float).copy()
            return None

    def get_chunk_progress(self) -> dict[str, int]:
        with self.lock:
            return {
                "chunk_id": int(self.chunk_id),
                "executed_steps": int(self.k),
                "remaining_steps": int(len(self.cur_chunk)),
                "action_horizon": int(self.prev_action_horizon),
            }

    def get_prev_action_chunk_model(self) -> np.ndarray | None:
        with self.lock:
            if self.prev_action_chunk_model is None:
                return None
            return np.asarray(self.prev_action_chunk_model, dtype=np.float32).copy()

    def pending_count(self) -> int:
        with self.lock:
            return len(self.cur_chunk)

    def get_current_action_value_sources(self):
        with self.lock:
            return self.cur_chunk.value_sources()

    def format_source_map(self, *, dim: int = 0, cell_width: int = 3) -> str:
        with self.lock:
            return self.cur_chunk.format_source_map(dim=dim, cell_width=cell_width)

    @staticmethod
    def _switch_info(
        old_chunk_id: int,
        new_chunk_id: int,
        old_executed_steps: int,
        old_remaining_steps: int,
        dropped_new_chunk_steps: int,
        new_chunk_len: int,
    ) -> dict[str, int]:
        return {
            "old_chunk_id": int(old_chunk_id),
            "new_chunk_id": int(new_chunk_id),
            "old_chunk_executed_steps": int(old_executed_steps),
            "old_chunk_remaining_steps": int(old_remaining_steps),
            "dropped_new_chunk_steps": int(dropped_new_chunk_steps),
            "new_chunk_len": int(new_chunk_len),
        }


def create_action_buffer(
    cfg: dict[str, Any],
    *,
    max_chunks: int,
    state_dim: int,
    smooth_method: str,
    ensemble_new_weight: float | None = None,
):
    """把配置中的 buffer 类型和平滑方法解析成具体实现。"""
    buffer_type = str(cfg.get("action_buffer", cfg.get("buffer_type", "stream"))).replace("-", "_").lower()
    if buffer_type in {"traceable", "traceable_stream", "trace"}:
        return TraceableStreamActionBuffer(
            max_chunks=max_chunks,
            state_dim=state_dim,
            smooth_method=smooth_method,
            ensemble_new_weight=float(ensemble_new_weight if ensemble_new_weight is not None else 0.6),
        )

    if buffer_type not in {"stream", "default"}:
        raise ValueError(f"Unsupported action_buffer: {buffer_type}")

    smooth_method = str(smooth_method).replace("-", "_").lower()
    if smooth_method in {"raw", "naive", "naive_async"}:
        return NaiveAsyncBuffer(
            chunk_size=int(cfg.get("chunk_size", 50)),
            state_dim=state_dim,
            smooth_method=smooth_method,
        )
    if smooth_method == "temporal_ensembling":
        exp_weight_m = cfg.get("exp_weight_m")
        if exp_weight_m is None:
            exp_weight_m = cfg.get("temporal_ensemble_exp_weight_m", 0.01)
        del ensemble_new_weight
        return TemporalEnsemblingBuffer(
            max_timesteps=int(cfg.get("max_publish_step", 10000)) + int(cfg.get("chunk_size", 50)),
            chunk_size=int(cfg.get("chunk_size", 50)),
            state_dim=state_dim,
            exp_weight_m=float(exp_weight_m),
            smooth_method=smooth_method,
        )
    return StreamActionBuffer(max_chunks=max_chunks, state_dim=state_dim, smooth_method=smooth_method)
