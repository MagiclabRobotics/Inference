from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import itertools
from typing import NamedTuple, overload

import numpy as np


class ActionValueSourceItem(NamedTuple):
    chunk_id: int
    step_index: int
    value: float
    weight: float = 1.0


ActionValueSource = tuple[ActionValueSourceItem, ...]
ActionValueSources = tuple[ActionValueSource, ...]


@dataclass(frozen=True)
class TraceableAction:
    value: np.ndarray
    value_source: ActionValueSources | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float).copy()
        object.__setattr__(self, "value", value)
        value_source = self.value_source
        if value_source is None:
            value_source = tuple(() for _ in range(value.size))
        object.__setattr__(self, "value_source", _normalize_value_sources(value_source, value.size))

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)

    def copy(self) -> TraceableAction:
        return TraceableAction(self.value.copy(), self.value_source)


class TraceableActionChunk:
    _id_counter = itertools.count(1)

    def __init__(self, actions: Iterable[TraceableAction], chunk_id: int | None = None):
        self.id = int(next(self._id_counter) if chunk_id is None else chunk_id)
        self._actions = [action.copy() for action in actions]

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        chunk_id: int | None = None,
        source_offset: int = 0,
    ) -> TraceableActionChunk:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"values must have shape [H, D], got {arr.shape}")
        resolved_id = int(next(cls._id_counter) if chunk_id is None else chunk_id)
        actions = [
            TraceableAction(
                row,
                tuple(
                    (ActionValueSourceItem(resolved_id, int(source_offset) + idx, float(row[dim_idx]), 1.0),)
                    for dim_idx in range(arr.shape[1])
                ),
            )
            for idx, row in enumerate(arr)
        ]
        return cls(actions, chunk_id=resolved_id)

    @classmethod
    def zeros(cls, shape: tuple[int, int], *, chunk_id: int | None = None) -> TraceableActionChunk:
        arr = np.zeros(shape, dtype=float)
        resolved_id = int(next(cls._id_counter) if chunk_id is None else chunk_id)
        return cls([TraceableAction(row) for row in arr], chunk_id=resolved_id)

    @classmethod
    def mean(cls, *chunks: TraceableActionChunk, chunk_id: int | None = None) -> TraceableActionChunk:
        if not chunks:
            raise ValueError("at least one chunk is required")
        weights = [1.0 / len(chunks)] * len(chunks)
        return cls.weighted_sum(zip(weights, chunks, strict=True), chunk_id=chunk_id)

    @classmethod
    def weighted_sum(
        cls,
        weighted_chunks: Iterable[tuple[float | np.ndarray, TraceableActionChunk]],
        *,
        chunk_id: int | None = None,
    ) -> TraceableActionChunk:
        pairs = list(weighted_chunks)
        if not pairs:
            raise ValueError("at least one weighted chunk is required")
        length = len(pairs[0][1])
        if any(len(chunk) != length for _, chunk in pairs):
            raise ValueError("all chunks must have the same length")
        weighted_pairs = [(_normalize_step_weights(weight, length), chunk) for weight, chunk in pairs]
        actions = []
        for idx in range(length):
            value = sum(weights[idx] * chunk[idx].value for weights, chunk in weighted_pairs)
            value_source = _merge_value_sources(
                _scale_value_sources(chunk[idx].value_source, weights[idx]) for weights, chunk in weighted_pairs
            )
            actions.append(TraceableAction(value, value_source))
        return cls(actions, chunk_id=chunk_id)

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        for action in self._actions:
            yield action.copy()

    @overload
    def __getitem__(self, idx: slice) -> TraceableActionChunk: ...

    @overload
    def __getitem__(self, idx: int) -> TraceableAction: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TraceableActionChunk(self._actions[idx], chunk_id=self.id)
        return self._actions[int(idx)].copy()

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, slice):
            replacement = coerce_traceable_chunk(value)
            positions = list(range(*idx.indices(len(self))))
            if len(positions) != len(replacement):
                raise ValueError(f"cannot assign {len(replacement)} actions to slice of length {len(positions)}")
            for position, action in zip(positions, replacement, strict=True):
                self._actions[position] = action.copy()
            return
        self._actions[int(idx)] = coerce_traceable_action(value)

    def popleft(self) -> TraceableAction:
        if not self._actions:
            raise IndexError("pop from an empty TraceableActionChunk")
        return self._actions.pop(0).copy()

    def __add__(self, other) -> TraceableActionChunk:
        other_chunk = coerce_traceable_chunk(other)
        if len(self) != len(other_chunk):
            raise ValueError("chunks must have the same length")
        actions = []
        for left, right in zip(self, other_chunk, strict=True):
            actions.append(
                TraceableAction(
                    left.value + right.value,
                    _merge_value_sources((left.value_source, right.value_source)),
                )
            )
        return TraceableActionChunk(actions)

    def __radd__(self, other) -> TraceableActionChunk:
        return self.__add__(other)

    def as_array(self) -> np.ndarray:
        return np.asarray([action.value for action in self._actions], dtype=float)

    def value_sources(self) -> list[ActionValueSources]:
        return [action.value_source for action in self._actions]

    def format_source_map(self, *, dim: int = 0, cell_width: int = 3) -> str:
        if not self._actions:
            return ""
        dim = int(dim)
        cell_width = max(1, int(cell_width))
        if dim < 0 or dim >= self._actions[0].value.size:
            raise IndexError(f"dim must be in [0, {self._actions[0].value.size}), got {dim}")
        per_step_sources = [action.value_source[dim] for action in self._actions]
        chunk_ids = []
        seen_chunk_ids = set()
        for sources in per_step_sources:
            for source in sources:
                if source.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(source.chunk_id)
                chunk_ids.append(source.chunk_id)
        if chunk_ids:
            cell_width = max(cell_width, *(len(str(chunk_id % 10)) for chunk_id in chunk_ids))
        lines = []
        for chunk_id in chunk_ids:
            cells = []
            for sources in per_step_sources:
                text = str(chunk_id % 10) if any(source.chunk_id == chunk_id for source in sources) else ""
                cells.append(text.center(cell_width))
            lines.append("".join(cells))
        lines.append("".join("□".center(cell_width) for _ in self._actions))
        return "\n".join(lines)

    def print_source_map(self, *, dim: int = 0, cell_width: int = 3) -> None:
        print(self.format_source_map(dim=dim, cell_width=cell_width))


def _normalize_value_sources(value_source: ActionValueSources, dim: int) -> ActionValueSources:
    normalized = tuple(
        tuple(_normalize_value_source_item(source_item) for source_item in dim_sources)
        for dim_sources in value_source
    )
    if len(normalized) != int(dim):
        raise ValueError(f"value_source must have one entry per action value: expected {dim}, got {len(normalized)}")
    return normalized


def _normalize_value_source_item(source_item) -> ActionValueSourceItem:
    if isinstance(source_item, ActionValueSourceItem):
        return source_item
    fields = tuple(source_item)
    if len(fields) == 3:
        chunk_id, step_idx, value = fields
        weight = 1.0
    elif len(fields) == 4:
        chunk_id, step_idx, value, weight = fields
    else:
        raise ValueError(
            "value source item must have 3 or 4 fields: chunk_id, step_index, value[, weight]; "
            f"got {len(fields)}"
        )
    return ActionValueSourceItem(int(chunk_id), int(step_idx), float(value), float(weight))


def _scale_value_sources(value_source: ActionValueSources, weight: float) -> ActionValueSources:
    scale = float(weight)
    return tuple(
        tuple(
            ActionValueSourceItem(
                source_item.chunk_id,
                source_item.step_index,
                source_item.value,
                source_item.weight * scale,
            )
            for source_item in dim_sources
        )
        for dim_sources in value_source
    )


def _normalize_step_weights(weight: float | np.ndarray, length: int) -> np.ndarray:
    weights = np.asarray(weight, dtype=float)
    if weights.ndim == 0:
        return np.full(length, float(weights), dtype=float)
    if weights.shape != (length,):
        raise ValueError(f"weight array must have shape [{length}], got {weights.shape}")
    return weights.copy()


def _merge_value_sources(sources: Iterable[ActionValueSources]) -> ActionValueSources:
    source_list = [tuple(source) for source in sources]
    if not source_list:
        return ()
    dim = len(source_list[0])
    if any(len(source) != dim for source in source_list):
        raise ValueError("all value sources must have the same action dimension")
    return tuple(tuple(dim_source for source in source_list for dim_source in source[dim_idx]) for dim_idx in range(dim))


def coerce_traceable_action(value) -> TraceableAction:
    if isinstance(value, TraceableAction):
        return value.copy()
    arr = np.asarray(value, dtype=float)
    return TraceableAction(arr, tuple(() for _ in range(arr.size)))


def coerce_traceable_chunk(value) -> TraceableActionChunk:
    if isinstance(value, TraceableActionChunk):
        return value
    arr = np.asarray(value, dtype=float)
    return TraceableActionChunk([TraceableAction(row) for row in arr])


def __getattr__(name: str):
    if name == "StreamActionBuffer":
        from action_buffers import TraceableStreamActionBuffer

        class StreamActionBuffer(TraceableStreamActionBuffer):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("include_action_trace", False)
                super().__init__(*args, **kwargs)

        return StreamActionBuffer
    raise AttributeError(name)
