from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class SharedObservationVLASHDataset(LeRobotDataset):
    """LeRobot dataset variant for VLASH shared-observation training.

    For each base index `t`, this dataset returns one shared observation with multiple
    offset branches:
      - observation.state: [O, state_dim]
      - action: [O, chunk_size, action_dim]
      - action_is_pad: [O, chunk_size]
    where O is dynamic per sample: O = max_valid_offset + 1.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        max_delay_steps: int = 0,
    ):
        self.max_delay_steps = max_delay_steps
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )

    def _get_query_indices_for_offset(
        self, idx: int, ep_idx: int, offset: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.BoolTensor]]:
        ep_start = self.episode_data_index["from"][ep_idx].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()

        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}

        for key, delta_idx in self.delta_indices.items():
            query_indices[key] = [max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx]
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        return query_indices, padding

    def __getitem__(self, idx: int) -> dict:
        ep_idx = self.hf_dataset[idx]["episode_index"].item()
        ep_start = self.episode_data_index["from"][ep_idx].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()

        if self.delta_indices is None or "action" not in self.delta_indices:
            raise ValueError("SharedObservationVLASHDataset requires delta_indices with 'action' key.")

        max_delta = self.delta_indices["action"][-1]
        max_valid_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        num_offsets = max_valid_offset + 1

        # Get offset=0 sample as shared prefix/base fields.
        base_item = super().__getitem__(idx)

        result = {}
        for key in base_item:
            if key.startswith("observation.images.") or key in ("task", "episode_index"):
                result[key] = base_item[key]

        base_state = base_item["observation.state"]
        base_action = base_item["action"]
        states = []
        actions = []
        action_is_pads = []

        for offset in range(num_offsets):
            if offset == 0:
                state = base_state
            else:
                prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))
                prev_action = self.hf_dataset[prev_idx]["action"]
                if base_state.dim() != 1 or prev_action.dim() != 1:
                    raise ValueError("Only 1D state/action is supported.")
                if base_state.shape[0] != prev_action.shape[0]:
                    raise ValueError(
                        f"state_dim ({base_state.shape[0]}) must equal action_dim ({prev_action.shape[0]})."
                    )
                state = prev_action

            query_indices, padding = self._get_query_indices_for_offset(idx, ep_idx, offset)
            query_result = self._query_hf_dataset(query_indices)
            action = query_result["action"]
            action_is_pad = padding["action_is_pad"]

            states.append(state)
            actions.append(action)
            action_is_pads.append(action_is_pad)

        result["observation.state"] = torch.stack(states, dim=0)
        result["action"] = torch.stack(actions, dim=0)
        result["action_is_pad"] = torch.stack(action_is_pads, dim=0)
        result["num_offsets"] = num_offsets
        return result


def shared_observation_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-offset samples to per-batch max offset and create offset_mask."""
    def _stack(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: _stack([item[k] for item in items]) for k in first}
        return np.stack([np.asarray(x) for x in items], axis=0)

    def _num_offsets(item: dict) -> int:
        if "num_offsets" in item:
            return int(item["num_offsets"])
        if "state" in item:
            return int(np.asarray(item["state"]).shape[0])
        if "observation.state" in item:
            return int(np.asarray(item["observation.state"]).shape[0])
        raise KeyError("Neither 'num_offsets' nor state tensor found in sample.")

    max_offsets = max(_num_offsets(item) for item in batch)

    # Support both raw LeRobot keys and transformed policy keys.
    state_key = "state" if "state" in batch[0] else "observation.state"
    action_key = "actions" if "actions" in batch[0] else "action"
    per_offset_keys = {state_key, action_key, "action_is_pad"}
    shared_keys = [k for k in batch[0].keys() if k not in per_offset_keys and k != "num_offsets"]

    shared_batch = [{k: item[k] for k in shared_keys} for item in batch]
    result = _stack(shared_batch)

    def _ensure_offset_dim(arr: np.ndarray, *, min_ndim: int, key: str) -> np.ndarray:
        # VLASH expects leading offset dimension. If it is missing (e.g. after an accidental squeeze),
        # recover by adding a singleton offset axis.
        if arr.ndim == min_ndim - 1:
            return np.expand_dims(arr, axis=0)
        if arr.ndim != min_ndim:
            raise ValueError(f"Unexpected ndim for {key}: got {arr.ndim}, expected {min_ndim - 1} or {min_ndim}.")
        return arr

    state_arrays = [_ensure_offset_dim(np.asarray(item[state_key]), min_ndim=2, key=state_key) for item in batch]
    action_arrays = [_ensure_offset_dim(np.asarray(item[action_key]), min_ndim=3, key=action_key) for item in batch]
    # Per-sample action_is_pad should be [offset, horizon]. If squeeze dropped the
    # offset axis for offset=1, recover [horizon] -> [1, horizon].
    pad_arrays = [_ensure_offset_dim(np.asarray(item["action_is_pad"]), min_ndim=2, key="action_is_pad") for item in batch]

    batch_size = len(batch)
    state_shape = state_arrays[0].shape[1:]
    action_shape = action_arrays[0].shape[1:]
    pad_shape = pad_arrays[0].shape[1:]

    for i, (s, a, p) in enumerate(zip(state_arrays, action_arrays, pad_arrays, strict=False)):
        if s.shape[1:] != state_shape:
            raise ValueError(f"Inconsistent {state_key} shape at batch index {i}: got {s.shape}, expected (*, {state_shape}).")
        if a.shape[1:] != action_shape:
            raise ValueError(
                f"Inconsistent {action_key} shape at batch index {i}: got {a.shape}, expected (*, {action_shape})."
            )
        if p.shape[1:] != pad_shape:
            raise ValueError(
                f"Inconsistent action_is_pad shape at batch index {i}: got {p.shape}, expected (*, {pad_shape})."
            )

    state_ref = state_arrays[0]
    action_ref = action_arrays[0]

    padded_states = np.zeros((batch_size, max_offsets, *state_shape), dtype=state_ref.dtype)
    padded_actions = np.zeros((batch_size, max_offsets, *action_shape), dtype=action_ref.dtype)
    padded_action_is_pad = np.ones((batch_size, max_offsets, *pad_shape), dtype=np.bool_)
    offset_mask = np.zeros((batch_size, max_offsets), dtype=np.bool_)

    for i, item in enumerate(batch):
        n = _num_offsets(item)
        state_arr = state_arrays[i]
        action_arr = action_arrays[i]
        pad_arr = pad_arrays[i]
        if state_arr.shape[0] != n or action_arr.shape[0] != n or pad_arr.shape[0] != n:
            raise ValueError(
                f"Inconsistent num_offsets at batch index {i}: num_offsets={n}, "
                f"{state_key}.shape[0]={state_arr.shape[0]}, {action_key}.shape[0]={action_arr.shape[0]}, "
                f"action_is_pad.shape[0]={pad_arr.shape[0]}."
            )
        padded_states[i, :n] = state_arr
        padded_actions[i, :n] = action_arr
        padded_action_is_pad[i, :n] = pad_arr
        offset_mask[i, :n] = True

    result[state_key] = padded_states
    result[action_key] = padded_actions
    result["action_is_pad"] = padded_action_is_pad
    result["offset_mask"] = offset_mask
    # Keep max_offsets batch-shaped to be compatible with JAX sharding.
    result["max_offsets"] = np.full((batch_size,), max_offsets, dtype=np.int32)
    return result
