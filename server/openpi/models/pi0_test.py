import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config
from openpi.models.pi0 import build_shared_obs_attention_mask_and_position_ids


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_shared_obs_suffix_can_only_attend_matching_state_owner():
    prefix_pad = jax.numpy.array([[1, 1, 1, 1, 1]], dtype=bool)
    prefix_attn = jax.numpy.array([[0, 0, 0, 0, 0]], dtype=bool)
    # Two shared prefix tokens + three state tokens owned by offsets 0/1/2.
    prefix_owner = jax.numpy.array([[-1, -1, 0, 1, 2]], dtype=jax.numpy.int32)

    suffix_pad = jax.numpy.array([[1, 1]], dtype=bool)
    suffix_attn = jax.numpy.array([[1, 0]], dtype=bool)
    offset_mask = jax.numpy.array([[1, 1, 1]], dtype=bool)

    attn_mask, _ = build_shared_obs_attention_mask_and_position_ids(
        prefix_pad_masks=prefix_pad,
        prefix_attn_masks=prefix_attn,
        prefix_state_owner=prefix_owner,
        suffix_pad_masks=suffix_pad,
        suffix_attn_masks=suffix_attn,
        num_offsets=3,
        offset_mask=offset_mask,
    )

    # First token in suffix offset=1 block.
    suffix_start = prefix_pad.shape[1]
    suffix_len = suffix_pad.shape[1]
    query_idx = suffix_start + suffix_len
    # Shared keys.
    assert bool(attn_mask[0, query_idx, 0])
    assert bool(attn_mask[0, query_idx, 1])
    # Only matching owner=1 state token is visible.
    assert not bool(attn_mask[0, query_idx, 2])
    assert bool(attn_mask[0, query_idx, 3])
    assert not bool(attn_mask[0, query_idx, 4])


def test_shared_obs_offset_mask_applies_to_prefix_state_tokens():
    prefix_pad = jax.numpy.array([[1, 1, 1, 1, 1]], dtype=bool)
    prefix_attn = jax.numpy.array([[0, 0, 0, 0, 0]], dtype=bool)
    prefix_owner = jax.numpy.array([[-1, -1, 0, 1, 2]], dtype=jax.numpy.int32)

    suffix_pad = jax.numpy.array([[1, 1]], dtype=bool)
    suffix_attn = jax.numpy.array([[1, 0]], dtype=bool)
    # Offset 1 is invalid.
    offset_mask = jax.numpy.array([[1, 0, 1]], dtype=bool)

    attn_mask, _ = build_shared_obs_attention_mask_and_position_ids(
        prefix_pad_masks=prefix_pad,
        prefix_attn_masks=prefix_attn,
        prefix_state_owner=prefix_owner,
        suffix_pad_masks=suffix_pad,
        suffix_attn_masks=suffix_attn,
        num_offsets=3,
        offset_mask=offset_mask,
    )

    invalid_state_idx = 3
    # Invalid state token is masked as both key and query.
    assert not bool(attn_mask[0, :, invalid_state_idx].any())
    assert not bool(attn_mask[0, invalid_state_idx, :].any())


def test_shared_obs_positions_align_state_offsets_to_same_anchor():
    # Prefix layout (all valid):
    # 0..2 shared tokens, then owner0(2 tokens), owner1(3 tokens), owner2(1 token), then one shared token.
    prefix_pad = jax.numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    prefix_attn = jax.numpy.array([[0] * 10], dtype=bool)
    prefix_owner = jax.numpy.array([[-1, -1, -1, 0, 0, 1, 1, 1, 2, -1]], dtype=jax.numpy.int32)

    suffix_pad = jax.numpy.array([[1, 1]], dtype=bool)
    suffix_attn = jax.numpy.array([[1, 0]], dtype=bool)
    offset_mask = jax.numpy.array([[1, 1, 1]], dtype=bool)

    _, positions = build_shared_obs_attention_mask_and_position_ids(
        prefix_pad_masks=prefix_pad,
        prefix_attn_masks=prefix_attn,
        prefix_state_owner=prefix_owner,
        suffix_pad_masks=suffix_pad,
        suffix_attn_masks=suffix_attn,
        num_offsets=3,
        offset_mask=offset_mask,
    )

    prefix_pos = positions[0, :10]
    # owner0 starts at shared_count(=4, including the trailing shared token)
    assert int(prefix_pos[3]) == 4
    # owner1 and owner2 also start at the same anchor 4.
    assert int(prefix_pos[5]) == 4
    assert int(prefix_pos[8]) == 4
    # Local ranks per owner still increase from the same anchor.
    assert int(prefix_pos[4]) == 5
    assert int(prefix_pos[6]) == 5
    assert int(prefix_pos[7]) == 6
