"""π0/π0.5 视觉语言动作模型。

图像和语言组成 prefix，机器人状态、带噪 action chunk 与 Flow Matching 时间组成
suffix。模型学习从噪声到真实动作的速度场；推理时从高斯噪声出发，多步积分得到
完整的 ``[action_horizon, action_dim]`` 动作序列。
"""

import logging
import numbers

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def build_shared_obs_attention_mask_and_position_ids(
    prefix_pad_masks: at.Bool[at.Array, "b p"],
    prefix_attn_masks: at.Bool[at.Array, "b p"] | at.Bool[at.Array, " p"],
    suffix_pad_masks: at.Bool[at.Array, "b s"],
    suffix_attn_masks: at.Bool[at.Array, "b s"] | at.Bool[at.Array, " s"],
    num_offsets: int,
    offset_mask: at.Bool[at.Array, "b o"],
    prefix_state_owner: at.Int[at.Array, "b p"] | None = None,
) -> tuple[at.Bool[at.Array, "b t t"], at.Int[at.Array, "b t"]]:
    """Build shared-observation attention mask and position ids.

    The final sequence layout is:
      [prefix, suffix(offset_0), suffix(offset_1), ..., suffix(offset_{O-1})]

    Attention rules:
      1. Base block-causal structure comes from `make_attn_mask`.
      2. Suffix tokens cannot attend to suffix tokens from other offsets.
      3. Invalid offsets (offset_mask=False) are fully masked as query+key.
      4. If `prefix_state_owner` is provided, suffix(offset_i) can only attend to
         prefix state tokens owned by i (and shared prefix tokens with owner=-1).
      5. Invalid-offset prefix state tokens are masked as query+key as well.
    """
    if num_offsets <= 0:
        raise ValueError(f"num_offsets must be positive, got {num_offsets}")

    batch_size, prefix_len = prefix_pad_masks.shape
    _, suffix_len = suffix_pad_masks.shape
    if offset_mask.shape != (batch_size, num_offsets):
        raise ValueError(
            f"offset_mask shape must be (b, num_offsets)=({batch_size}, {num_offsets}), "
            f"got {offset_mask.shape}"
        )

    if prefix_attn_masks.ndim == 1:
        prefix_attn_masks = jnp.broadcast_to(prefix_attn_masks[None, :], (batch_size, prefix_len))
    if suffix_attn_masks.ndim == 1:
        suffix_attn_masks = jnp.broadcast_to(suffix_attn_masks[None, :], (batch_size, suffix_len))
    if prefix_state_owner is not None and prefix_state_owner.shape != (batch_size, prefix_len):
        raise ValueError(
            f"prefix_state_owner shape must be (b, prefix_len)=({batch_size}, {prefix_len}), "
            f"got {prefix_state_owner.shape}"
        )

    suffix_pad_tiled = einops.repeat(suffix_pad_masks, "b s -> b (o s)", o=num_offsets)
    suffix_attn_tiled = einops.repeat(suffix_attn_masks, "b s -> b (o s)", o=num_offsets)

    full_input_mask = jnp.concatenate([prefix_pad_masks, suffix_pad_tiled], axis=1)
    full_ar_mask = jnp.concatenate([prefix_attn_masks, suffix_attn_tiled], axis=1)

    attn_mask = make_attn_mask(full_input_mask, full_ar_mask)

    suffix_start = prefix_len
    total_suffix_len = num_offsets * suffix_len
    suffix_positions = jnp.arange(total_suffix_len)
    offset_ids = suffix_positions // suffix_len

    if prefix_state_owner is not None:
        prefix_state_owner = prefix_state_owner.astype(jnp.int32)
        prefix_is_state = prefix_state_owner >= 0
        # Apply offset validity to state tokens in the prefix (query and key).
        owner_index = jnp.clip(prefix_state_owner, 0, num_offsets - 1)
        owner_is_valid = jnp.take_along_axis(offset_mask, owner_index, axis=1)
        prefix_token_valid = jnp.logical_or(~prefix_is_state, owner_is_valid)
        attn_mask = attn_mask.at[:, :, :prefix_len].set(
            jnp.logical_and(attn_mask[:, :, :prefix_len], prefix_token_valid[:, None, :])
        )
        attn_mask = attn_mask.at[:, :prefix_len, :].set(
            jnp.logical_and(attn_mask[:, :prefix_len, :], prefix_token_valid[:, :, None])
        )

        # Restrict prefix-prefix mixing to block state information leakage across offsets:
        # - non-state queries cannot read any state keys;
        # - state_i queries can only read state_i keys (plus shared keys).
        same_owner = prefix_state_owner[:, :, None] == prefix_state_owner[:, None, :]
        q_is_state = prefix_is_state[:, :, None]
        k_is_state = prefix_is_state[:, None, :]
        prefix_prefix_allowed = jnp.logical_or(~k_is_state, jnp.logical_and(q_is_state, same_owner))
        attn_mask = attn_mask.at[:, :prefix_len, :prefix_len].set(
            jnp.logical_and(attn_mask[:, :prefix_len, :prefix_len], prefix_prefix_allowed)
        )

        # For each suffix offset branch, allow reading only shared prefix tokens and
        # the matching state-owner prefix tokens.
        suffix_to_prefix_allowed = jnp.logical_or(
            ~prefix_is_state[:, None, :],
            prefix_state_owner[:, None, :] == offset_ids[None, :, None],
        )
        attn_mask = attn_mask.at[:, suffix_start:, :prefix_len].set(
            jnp.logical_and(attn_mask[:, suffix_start:, :prefix_len], suffix_to_prefix_allowed)
        )

    # Keep only within-branch suffix-to-suffix attention.
    same_offset = offset_ids[:, None] == offset_ids[None, :]
    attn_mask = attn_mask.at[:, suffix_start:, suffix_start:].set(
        jnp.logical_and(attn_mask[:, suffix_start:, suffix_start:], same_offset[None, :, :])
    )

    # Mask invalid offsets as both query and key.
    offset_validity = einops.repeat(offset_mask, "b o -> b (o s)", s=suffix_len)
    attn_mask = attn_mask.at[:, suffix_start:, :].set(
        jnp.logical_and(attn_mask[:, suffix_start:, :], offset_validity[:, :, None])
    )
    attn_mask = attn_mask.at[:, :, suffix_start:].set(
        jnp.logical_and(attn_mask[:, :, suffix_start:], offset_validity[:, None, :])
    )

    total_len = prefix_len + total_suffix_len
    positions = jnp.zeros((batch_size, total_len), dtype=jnp.int32)

    if prefix_state_owner is None:
        prefix_pos = jnp.cumsum(prefix_pad_masks.astype(jnp.int32), axis=1) - 1
    else:
        # Shared (non-state) tokens keep one global timeline.
        shared_mask = jnp.logical_and(prefix_pad_masks, prefix_state_owner < 0)
        shared_pos = jnp.cumsum(shared_mask.astype(jnp.int32), axis=1) - 1
        shared_count = jnp.sum(shared_mask.astype(jnp.int32), axis=1)

        # State tokens get per-owner local positions with a common start anchor.
        # This aligns each offset's state segment to the same relative start.
        prefix_pos = shared_pos
        state_mask = jnp.logical_and(prefix_pad_masks, prefix_state_owner >= 0)
        for owner_id in range(num_offsets):
            owner_mask = jnp.logical_and(state_mask, prefix_state_owner == owner_id)
            owner_local_rank = jnp.cumsum(owner_mask.astype(jnp.int32), axis=1) - 1
            owner_pos = shared_count[:, None] + owner_local_rank
            prefix_pos = jnp.where(owner_mask, owner_pos, prefix_pos)

        # Keep padded tokens at 0; attention mask already prevents attending to padding.
        prefix_pos = jnp.where(prefix_pad_masks, prefix_pos, 0)

    positions = positions.at[:, :prefix_len].set(prefix_pos)

    # Suffix starts after the largest valid prefix position to avoid overlap.
    last_prefix_pos = jnp.max(jnp.where(prefix_pad_masks, prefix_pos, -1), axis=1)
    suffix_pos_base = jnp.cumsum(suffix_pad_masks.astype(jnp.int32), axis=1)
    suffix_pos_tiled = einops.repeat(suffix_pos_base, "b s -> b (o s)", o=num_offsets)
    positions = positions.at[:, prefix_len:].set(last_prefix_pos[:, None] + suffix_pos_tiled)

    return attn_mask, positions


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@at.typecheck
def build_legato_schedule(
    horizon: int,
    inference_delay: at.Int[at.Array, "*b"] | int,
    ramp_down: at.Int[at.Array, "*b"] | int,
) -> at.Float[at.Array, "*b h"]:
    """Build Legato continuation schedule omega in [0, 1] with shape [..., H]."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    delay = jnp.clip(jnp.asarray(inference_delay, dtype=jnp.int32), 0, horizon)
    max_ramp = jnp.maximum(horizon - delay, 0)
    ramp = jnp.clip(jnp.asarray(ramp_down, dtype=jnp.int32), 0, max_ramp)

    pos = jnp.arange(horizon, dtype=jnp.int32)
    full_mask = pos < delay[..., None]
    ramp_mask = jnp.logical_and(pos >= delay[..., None], pos < (delay + ramp)[..., None])
    ramp_denom = jnp.maximum(ramp[..., None], 1)
    ramp_progress = (pos - delay[..., None] + 1).astype(jnp.float32) / ramp_denom.astype(jnp.float32)
    ramp_values = jnp.clip(1.0 - ramp_progress, 0.0, 1.0)

    omega = jnp.where(full_mask, 1.0, 0.0)
    omega = jnp.where(ramp_mask, ramp_values, omega)
    return omega.astype(jnp.float32)


class Pi0(_model.BaseModel):
    """SigLIP 视觉编码器、Gemma 语言骨干和 Action Expert 的组合模型。"""

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.action_condition_on_omega = config.action_condition_on_omega
        self.snapflow_enabled = config.enable_snapflow
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        action_in_dim = config.action_dim + 1 if self.action_condition_on_omega else config.action_dim
        self.action_in_proj = nnx.Linear(action_in_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.snapflow_enabled:
            self.target_time_mlp_1 = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.target_time_mlp_2 = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.target_time_mlp_2.kernel.value = jnp.zeros_like(self.target_time_mlp_2.kernel.value)
            self.target_time_mlp_2.bias.value = jnp.zeros_like(self.target_time_mlp_2.bias.value)

        self.paper_rl_token_enabled = config.paper_rl_token or config.paper_rl_actor_critic
        self.paper_rl_actor_critic = config.paper_rl_actor_critic
        self.paper_rl_actor_delta_scale = config.paper_rl_actor_delta_scale
        self.paper_rl_reference_num_steps = config.paper_rl_reference_num_steps
        self.paper_rl_inference_num_steps = config.paper_rl_inference_num_steps
        self.paper_rl_use_decoded_token = config.paper_rl_use_decoded_token
        self.paper_rl_actor_bc_weight = config.paper_rl_actor_bc_weight
        self.paper_rl_delta_l2_weight = config.paper_rl_delta_l2_weight
        self.paper_rl_critic_weight = config.paper_rl_critic_weight
        self.paper_rl_recon_weight = config.paper_rl_recon_weight
        self.paper_rl_actor_pg_weight = config.paper_rl_actor_pg_weight
        self.paper_rl_critic_td_weight = config.paper_rl_critic_td_weight
        self.paper_rl_logprob_std = config.paper_rl_logprob_std
        self.paper_rl_discount = config.paper_rl_discount
        self.paper_rl_advantage_clip = config.paper_rl_advantage_clip
        self.dsrl_steering_enabled = config.dsrl_steering
        self.dsrl_noise_scale = config.dsrl_noise_scale
        self.dsrl_train_num_steps = config.dsrl_train_num_steps
        self.dsrl_inference_num_steps = config.dsrl_inference_num_steps
        self.dsrl_noise_l2_weight = config.dsrl_noise_l2_weight
        if self.paper_rl_token_enabled:
            self.paper_rl_token = nnx.Param(
                jax.random.normal(rngs.params(), (1, 1, paligemma_config.width), dtype=jnp.float32) * 0.02
            )
            self.paper_rl_encoder = nnx.Linear(paligemma_config.width, config.paper_rl_latent_dim, rngs=rngs)
            self.paper_rl_decoder = nnx.Linear(config.paper_rl_latent_dim, paligemma_config.width, rngs=rngs)

        if self.dsrl_steering_enabled:
            action_features = config.action_horizon * config.action_dim
            hidden_dim = config.dsrl_hidden_dim
            self.dsrl_actor_in = nnx.Linear(paligemma_config.width, hidden_dim, rngs=rngs)
            self.dsrl_actor_hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.dsrl_actor_out = nnx.Linear(hidden_dim, action_features, rngs=rngs)

        if self.paper_rl_actor_critic:
            action_features = config.action_horizon * config.action_dim
            actor_input_dim = action_features + config.paper_rl_latent_dim
            hidden_dim = config.paper_rl_mlp_hidden_dim
            self.paper_rl_actor_in = nnx.Linear(actor_input_dim, hidden_dim, rngs=rngs)
            self.paper_rl_actor_hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.paper_rl_actor_out = nnx.Linear(hidden_dim, action_features, rngs=rngs)
            self.paper_rl_critic_in = nnx.Linear(actor_input_dim, hidden_dim, rngs=rngs)
            self.paper_rl_critic_hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.paper_rl_critic_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _paper_rl_decode_token_for_action(self, obs: _model.Observation) -> bool:
        return self.paper_rl_token_enabled and self.paper_rl_use_decoded_token

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
        paper_rl_token_override: at.Float[at.Array, "b emb"] | at.Float[at.Array, "b 1 emb"] | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if self.paper_rl_token_enabled:
            batch_size = tokens[0].shape[0]
            if paper_rl_token_override is None:
                rl_token = jnp.asarray(self.paper_rl_token.value, dtype=tokens[0].dtype)
                rl_token = jnp.broadcast_to(rl_token, (batch_size, 1, rl_token.shape[-1]))
            else:
                rl_token = jnp.asarray(paper_rl_token_override, dtype=tokens[0].dtype)
                if rl_token.ndim == 2:
                    rl_token = rl_token[:, None, :]
            tokens.append(rl_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            ar_mask += [False]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def _align_prefix_state_owner(
        self, obs: _model.Observation, prefix_mask: at.Bool[at.Array, "b p"]
    ) -> at.Int[at.Array, "b p"] | None:
        """Align tokenized prompt state-owner map with full prefix layout [images + language]."""
        if obs.tokenized_prompt_state_owner is None:
            return None
        if obs.tokenized_prompt is None:
            raise ValueError("tokenized_prompt_state_owner is present but tokenized_prompt is missing.")
        if obs.tokenized_prompt_state_owner.shape != obs.tokenized_prompt.shape:
            raise ValueError(
                "tokenized_prompt_state_owner must match tokenized_prompt shape, got "
                f"{obs.tokenized_prompt_state_owner.shape} vs {obs.tokenized_prompt.shape}"
            )

        batch_size, prefix_len = prefix_mask.shape
        _, prompt_len = obs.tokenized_prompt_state_owner.shape
        if prompt_len > prefix_len:
            raise ValueError(f"Prompt token length {prompt_len} exceeds prefix length {prefix_len}.")

        prefix_state_owner = jnp.full((batch_size, prefix_len), -1, dtype=obs.tokenized_prompt_state_owner.dtype)
        prefix_state_owner = prefix_state_owner.at[:, prefix_len - prompt_len :].set(obs.tokenized_prompt_state_owner)
        return prefix_state_owner

    @at.typecheck
    def _target_time_embedding(self, target_time: at.Float[at.Array, " b"]) -> at.Float[at.Array, "b emb"]:
        if not self.snapflow_enabled:
            raise ValueError("SnapFlow target-time embedding requested but enable_snapflow=False.")
        target_emb = posemb_sincos(target_time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        target_emb = self.target_time_mlp_1(target_emb)
        target_emb = nnx.swish(target_emb)
        target_emb = self.target_time_mlp_2(target_emb)
        return target_emb

    @at.typecheck
    def embed_suffix(
        self,
        obs_or_state: _model.Observation | at.Float[at.Array, "b s"],
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        target_time: at.Float[at.Array, " b"] | None = None,
        omega: at.Float[at.Array, "b h"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if hasattr(obs_or_state, "state"):
            state = obs_or_state.state
        else:
            state = obs_or_state

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        if target_time is None:
            target_time = timestep
        if self.action_condition_on_omega:
            if omega is None:
                omega = jnp.zeros(noisy_actions.shape[:-1], dtype=noisy_actions.dtype)
            elif omega.shape != noisy_actions.shape[:-1]:
                raise ValueError(
                    f"omega shape must match noisy_actions leading dims {noisy_actions.shape[:-1]}, got {omega.shape}"
                )
            action_inputs = jnp.concatenate([noisy_actions, omega[..., None]], axis=-1)
        else:
            action_inputs = noisy_actions
        action_tokens = self.action_in_proj(action_inputs)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.snapflow_enabled:
            time_emb = time_emb + self._target_time_embedding(target_time)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _prefix_summary(self, obs: _model.Observation) -> at.Float[at.Array, "b emb"]:
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = self._embed_prefix_for_action(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        last_index = jnp.maximum(jnp.sum(prefix_mask.astype(jnp.int32), axis=-1) - 1, 0)
        return prefix_out[jnp.arange(prefix_out.shape[0]), last_index]

    def _dsrl_predict_noise(self, observation: _model.Observation) -> _model.Actions:
        if not self.dsrl_steering_enabled:
            raise ValueError("DSRL steering network is not enabled for this Pi0 model.")
        features = jax.lax.stop_gradient(self._prefix_summary(observation))
        hidden = self.dsrl_actor_in(features)
        hidden = nnx.swish(hidden)
        hidden = self.dsrl_actor_hidden(hidden)
        hidden = nnx.swish(hidden)
        noise_flat = self.dsrl_actor_out(hidden)
        noise = jnp.reshape(noise_flat, (observation.state.shape[0], self.action_horizon, self.action_dim))
        return self.dsrl_noise_scale * jnp.tanh(noise)

    def _paper_rl_encode(self, rl_hidden: at.Float[at.Array, "b emb"]):
        rl_hidden = jnp.asarray(rl_hidden, dtype=jnp.float32)
        latent = self.paper_rl_encoder(rl_hidden)
        latent = nnx.swish(latent)
        reconstruction = self.paper_rl_decoder(latent)
        return latent, reconstruction

    def _paper_rl_apply_actor(
        self,
        reference_actions: _model.Actions,
        rl_latent: at.Float[at.Array, "b emb"],
    ):
        reference_actions = jnp.asarray(reference_actions, dtype=jnp.float32)
        batch_size = reference_actions.shape[0]
        reference_flat = jnp.reshape(reference_actions, (batch_size, -1))
        actor_inputs = jnp.concatenate([reference_flat, jnp.asarray(rl_latent, dtype=jnp.float32)], axis=-1)

        actor_hidden = self.paper_rl_actor_in(actor_inputs)
        actor_hidden = nnx.swish(actor_hidden)
        actor_hidden = self.paper_rl_actor_hidden(actor_hidden)
        actor_hidden = nnx.swish(actor_hidden)
        delta_flat = self.paper_rl_actor_out(actor_hidden)
        delta = self.paper_rl_actor_delta_scale * jnp.tanh(jnp.reshape(delta_flat, reference_actions.shape))
        corrected_actions = reference_actions + delta

        critic_hidden = self.paper_rl_critic_in(actor_inputs)
        critic_hidden = nnx.swish(critic_hidden)
        critic_hidden = self.paper_rl_critic_hidden(critic_hidden)
        critic_hidden = nnx.swish(critic_hidden)
        critic_value = jnp.squeeze(self.paper_rl_critic_out(critic_hidden), axis=-1)
        return corrected_actions, delta, critic_value

    def _embed_prefix_for_action(self, obs: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        paper_rl_info = None
        if not self._paper_rl_decode_token_for_action(obs):
            return prefix_tokens, prefix_mask, prefix_ar_mask, paper_rl_info

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        rl_hidden = prefix_out[:, -1, :]
        rl_latent, rl_reconstruction = self._paper_rl_encode(rl_hidden)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            obs, paper_rl_token_override=rl_reconstruction
        )
        paper_rl_info = {
            "rl_hidden": rl_hidden,
            "rl_latent": rl_latent,
            "rl_reconstruction": rl_reconstruction,
        }
        return prefix_tokens, prefix_mask, prefix_ar_mask, paper_rl_info

    @at.typecheck
    def forward_shared_observation(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: at.Float[at.Array, "b o ah ad"],
        offset_mask: at.Bool[at.Array, "b o"],
        action_is_pad: at.Bool[at.Array, "b o ah"] | None = None,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b o ah"]:
        """Shared-observation training forward for async offset batches.

        This path computes prefix embeddings once, then runs all suffix offset branches
        in a single transformer pass using branch-isolated attention masks.
        """
        if actions.ndim != 4:
            raise ValueError(f"actions must be [b, o, ah, ad], got shape {actions.shape}")
        if observation.state.ndim != 3:
            raise ValueError(f"observation.state must be [b, o, s], got shape {observation.state.shape}")
        if offset_mask.ndim != 2:
            raise ValueError(f"offset_mask must be [b, o], got shape {offset_mask.shape}")

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size, num_offsets, _ = observation.state.shape
        if actions.shape[:2] != (batch_size, num_offsets):
            raise ValueError(
                f"actions leading dims must match state batch/offset dims {(batch_size, num_offsets)}, got {actions.shape[:2]}"
            )
        if offset_mask.shape != (batch_size, num_offsets):
            raise ValueError(
                f"offset_mask must have shape {(batch_size, num_offsets)}, got {offset_mask.shape}"
            )
        if action_is_pad is not None and action_is_pad.shape != (batch_size, num_offsets, self.action_horizon):
            raise ValueError(
                "action_is_pad must have shape "
                f"{(batch_size, num_offsets, self.action_horizon)}, got {action_is_pad.shape}"
            )

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size, num_offsets)) * 0.999 + 0.001

        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.pi05:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_state_owner = self._align_prefix_state_owner(observation, prefix_mask)

            states_flat = einops.rearrange(observation.state, "b o s -> (b o) s")
            x_t_flat = einops.rearrange(x_t, "b o ah ad -> (b o) ah ad")
            time_flat = einops.rearrange(time, "b o -> (b o)")
            suffix_tokens_flat, suffix_mask_flat, suffix_ar_flat, adarms_cond_flat = self.embed_suffix(
                states_flat, x_t_flat, time_flat
            )
            if adarms_cond_flat is None:
                raise ValueError("Expected adaRMS conditioning for pi05 shared forward.")

            suffix_len = suffix_tokens_flat.shape[1]
            suffix_tokens = einops.rearrange(
                suffix_tokens_flat, "(b o) s e -> b (o s) e", b=batch_size, o=num_offsets
            )
            suffix_pad = suffix_mask_flat[:batch_size]
            suffix_ar = jnp.broadcast_to(suffix_ar_flat[None, :], (batch_size, suffix_len))
            suffix_adarms_conds = einops.rearrange(adarms_cond_flat, "(b o) d -> b o d", b=batch_size, o=num_offsets)

            attn_mask, positions = build_shared_obs_attention_mask_and_position_ids(
                prefix_pad_masks=prefix_mask,
                prefix_attn_masks=prefix_ar_mask,
                prefix_state_owner=prefix_state_owner,
                suffix_pad_masks=suffix_pad,
                suffix_attn_masks=suffix_ar,
                num_offsets=num_offsets,
                offset_mask=offset_mask,
            )

            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, suffix_adarms_conds],
            )
            suffix_out = einops.rearrange(suffix_out, "b (o s) e -> b o s e", o=num_offsets, s=suffix_len)
            action_out = suffix_out[:, :, -self.action_horizon :]
            v_t = self.action_out_proj(action_out)

            losses = jnp.square(v_t - u_t)
            valid_steps = offset_mask[:, :, None]
            if action_is_pad is not None:
                valid_steps = jnp.logical_and(valid_steps, jnp.logical_not(action_is_pad))
            losses = losses * valid_steps[:, :, :, None]
            return jnp.mean(losses, axis=-1)

        # Shared prefix (images + language): computed once per batch element.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_state_owner = self._align_prefix_state_owner(observation, prefix_mask)

        # Flatten offset branches for suffix embedding.
        states_flat = einops.rearrange(observation.state, "b o s -> (b o) s")
        x_t_flat = einops.rearrange(x_t, "b o ah ad -> (b o) ah ad")
        time_flat = einops.rearrange(time, "b o -> (b o)")
        suffix_tokens_flat, suffix_pad_flat, suffix_ar_flat, adarms_cond = self.embed_suffix(states_flat, x_t_flat, time_flat)
        if adarms_cond is not None:
            raise NotImplementedError("Unexpected adaRMS condition in pi0 shared forward.")

        suffix_len = suffix_tokens_flat.shape[1]
        suffix_tokens = einops.rearrange(
            suffix_tokens_flat, "(b o) s e -> b (o s) e", b=batch_size, o=num_offsets
        )
        # Suffix masks have the same structure for each offset; keep one template per batch element.
        suffix_pad = suffix_pad_flat[:batch_size]
        suffix_ar = jnp.broadcast_to(suffix_ar_flat[None, :], (batch_size, suffix_len))

        attn_mask, positions = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_mask,
            prefix_attn_masks=prefix_ar_mask,
            prefix_state_owner=prefix_state_owner,
            suffix_pad_masks=suffix_pad,
            suffix_attn_masks=suffix_ar,
            num_offsets=num_offsets,
            offset_mask=offset_mask,
        )

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )

        suffix_out = einops.rearrange(
            suffix_out, "b (o s) e -> b o s e", o=num_offsets, s=suffix_len
        )
        action_out = suffix_out[:, :, -self.action_horizon :]
        v_t = self.action_out_proj(action_out)

        losses = jnp.square(v_t - u_t)
        valid_steps = offset_mask[:, :, None]
        if action_is_pad is not None:
            valid_steps = jnp.logical_and(valid_steps, jnp.logical_not(action_is_pad))
        losses = losses * valid_steps[:, :, :, None]
        return jnp.mean(losses, axis=-1)

    def compute_legato_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        num_denoising_steps: int = 5,
        delay_range: tuple[int, int] = (0, 10),
        ramp_range: tuple[int, int] = (0, 50),
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Legato loss on the base (non-shared-observation) training path."""
        preprocess_rng, noise_rng, time_rng, schedule_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]

        # Use uniform time sampling for Legato loss.
        noise = jax.random.normal(noise_rng, actions.shape)
        # time = jax.random.uniform(time_rng, batch_shape, minval=0.001, maxval=1.0)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        # Training-time randomized schedule conditioning (paper Appendix Table A.3):
        # d ~ Uni[0,10], r ~ Uni[0,50] by default.
        # Keep this always randomized during training for better inference-time generalization.
        delay_rng, ramp_rng = jax.random.split(schedule_rng)
        delay = jax.random.randint(delay_rng, batch_shape, minval=delay_range[0], maxval=delay_range[1] + 1)
        ramp = jax.random.randint(ramp_rng, batch_shape, minval=ramp_range[0], maxval=ramp_range[1] + 1)

        omega = build_legato_schedule(
            self.action_horizon,
            delay,
            ramp_down=ramp,
        )
        omega_expanded = omega[..., None]

        # Pseudocode-aligned construction:
        # eps_eff <- omega * A + (1-omega) * eps
        # Y_t     <- t * eps_eff + (1-t) * A
        # where t follows this codebase's time convention (1=noise, 0=action).
        eps_eff = omega_expanded * actions + (1.0 - omega_expanded) * noise
        y_t = time_expanded * eps_eff + (1.0 - time_expanded) * actions

        dt = -1.0 / float(num_denoising_steps)
        kappa = omega_expanded / dt

        # Keep masking behavior identical to the original compute_loss path.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            y_t,
            time,
            omega=omega if self.action_condition_on_omega else None,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # g-space supervision for Legato consistency:
        # g_pred = (1 - omega) * v_t - kappa * (Y_t - A)
        # g_tgt  = (1 - omega) * (epsilon - A)
        one_minus_omega = 1.0 - omega_expanded
        u_t = noise - actions
        g_pred = one_minus_omega * v_t - kappa * (y_t - actions)
        g_tgt = one_minus_omega * u_t

        # Mask out the near-singular omega≈1 boundary and normalize by valid ratio.
        omega_eps = 1e-4
        valid = (omega < (1.0 - omega_eps)).astype(v_t.dtype)
        err = jnp.mean(jnp.square(g_pred - g_tgt), axis=-1)
        valid_ratio = jnp.maximum(jnp.mean(valid), jnp.asarray(1e-8, dtype=v_t.dtype))
        return (err * valid) / valid_ratio

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """标准 Flow Matching 训练损失。

        ``x_t = t * noise + (1 - t) * actions`` 在噪声和真实动作之间插值，
        目标速度 ``u_t = noise - actions``；网络预测 ``v_t`` 并最小化二者均方差。
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def _velocity_from_prefix(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
        obs: _model.Observation,
        x_t: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        target_time: at.Float[at.Array, " b"],
        prediction_clamp: float = 20.0,
    ) -> at.Float[at.Array, "b ah ad"]:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs, x_t, timestep, target_time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.clip(v_t, -prediction_clamp, prediction_clamp)

    @at.typecheck
    def _sample_actions_one_step(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        noise: _model.Actions | None = None,
    ) -> _model.Actions:
        if not self.snapflow_enabled:
            raise ValueError("sample_actions_one_step requires enable_snapflow=True.")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        ones = jnp.ones(batch_size)
        zeros = jnp.zeros(batch_size)
        v_t = self._velocity_from_prefix(prefix_tokens, prefix_mask, prefix_ar_mask, observation, noise, ones, zeros)
        return noise - v_t

    @override
    def _sample_vla_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_rl_hidden: bool = False,
    ):
        """从高斯噪声开始，以 Euler 步进从 t=1 积分到 t=0。"""
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 图像和语言 prefix 在所有去噪步中保持不变，只计算一次并复用 KV cache。
        prefix_tokens, prefix_mask, prefix_ar_mask, paper_rl_info = self._embed_prefix_for_action(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        if self._paper_rl_decode_token_for_action(observation):
            if paper_rl_info is None:
                raise ValueError("Decoded RL token path expected paper_rl_info from _embed_prefix_for_action.")
            rl_hidden = paper_rl_info["rl_hidden"]
        elif self.paper_rl_token_enabled:
            rl_hidden = paper_rl_info["rl_hidden"] if paper_rl_info is not None else jax.lax.stop_gradient(prefix_out[:, -1, :])
        else:
            rl_hidden = None

        def step(carry):
            x_t, time = carry
            # suffix 随当前噪声动作和时间变化，每个去噪步都需要重新计算。
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # dt < 0，因此沿学习到的速度场从噪声端 t=1 走向动作端 t=0。
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        if isinstance(num_steps, numbers.Integral):
            def scan_step(carry, _):
                return step(carry), None

            (x_0, _), _ = jax.lax.scan(scan_step, (noise, 1.0), None, length=int(num_steps))
        else:
            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if return_rl_hidden:
            return x_0, rl_hidden
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """统一采样入口，根据配置选择标准多步、SnapFlow 或 RL 修正路径。"""
        observation = _model.preprocess_observation(None, observation, train=False)
        if num_steps is None and self.paper_rl_inference_num_steps is not None:
            num_steps = self.paper_rl_inference_num_steps
        if num_steps is None:
            num_steps = 10
        if self.dsrl_steering_enabled and noise is None:
            noise = self._dsrl_predict_noise(observation)
            if self.dsrl_inference_num_steps is not None:
                num_steps = self.dsrl_inference_num_steps
        if self.paper_rl_actor_critic:
            reference_actions, rl_hidden = self._sample_vla_actions(
                rng,
                observation,
                num_steps=num_steps,
                noise=noise,
                return_rl_hidden=True,
            )
            rl_latent, _ = self._paper_rl_encode(rl_hidden)
            corrected_actions, _, _ = self._paper_rl_apply_actor(reference_actions, rl_latent)
            return corrected_actions

        if self.snapflow_enabled:
            return self._sample_actions_one_step(rng, observation, noise=noise)

        return self._sample_vla_actions(rng, observation, num_steps=num_steps, noise=noise)

    @at.typecheck
    def legato_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 5,
        prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
        inference_delay: at.Int[at.Array, ""] | at.Int[at.Array, "b"] | int = 8,
        ramp_down: at.Int[at.Array, ""] | at.Int[at.Array, "b"] | int = 22,
        execute_horizon: at.Int[at.Array, ""] | at.Int[at.Array, "b"] | int = 30,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions with Legato per-step guidance.

        This follows the Legato-style inference update:
          1) guiding:  Y_t = (1 - omega) * X_t + omega * A_ref
          2) denoising: X_{t+dt} = Y_t + dt * f_theta(Y_t, t, omega)

        Notes:
          - Reference is built as A_ref = PADLAST(A_prev[s:H]), where s=execute_horizon.
          - If `prev_actions` is None, zeros are used as reference (cold start).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        s = jnp.asarray(execute_horizon, dtype=jnp.int32)
        s = jnp.clip(s, 1, self.action_horizon)
        if prev_actions is None:
            a_ref = jnp.zeros_like(noise)
        else:
            if prev_actions.shape != noise.shape:
                raise ValueError(
                    f"prev_actions shape must match noise/actions shape {noise.shape}, got {prev_actions.shape}"
                )
            # A_ref = PADLAST(A_prev[s:H]) with fixed target horizon H.
            # Implemented with dynamic gather so execute_horizon can be a traced value.
            pos = jnp.arange(self.action_horizon, dtype=jnp.int32)
            if s.ndim == 0:
                gather_idx = jnp.clip(pos + s, 0, self.action_horizon - 1)
                a_ref = jnp.take(prev_actions, gather_idx, axis=1)
            else:
                s = jnp.broadcast_to(s.reshape(-1), (batch_size,))
                gather_idx = jnp.clip(pos[None, :] + s[:, None], 0, self.action_horizon - 1)
                a_ref = jnp.take_along_axis(prev_actions, gather_idx[..., None], axis=1)

        omega = build_legato_schedule(
            self.action_horizon,
            inference_delay,
            ramp_down=ramp_down,
        )
        omega = jnp.broadcast_to(omega, (batch_size, self.action_horizon))
        omega_expanded = omega[..., None]

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Start from guided initialization Y0 = (1-omega)*eps + omega*A_ref.
        y_init = (1.0 - omega_expanded) * noise + omega_expanded * a_ref

        def step(carry):
            y_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                y_t,
                jnp.broadcast_to(time, batch_size),
                omega=omega if self.action_condition_on_omega else None,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            # Eq.(11)-equivalent discrete update:
            # X_{k+1} = Y_k + dt * f(Y_k), then Y_{k+1} = (1-omega)*X_{k+1} + omega*A_ref.
            x_next = y_t + dt * v_t
            y_next = (1.0 - omega_expanded) * x_next + omega_expanded * a_ref
            return y_next, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        y_0, _ = jax.lax.while_loop(cond, step, (y_init, 1.0))
        return y_0
