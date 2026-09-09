"""Inference-only Training-Time RTC sampling for Pi0/Pi0.5 checkpoints.

The training objective and dataset pipeline intentionally live outside this
repository.  This module only implements the clean action-prefix sampler needed
to serve an already TTRTC-trained, architecture-compatible JAX checkpoint.
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


def posemb_sincos_ttrtc(
    pos: jax.Array,
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> jax.Array:
    """Return sine/cosine embeddings for scalar or per-action timesteps."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.expand_dims(pos, axis=-1) * (1.0 / period * 2 * jnp.pi)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0TTRTC(pi0.Pi0):
    """Pi0/Pi0.5 sampler with clean-prefix action conditioning.

    The class deliberately inherits the checkpoint architecture unchanged.  It
    adds only the inference path; there is no TTRTC loss or training entry here.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

    @at.typecheck
    def ttrtc_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | jax.Array = 10,
        prev_actions: _model.Actions,
        execute_horizon: int | jax.Array = 0,
        inference_delay: int | jax.Array = 0,
        noise: _model.Actions | None = None,
    ) -> _model.Actions:
        """Generate a chunk while pinning the still-in-flight action prefix.

        ``execute_horizon`` locates the first unexecuted action in the previous
        model-space chunk. ``inference_delay`` selects how many following actions
        remain clean and fixed while the new postfix is denoised.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        horizon = self.action_horizon

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, horizon, self.action_dim))

        prev_actions = jnp.asarray(prev_actions, dtype=noise.dtype)
        if prev_actions.ndim == 2:
            prev_actions = prev_actions[None, ...]
        if prev_actions.shape[0] == 1 and batch_size != 1:
            prev_actions = jnp.broadcast_to(prev_actions, (batch_size, *prev_actions.shape[1:]))
        if prev_actions.shape[0] != batch_size:
            raise ValueError(
                f"prev_actions batch must be 1 or {batch_size}, got {prev_actions.shape[0]}"
            )
        if prev_actions.shape[1] == 0:
            prev_actions = jnp.zeros_like(noise)
        elif prev_actions.shape[1] < horizon:
            tail = jnp.repeat(prev_actions[:, -1:, :], horizon - prev_actions.shape[1], axis=1)
            prev_actions = jnp.concatenate([prev_actions, tail], axis=1)
        elif prev_actions.shape[1] > horizon:
            prev_actions = prev_actions[:, :horizon, :]
        if prev_actions.shape[-1] > self.action_dim:
            prev_actions = prev_actions[..., : self.action_dim]
        elif prev_actions.shape[-1] < self.action_dim:
            pad = jnp.zeros(
                (*prev_actions.shape[:-1], self.action_dim - prev_actions.shape[-1]),
                dtype=prev_actions.dtype,
            )
            prev_actions = jnp.concatenate([prev_actions, pad], axis=-1)

        pos = jnp.arange(horizon, dtype=jnp.int32)
        start = jnp.clip(jnp.asarray(execute_horizon, dtype=jnp.int32), 0, horizon)
        delay = jnp.clip(jnp.asarray(inference_delay, dtype=jnp.int32), 0, horizon)
        if start.ndim == 0:
            gather_idx = jnp.clip(start + pos, 0, horizon - 1)
            prefix_ref = jnp.take(prev_actions, gather_idx, axis=1)
            prefix_mask = jnp.broadcast_to((pos < delay)[None, :], (batch_size, horizon))
        else:
            start = jnp.broadcast_to(start.reshape(-1), (batch_size,))
            delay = jnp.broadcast_to(delay.reshape(-1), (batch_size,))
            gather_idx = jnp.clip(start[:, None] + pos[None, :], 0, horizon - 1)
            prefix_ref = jnp.take_along_axis(prev_actions, gather_idx[..., None], axis=1)
            prefix_mask = pos[None, :] < delay[:, None]

        prefix_ref = jnp.nan_to_num(prefix_ref, nan=0.0, posinf=0.0, neginf=0.0)
        x_init = jnp.where(prefix_mask[..., None], prefix_ref, noise)

        prefix_tokens, prefix_pad_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_pad_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_pad_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def step(carry):
            x_t, time = carry
            token_time = jnp.where(prefix_mask, 0.0, time)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                token_time,
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_local = einops.repeat(
                prefix_pad_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
            positions_local = (
                jnp.sum(prefix_pad_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_local,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            velocity = self.action_out_proj(suffix_out[:, -horizon:])
            velocity = jnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            x_next = x_t + dt * velocity
            x_next = jnp.where(prefix_mask[..., None], prefix_ref, x_next)
            return x_next, time + dt

        def scan_step(carry, _):
            return step(carry), None

        (actions, _), _ = jax.lax.scan(scan_step, (x_init, 1.0), xs=None, length=num_steps)
        return jnp.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

    @at.typecheck
    def embed_suffix(
        self,
        obs_or_state: _model.Observation | jax.Array,
        noisy_actions: _model.Actions,
        timestep: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
    ]:
        """Embed action tokens with an individual flow timestep per token."""
        input_mask = []
        ar_mask = []
        tokens = []
        state = obs_or_state.state if hasattr(obs_or_state, "state") else obs_or_state

        if not self.pi05:
            state_token = self.state_proj(state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        timestep = jnp.asarray(timestep, dtype=noisy_actions.dtype)
        if timestep.ndim == 1:
            timestep = jnp.broadcast_to(timestep[:, None], noisy_actions.shape[:2])
        if timestep.shape != noisy_actions.shape[:2]:
            raise ValueError(f"timestep must have shape [B] or [B, H], got {timestep.shape}")

        time_emb = posemb_sincos_ttrtc(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            action_time_tokens = jnp.concatenate([action_tokens, time_emb], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.array(ar_mask),
            adarms_cond,
        )
