from openpi_client import action_chunk_broker
import numpy as np
import pytest

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


CONFIG_NAME = "pi05_flatten_fold_normal_follow_paper"
CHECKPOINT_DIR = "/path/to/checkpoint"
ACTION_DIM = 14


def make_agilex_example() -> dict:
    return {
        "images": {
            "top_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "hand_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "hand_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "state": np.random.rand(ACTION_DIM),
        "prompt": "Flatten and fold the cloth.",
    }


@pytest.mark.manual
def test_infer():
    config = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(config, CHECKPOINT_DIR)

    example = make_agilex_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, ACTION_DIM)


@pytest.mark.manual
def test_broker():
    config = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(config, CHECKPOINT_DIR)

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = make_agilex_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (ACTION_DIM,)
