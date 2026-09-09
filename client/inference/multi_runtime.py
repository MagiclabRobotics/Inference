from __future__ import annotations

from typing import Any

from policy_clients import create_policy_client
from runtime import InferenceRuntime


class MultiServerInferenceRuntime(InferenceRuntime):
    """Compatibility wrapper for configs that select the multi websocket transport."""

    def _make_policy_client(self):
        return create_policy_client(self.cfg)
