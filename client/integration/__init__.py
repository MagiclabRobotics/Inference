"""采集对接层：与 InferenceRuntime / 策略实现解耦，策略变更时主要改 inference/ 与本包配置。"""

from integration.collector_contract import ACTION_DIM, IMAGE_KEYS, INFERENCE_SDK_VERSION

__all__ = ["ACTION_DIM", "IMAGE_KEYS", "INFERENCE_SDK_VERSION"]
