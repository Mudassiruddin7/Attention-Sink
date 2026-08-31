"""SinkProbe. Separating attention sinks from position bias in long context models."""

__version__ = "0.2.0"

from .costmodel import HybridSpec
from .data import HaystackTask, TaskConfig
from .layers import GatedDeltaNet, SoftmaxAttention
from .metrics import evaluate, trials_for_halfwidth, wilson_interval
from .model import MAIN_LADDER, SIDE_BRANCHES, VARIANTS, ModelConfig, TinyLM

__all__ = [
    "HybridSpec", "HaystackTask", "TaskConfig", "GatedDeltaNet", "SoftmaxAttention",
    "evaluate", "trials_for_halfwidth", "wilson_interval",
    "MAIN_LADDER", "SIDE_BRANCHES", "VARIANTS", "ModelConfig", "TinyLM",
]
