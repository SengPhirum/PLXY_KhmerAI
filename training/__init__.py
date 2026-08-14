"""Training stack: continued pretraining, SFT, DPO, merge and export.

Heavy dependencies (torch, transformers, peft, trl) are imported lazily inside
functions so that ``import training`` works from ``requirements/base.txt`` and
the configuration/dataset logic stays testable in CI.
"""

from training.chat_template import (
    DEFAULT_SYSTEM_PROMPT_KM,
    IGNORE_INDEX,
    ChatMLTemplate,
    build_completion_mask,
    render_conversation,
)
from training.common import (
    MEMORY_PROFILES,
    MemoryProfile,
    TrainingConfig,
    build_run_manifest,
    detect_hardware,
    set_seed,
)
from training.dataset_loader import (
    DatasetStats,
    build_splits,
    load_sft_records,
    split_records,
    validate_records,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT_KM",
    "IGNORE_INDEX",
    "MEMORY_PROFILES",
    "ChatMLTemplate",
    "DatasetStats",
    "MemoryProfile",
    "TrainingConfig",
    "build_completion_mask",
    "build_run_manifest",
    "build_splits",
    "detect_hardware",
    "load_sft_records",
    "render_conversation",
    "set_seed",
    "split_records",
    "validate_records",
]
