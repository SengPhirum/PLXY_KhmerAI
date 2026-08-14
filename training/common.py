"""Shared training utilities: reproducibility, hardware profiles, run manifests.

Everything in this module works without ``torch`` installed, so the config
loading and the run-manifest logic are testable in CI.  Heavy imports happen
inside functions, on the GPU host only.

Reproducibility (§1.4) is not optional here: :func:`build_run_manifest` records
the model revision, dataset revision, preprocessing version, code commit, seed,
hyper-parameters, environment and hardware for every run, and the manifest is
written next to the checkpoint.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.config import load_config
from common.hashing import sha256_file, sha256_text
from common.io import read_json, write_json
from common.logging import get_logger
from common.versions import git_commit

log = get_logger(__name__)

__all__ = [
    "MEMORY_PROFILES",
    "MemoryProfile",
    "TrainingConfig",
    "assert_upload_approved",
    "build_run_manifest",
    "detect_hardware",
    "find_target_modules",
    "resolve_memory_profile",
    "set_seed",
]


# --- memory profiles (§Phase 8) --------------------------------------------
@dataclass(slots=True, frozen=True)
class MemoryProfile:
    """A configuration that is known to fit a given GPU."""

    name: str
    vram_gb: int
    load_in_4bit: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    bf16: bool
    packing: bool
    notes: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


MEMORY_PROFILES: dict[str, MemoryProfile] = {
    "16gb": MemoryProfile(
        name="16gb",
        vram_gb=16,
        load_in_4bit=True,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        max_seq_length=2048,
        gradient_checkpointing=True,
        lora_rank=32,
        lora_alpha=64,
        bf16=False,
        packing=True,
        notes="T4/V100 class. fp16 compute; 4-bit base weights are mandatory.",
    ),
    "24gb": MemoryProfile(
        name="24gb",
        vram_gb=24,
        load_in_4bit=True,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=3072,
        gradient_checkpointing=True,
        lora_rank=32,
        lora_alpha=64,
        bf16=True,
        packing=True,
        notes="L4/A10/3090 class. Comfortable for the 4B, tight for the 9B.",
    ),
    "40gb": MemoryProfile(
        name="40gb",
        vram_gb=40,
        load_in_4bit=True,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=4096,
        gradient_checkpointing=True,
        lora_rank=64,
        lora_alpha=128,
        bf16=True,
        packing=True,
        notes="A100 40GB. The recommended profile for the 9B SFT run.",
    ),
    "80gb": MemoryProfile(
        name="80gb",
        vram_gb=80,
        load_in_4bit=False,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
        max_seq_length=4096,
        gradient_checkpointing=False,
        lora_rank=64,
        lora_alpha=128,
        bf16=True,
        packing=True,
        notes="A100/H100 80GB. LoRA on bf16 weights; no quantisation needed.",
    ),
}


@dataclass(slots=True)
class TrainingConfig:
    """Flattened training configuration loaded from ``configs/training/*.yaml``."""

    run_name: str = "sft"
    base_model: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "main"
    output_dir: str = "outputs/sft"
    dataset_path: str = "data/sft/train.jsonl"
    eval_dataset_path: str = "data/sft/validation.jsonl"
    seed: int = 20260814

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    modules_to_save: list[str] = field(default_factory=list)

    # Optimisation
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    max_steps: int = -1
    warmup_ratio: float = 0.04
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"

    # Runtime
    memory_profile: str = "40gb"
    max_seq_length: int = 4096
    packing: bool = True
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    resume_from_checkpoint: bool = True
    report_to: str = "none"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, overrides: dict[str, Any] | None = None
    ) -> TrainingConfig:
        raw = load_config(path, overrides=overrides)
        section = raw.get("training", raw)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in section.items() if k in known}
        extra = {k: v for k, v in section.items() if k not in known}
        config = cls(**kwargs)
        config.extra = extra
        return resolve_memory_profile(config)

    def fingerprint(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "extra"}
        return sha256_text(json.dumps(payload, sort_keys=True, default=str))[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_memory_profile(config: TrainingConfig) -> TrainingConfig:
    """Apply the named memory profile, leaving explicit overrides intact."""
    profile = MEMORY_PROFILES.get(config.memory_profile)
    if profile is None:
        raise ValueError(
            f"unknown memory profile {config.memory_profile!r}; "
            f"expected one of {', '.join(MEMORY_PROFILES)}"
        )
    defaults = TrainingConfig()
    for field_name, value in (
        ("per_device_train_batch_size", profile.per_device_batch_size),
        ("gradient_accumulation_steps", profile.gradient_accumulation_steps),
        ("max_seq_length", profile.max_seq_length),
        ("gradient_checkpointing", profile.gradient_checkpointing),
        ("load_in_4bit", profile.load_in_4bit),
        ("bf16", profile.bf16),
        ("packing", profile.packing),
        ("lora_rank", profile.lora_rank),
        ("lora_alpha", profile.lora_alpha),
    ):
        # Only fill in values the config left at its default.
        if getattr(config, field_name) == getattr(defaults, field_name):
            setattr(config, field_name, value)
    return config


# --- reproducibility --------------------------------------------------------
def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG in play.  Safe to call without torch installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    log.info("training.seed_set", extra={"seed": seed, "deterministic": deterministic})


def detect_hardware() -> dict[str, Any]:
    """Describe the machine a run happened on - part of the run manifest."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except ImportError:
        info["torch"] = None
    return info


def build_run_manifest(
    config: TrainingConfig,
    *,
    stage: str,
    dataset_paths: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The reproducibility record required by §1.4."""
    datasets: dict[str, Any] = {}
    for name, path in (dataset_paths or {}).items():
        target = Path(path)
        datasets[name] = {
            "path": str(target),
            "exists": target.is_file(),
            "sha256": sha256_file(str(target)) if target.is_file() else "",
            "bytes": target.stat().st_size if target.is_file() else 0,
        }

    manifest: dict[str, Any] = {
        "stage": stage,
        "run_name": config.run_name,
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "datasets": datasets,
        "preprocessing_version": _preprocessing_version(),
        "code_commit": git_commit(short=False),
        "seed": config.seed,
        "config_fingerprint": config.fingerprint(),
        "hyperparameters": config.to_dict(),
        "environment": {
            "packages": _installed_versions(
                ["torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]
            ),
            "env_vars": {
                k: os.environ.get(k, "")
                for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "WANDB_MODE", "KHMERAI_SEED")
            },
        },
        "hardware": detect_hardware(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def _preprocessing_version() -> str:
    """Fingerprint the preprocessing rules that produced the training data."""
    from preprocessing.unicode_normalization import NormalizationConfig

    return sha256_text(json.dumps(asdict(NormalizationConfig()), sort_keys=True))[:12]


def _installed_versions(packages: list[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


def write_run_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir) / "run_manifest.json"
    write_json(target, manifest)
    log.info("training.manifest_written", extra={"path": str(target)})
    return target


# --- cloud-training safety gate (§36) ---------------------------------------
def assert_upload_approved(
    report_path: str | Path = "data/manifests/pre_upload_report.json",
    *,
    require: bool = True,
) -> None:
    """Refuse to train on data that has not passed the PII/secret pre-upload scan.

    This is the mechanical enforcement of §36.  Training scripts call it before
    reading any dataset when running off-premise.
    """
    if not require:
        return
    target = Path(report_path)
    if not target.is_file():
        raise RuntimeError(
            f"no pre-upload scan at {target}. Generate one before cloud training:\n"
            '    python -c "from preprocessing.pii_filter import build_pre_upload_report; '
            "from common.io import write_json; "
            "write_json('data/manifests/pre_upload_report.json', "
            "build_pre_upload_report(['data/sft/train.jsonl']))\""
        )
    report = read_json(target)
    if not report.get("approved", False):
        findings = report.get("blocking_findings", {})
        raise RuntimeError(
            f"pre-upload scan did NOT approve this dataset: {findings}. "
            "Resolve the findings, or record a written data-handling approval "
            "(docs/security.md) before overriding."
        )
    log.info("training.upload_approved", extra={"report": str(target)})


# --- LoRA target discovery --------------------------------------------------
def find_target_modules(
    model: Any, *, include_mlp: bool = True, exclude_multimodal: bool = True
) -> list[str]:
    """Discover the linear projections LoRA should adapt (§Phase 8).

    Rather than hard-coding ``["q_proj", "k_proj", ...]`` - which silently
    adapts nothing when a model uses different names - this inspects the actual
    module tree.  Vision/audio towers are excluded because this is a text-only
    project and adapting them wastes parameters and memory.
    """
    import torch.nn as nn

    attention_hints = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj", "wqkv")
    mlp_hints = ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2")
    multimodal_hints = (
        "vision",
        "visual",
        "image",
        "audio",
        "speech",
        "mm_projector",
        "patch_embed",
    )

    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lowered = name.lower()
        if exclude_multimodal and any(hint in lowered for hint in multimodal_hints):
            continue
        if "lm_head" in lowered or "embed" in lowered:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in attention_hints or (include_mlp and leaf in mlp_hints):
            found.add(leaf)

    if not found:
        raise RuntimeError(
            "no LoRA target modules discovered. Inspect the model with "
            "`print([n for n, _ in model.named_modules()])` and set "
            "`lora_target_modules` explicitly in the training config."
        )
    ordered = sorted(found)
    log.info("training.lora_targets", extra={"modules": ordered})
    return ordered
