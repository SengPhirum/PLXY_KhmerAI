"""QLoRA / LoRA supervised fine-tuning (Phase 8).

Run on a GPU host or Colab::

    pip install -r requirements/training.txt
    python training/train_sft.py --config configs/training/sft_9b.yaml
    python training/train_sft.py --config configs/training/sft_4b.yaml --memory-profile 24gb
    python training/train_sft.py --config configs/training/sft_9b.yaml --resume

**Not executed in this environment** (no GPU): the commands above are the exact
ones to run, and every artefact the run produces is listed under "Expected
output" in ``docs/training_guide.md``.  ``--dry-run`` validates the config, the
dataset and the LoRA target discovery without loading weights, and is what CI
runs.

Design notes
------------
* LoRA targets are **discovered from the model** (``find_target_modules``)
  rather than hard-coded, so a naming change in a new Qwen release does not
  silently produce a no-op adapter.
* Loss is computed on assistant turns only (``training/chat_template.py``).
* Checkpoint resume is on by default and is verified by looking for a real
  checkpoint directory rather than trusting a flag.
* Every run writes ``run_manifest.json`` with the full reproducibility record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io import write_json
from common.logging import get_logger
from training.common import (
    MEMORY_PROFILES,
    TrainingConfig,
    assert_upload_approved,
    build_run_manifest,
    find_target_modules,
    set_seed,
    write_run_manifest,
)
from training.dataset_loader import load_sft_records, validate_records

log = get_logger(__name__)

__all__ = ["dry_run", "main", "prepare_dataset", "run_training"]


def prepare_dataset(config: TrainingConfig) -> dict[str, Any]:
    """Validate the dataset before a single GPU-second is spent on it."""
    train_records, train_stats = validate_records(
        load_sft_records(config.dataset_path), dedupe=True, near_dedupe=False
    )
    summary: dict[str, Any] = {"train": train_stats.to_dict()}
    if not train_records:
        raise ValueError(f"no valid SFT records in {config.dataset_path}")

    eval_path = Path(config.eval_dataset_path)
    if eval_path.is_file():
        _, eval_stats = validate_records(
            load_sft_records(eval_path), dedupe=False, near_dedupe=False
        )
        summary["validation"] = eval_stats.to_dict()
    else:
        log.warning("training.no_validation_set", extra={"path": str(eval_path)})
    return summary


def dry_run(config: TrainingConfig) -> dict[str, Any]:
    """Validate everything that does not need GPU weights.  Used by CI."""
    set_seed(config.seed)
    summary = prepare_dataset(config)
    manifest = build_run_manifest(
        config,
        stage="sft-dry-run",
        dataset_paths={"train": config.dataset_path, "validation": config.eval_dataset_path},
        extra={"dataset_summary": summary},
    )
    profile = MEMORY_PROFILES[config.memory_profile]
    manifest["memory_profile"] = {
        "name": profile.name,
        "vram_gb": profile.vram_gb,
        "effective_batch_size": profile.effective_batch_size,
        "notes": profile.notes,
    }
    manifest["executed"] = False
    manifest["note"] = (
        "Dry run: the dataset and configuration were validated, but no model was "
        "loaded and no training was performed."
    )
    return manifest


def run_training(config: TrainingConfig, *, resume: bool | None = None) -> dict[str, Any]:
    """Execute the SFT run.  Requires ``requirements/training.txt`` and a GPU."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from datasets import Dataset
    from training.chat_template import build_completion_mask, supervised_token_ratio

    set_seed(config.seed)
    dataset_summary = prepare_dataset(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.model_revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- model -----------------------------------------------------------
    quantization = None
    if config.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        quantization_config=quantization,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    targets = config.lora_target_modules or find_target_modules(model)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=(config.lora_target_modules and config.modules_to_save) or None,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "training.lora_applied",
        extra={
            "targets": targets,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
        },
    )

    # --- data ------------------------------------------------------------
    def _encode(path: str) -> Dataset:
        rows = []
        ratios = []
        for record in load_sft_records(path):
            encoded = build_completion_mask(
                [{"role": m.role, "content": m.content} for m in record.messages],
                tokenizer,
                system_prompt=None,  # SFT records carry their own system message
                max_length=config.max_seq_length,
            )
            if not encoded["input_ids"]:
                continue
            ratio = supervised_token_ratio(encoded)
            if ratio == 0.0:
                continue  # nothing to learn from this record
            ratios.append(ratio)
            rows.append(encoded)
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            log.info(
                "training.supervised_ratio",
                extra={"mean": round(mean_ratio, 4), "records": len(rows)},
            )
            if mean_ratio < 0.05:
                raise RuntimeError(
                    f"only {mean_ratio:.1%} of tokens are supervised - the completion mask is "
                    "almost certainly wrong for this tokenizer's chat template"
                )
        return Dataset.from_list(rows)

    train_dataset = _encode(config.dataset_path)
    eval_dataset = (
        _encode(config.eval_dataset_path) if Path(config.eval_dataset_path).is_file() else None
    )

    # --- training --------------------------------------------------------
    resume_from = (
        _find_checkpoint(output_dir)
        if (resume if resume is not None else config.resume_from_checkpoint)
        else None
    )

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.run_name,
        seed=config.seed,
        data_seed=config.seed,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        group_by_length=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        ),
    )

    manifest = build_run_manifest(
        config,
        stage="sft",
        dataset_paths={"train": config.dataset_path, "validation": config.eval_dataset_path},
        extra={
            "dataset_summary": dataset_summary,
            "lora_targets": targets,
            "trainable_params": trainable,
            "resumed_from": str(resume_from) if resume_from else None,
        },
    )
    write_run_manifest(manifest, output_dir)

    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = dict(result.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    manifest["metrics"] = metrics
    manifest["executed"] = True
    write_run_manifest(manifest, output_dir)
    write_json(output_dir / "metrics.json", metrics)

    log.info("training.complete", extra={"output": str(output_dir), "metrics": metrics})
    return manifest


def _find_checkpoint(output_dir: Path) -> Path | None:
    """Latest ``checkpoint-N`` directory, or None.

    Checked on disk rather than trusting the flag, so ``--resume`` on a fresh
    output directory starts cleanly instead of failing inside the Trainer.
    """
    candidates = [
        p
        for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and (p / "trainer_state.json").is_file()
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    log.info("training.resuming", extra={"checkpoint": str(latest)})
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python training/train_sft.py")
    parser.add_argument("--config", required=True, help="configs/training/sft_*.yaml")
    parser.add_argument("--memory-profile", choices=sorted(MEMORY_PROFILES), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate only; no model load")
    parser.add_argument(
        "--skip-upload-check",
        action="store_true",
        help="skip the §36 PII/secret pre-upload gate (on-premise training only)",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {"training": {}}
    if args.memory_profile:
        overrides["training"]["memory_profile"] = args.memory_profile
    if args.output_dir:
        overrides["training"]["output_dir"] = args.output_dir
    if args.dataset:
        overrides["training"]["dataset_path"] = args.dataset

    config = TrainingConfig.from_yaml(
        args.config, overrides=overrides if overrides["training"] else None
    )

    if not args.skip_upload_check:
        try:
            assert_upload_approved()
        except RuntimeError as exc:
            print(f"\nPRE-UPLOAD GATE: {exc}\n", file=sys.stderr)
            print(
                "Pass --skip-upload-check only when training on premises with data that "
                "never leaves the building.",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        manifest = dry_run(config)
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    try:
        manifest = run_training(config, resume=False if args.no_resume else (args.resume or None))
    except ImportError as exc:
        print(
            f"\nThe training stack is not installed ({exc}).\n"
            "    pip install -r requirements/training.txt\n"
            "Use --dry-run to validate the configuration without it.",
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "hyperparameters"}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
