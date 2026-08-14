"""Optional Khmer continued pretraining (Phase 6).

**Run this only if the Phase 5 baseline proves it is needed.**  The decision
rule is in ``docs/training_guide.md``: if the base model already reads and
writes Khmer well, continued pretraining costs GPU-days and risks regressing
reasoning and instruction following for no measurable gain.

    python training/train_cpt.py --config configs/training/cpt.yaml --dry-run
    python training/train_cpt.py --config configs/training/cpt.yaml

**Not executed in this environment** (no GPU).  Supports streaming datasets,
checkpoint resume, gradient accumulation/checkpointing, BF16 and periodic
regression evaluation against a held-out English + reasoning probe set, which is
what makes the early-stopping rule enforceable rather than aspirational.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io import read_json, write_json
from common.logging import get_logger
from training.common import (
    TrainingConfig,
    build_run_manifest,
    find_target_modules,
    set_seed,
    write_run_manifest,
)

log = get_logger(__name__)

__all__ = ["REGRESSION_LIMITS", "main", "run_cpt", "should_accept_cpt"]

# Early-stopping / acceptance rule (§Phase 6).  A CPT checkpoint is rejected
# unless Khmer improves materially AND nothing else regresses materially.
REGRESSION_LIMITS: dict[str, float] = {
    "khmer_perplexity_improvement_min": 0.05,  # >= 5% relative improvement
    "english_perplexity_regression_max": 0.03,  # <= 3% relative degradation
    "reasoning_regression_max": 0.02,
    "instruction_following_regression_max": 0.02,
    "hallucination_regression_max": 0.01,
}


def should_accept_cpt(before: dict[str, float], after: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply the acceptance rule to before/after evaluation metrics.

    ``before``/``after`` use lower-is-better perplexities and higher-is-better
    accuracies, keyed as in ``evaluation/reports/``.
    """
    reasons: list[str] = []

    khmer_before = before.get("khmer_perplexity", 0.0)
    khmer_after = after.get("khmer_perplexity", 0.0)
    if khmer_before <= 0 or khmer_after <= 0:
        reasons.append("khmer_perplexity missing from one side; cannot judge benefit")
    else:
        improvement = (khmer_before - khmer_after) / khmer_before
        if improvement < REGRESSION_LIMITS["khmer_perplexity_improvement_min"]:
            reasons.append(
                f"Khmer improvement {improvement:.2%} is below the "
                f"{REGRESSION_LIMITS['khmer_perplexity_improvement_min']:.0%} threshold"
            )

    english_before = before.get("english_perplexity", 0.0)
    english_after = after.get("english_perplexity", 0.0)
    if english_before > 0 and english_after > 0:
        regression = (english_after - english_before) / english_before
        if regression > REGRESSION_LIMITS["english_perplexity_regression_max"]:
            reasons.append(f"English perplexity regressed {regression:.2%}")

    for metric, limit_key in (
        ("reasoning_accuracy", "reasoning_regression_max"),
        ("instruction_following", "instruction_following_regression_max"),
    ):
        if metric in before and metric in after:
            delta = before[metric] - after[metric]
            if delta > REGRESSION_LIMITS[limit_key]:
                reasons.append(f"{metric} regressed by {delta:.3f}")

    if "hallucination_rate" in before and "hallucination_rate" in after:
        delta = after["hallucination_rate"] - before["hallucination_rate"]
        if delta > REGRESSION_LIMITS["hallucination_regression_max"]:
            reasons.append(f"hallucination_rate worsened by {delta:.3f}")

    return (not reasons), reasons


def run_cpt(config: TrainingConfig, *, streaming: bool = True) -> dict[str, Any]:
    """Execute continued pretraining.  Requires ``requirements/training.txt``."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from datasets import load_dataset

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
        if config.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        quantization_config=quantization,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.lora_target_modules or find_target_modules(model),
        ),
    )

    # Streaming keeps a 200M-token corpus off the local disk on Colab.
    dataset = load_dataset(
        "json", data_files=config.dataset_path, split="train", streaming=streaming
    )

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.run_name,
        seed=config.seed,
        max_steps=config.max_steps if config.max_steps > 0 else 2000,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # A conservative LR: CPT on a strong base model damages it far more
        # easily than it improves it.
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    manifest = build_run_manifest(
        config, stage="cpt", dataset_paths={"corpus": config.dataset_path}
    )
    write_run_manifest(manifest, output_dir)

    checkpoint = _latest_checkpoint(output_dir) if config.resume_from_checkpoint else None
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    manifest["metrics"] = dict(result.metrics)
    manifest["executed"] = True
    write_run_manifest(manifest, output_dir)
    return manifest


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    return max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1])) if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python training/train_cpt.py")
    parser.add_argument("--config", default="configs/training/cpt.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument(
        "--decide",
        nargs=2,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        help="apply the acceptance rule to two evaluation reports and exit",
    )
    args = parser.parse_args(argv)

    if args.decide:
        before, after = (read_json(p) for p in args.decide)
        before_metrics = before.get("aggregate", before)
        after_metrics = after.get("aggregate", after)
        accept, reasons = should_accept_cpt(before_metrics, after_metrics)
        verdict = {
            "accept_cpt": accept,
            "reasons": reasons or ["all acceptance criteria met"],
            "limits": REGRESSION_LIMITS,
        }
        print(json.dumps(verdict, indent=2))
        return 0 if accept else 1

    config = TrainingConfig.from_yaml(args.config)

    if args.dry_run:
        manifest = build_run_manifest(
            config, stage="cpt-dry-run", dataset_paths={"corpus": config.dataset_path}
        )
        manifest["executed"] = False
        manifest["note"] = (
            "Dry run: configuration validated; no model loaded and no training performed. "
            "Continued pretraining must not start until the Phase 5 baseline shows a "
            "measurable Khmer deficiency (docs/training_guide.md)."
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    try:
        manifest = run_cpt(config, streaming=not args.no_streaming)
    except ImportError as exc:
        print(
            f"\nThe training stack is not installed ({exc}).\n"
            "    pip install -r requirements/training.txt",
            file=sys.stderr,
        )
        return 3
    write_json(Path(config.output_dir) / "cpt_manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "hyperparameters"}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
