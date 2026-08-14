"""Optional preference optimisation with DPO (Phase 9).

DPO runs **after** SFT and is kept only if it beats the SFT checkpoint on the
held-out tests.  §Phase 9 is explicit that this phase should be rejected when
preference data is weak or gains are not measurable, so the acceptance rule is
implemented (:func:`should_accept_dpo`) rather than left to judgement.

    python training/train_dpo.py --config configs/training/dpo.yaml --dry-run
    python training/train_dpo.py --config configs/training/dpo.yaml
    python training/train_dpo.py --decide sft_report.json dpo_report.json

Preference criteria the pairs must encode (§Phase 9):
factual > fabricated, concise > verbose, natural Khmer > translation-like Khmer,
grounded > unsupported, polite > robotic, clarification > guessing,
correct escalation > flat refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io import read_json
from common.logging import get_logger
from training.common import TrainingConfig, build_run_manifest, set_seed, write_run_manifest
from training.dataset_loader import load_preference_pairs

log = get_logger(__name__)

__all__ = ["main", "run_dpo", "should_accept_dpo", "audit_pairs"]

# A DPO checkpoint is kept only if it wins on quality without regressing safety.
ACCEPTANCE = {
    "min_pairs": 1000,
    "support_accuracy_improvement_min": 0.01,
    "hallucination_regression_max": 0.005,
    "grounding_regression_max": 0.01,
    "khmer_fluency_regression_max": 0.02,
}


def audit_pairs(pairs: list[dict[str, str]]) -> dict[str, Any]:
    """Quality audit of the preference set before spending GPU time on it."""
    from preprocessing.khmer_detection import TextLanguage, detect_language  # noqa: PLC0415
    from preprocessing.unicode_normalization import normalize_for_hashing  # noqa: PLC0415

    identical = 0
    non_khmer = 0
    chosen_longer = 0
    seen: set[str] = set()
    duplicates = 0

    for pair in pairs:
        if normalize_for_hashing(pair["chosen"]) == normalize_for_hashing(pair["rejected"]):
            identical += 1
        language, _ = detect_language(pair["chosen"], khmer_present=0.10)
        if language is TextLanguage.ENGLISH:
            non_khmer += 1
        if len(pair["chosen"]) > len(pair["rejected"]):
            chosen_longer += 1
        key = normalize_for_hashing(pair["prompt"])
        if key in seen:
            duplicates += 1
        seen.add(key)

    total = len(pairs) or 1
    audit = {
        "pairs": len(pairs),
        "identical_chosen_rejected": identical,
        "non_khmer_chosen": non_khmer,
        "duplicate_prompts": duplicates,
        "chosen_longer_ratio": round(chosen_longer / total, 4),
        "usable": len(pairs) - identical,
    }
    # If "chosen" is almost always the longer answer, DPO will learn "be verbose"
    # rather than "be correct" - the opposite of the §32 generation policy.
    audit["length_bias_warning"] = audit["chosen_longer_ratio"] > 0.75
    audit["sufficient"] = audit["usable"] >= ACCEPTANCE["min_pairs"]
    return audit


def should_accept_dpo(sft: dict[str, float], dpo: dict[str, float]) -> tuple[bool, list[str]]:
    """Keep the DPO checkpoint only if it beats SFT on held-out tests."""
    reasons: list[str] = []

    improvement = dpo.get("support_accuracy", 0.0) - sft.get("support_accuracy", 0.0)
    if improvement < ACCEPTANCE["support_accuracy_improvement_min"]:
        reasons.append(
            f"support accuracy improved by only {improvement:+.3f} "
            f"(needs >= {ACCEPTANCE['support_accuracy_improvement_min']})"
        )

    for metric, limit_key, higher_is_worse in (
        ("hallucination_rate", "hallucination_regression_max", True),
        ("grounding_precision", "grounding_regression_max", False),
        ("khmer_fluency", "khmer_fluency_regression_max", False),
    ):
        if metric not in sft or metric not in dpo:
            continue
        delta = (dpo[metric] - sft[metric]) if higher_is_worse else (sft[metric] - dpo[metric])
        if delta > ACCEPTANCE[limit_key]:
            reasons.append(f"{metric} regressed by {delta:.3f}")

    return (not reasons), reasons


def run_dpo(config: TrainingConfig, *, sft_adapter: str | None = None) -> dict[str, Any]:
    """Execute DPO on top of the SFT adapter."""
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig, PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    set_seed(config.seed)
    pairs = load_preference_pairs(config.dataset_path)
    audit = audit_pairs(pairs)
    log.info("training.dpo.pair_audit", extra=audit)
    if not audit["sufficient"]:
        raise RuntimeError(
            f"only {audit['usable']} usable preference pairs; §Phase 9 requires at least "
            f"{ACCEPTANCE['min_pairs']}. Reject this phase rather than training on weak data."
        )
    if audit["length_bias_warning"]:
        log.warning(
            "training.dpo.length_bias",
            extra={
                "chosen_longer_ratio": audit["chosen_longer_ratio"],
                "risk": "DPO may learn verbosity instead of correctness",
            },
        )

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
    )
    adapter = sft_adapter or config.extra.get("sft_adapter")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        log.info("training.dpo.loaded_sft_adapter", extra={"adapter": adapter})

    dataset = Dataset.from_list(pairs)
    beta = float(config.extra.get("dpo_beta", 0.1))

    arguments = DPOConfig(
        output_dir=str(output_dir),
        run_name=config.run_name,
        seed=config.seed,
        beta=beta,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # DPO needs a much smaller LR than SFT; the usual failure is a collapsed
        # policy from reusing the SFT learning rate.
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
        report_to=config.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: the reference is the adapter-disabled base model
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=(
            None
            if adapter
            else LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        ),
    )

    manifest = build_run_manifest(
        config, stage="dpo", dataset_paths={"preference": config.dataset_path},
        extra={"pair_audit": audit, "beta": beta, "sft_adapter": adapter},
    )
    write_run_manifest(manifest, output_dir)

    result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    manifest["metrics"] = dict(result.metrics)
    manifest["executed"] = True
    write_run_manifest(manifest, output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python training/train_dpo.py")
    parser.add_argument("--config", default="configs/training/dpo.yaml")
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="audit the pairs and exit")
    parser.add_argument(
        "--decide",
        nargs=2,
        metavar=("SFT_REPORT", "DPO_REPORT"),
        help="apply the acceptance rule to two evaluation reports and exit",
    )
    args = parser.parse_args(argv)

    if args.decide:
        sft, dpo = (read_json(p) for p in args.decide)
        accept, reasons = should_accept_dpo(sft.get("aggregate", sft), dpo.get("aggregate", dpo))
        print(
            json.dumps(
                {
                    "keep_dpo_checkpoint": accept,
                    "reasons": reasons or ["DPO beats SFT on every gate"],
                    "acceptance": ACCEPTANCE,
                },
                indent=2,
            )
        )
        return 0 if accept else 1

    config = TrainingConfig.from_yaml(args.config)

    if args.audit_only or args.dry_run:
        pairs = load_preference_pairs(config.dataset_path) if Path(config.dataset_path).is_file() else []
        audit = audit_pairs(pairs)
        manifest = build_run_manifest(
            config, stage="dpo-dry-run", dataset_paths={"preference": config.dataset_path}
        )
        manifest["pair_audit"] = audit
        manifest["executed"] = False
        manifest["recommendation"] = (
            "proceed" if audit["sufficient"] else "REJECT this phase: not enough usable pairs"
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0 if audit["sufficient"] else 1

    try:
        manifest = run_dpo(config, sft_adapter=args.sft_adapter)
    except ImportError as exc:
        print(
            f"\nThe training stack is not installed ({exc}).\n"
            "    pip install -r requirements/training.txt",
            file=sys.stderr,
        )
        return 3
    except RuntimeError as exc:
        print(f"\nDPO rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in manifest.items() if k != "hyperparameters"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
