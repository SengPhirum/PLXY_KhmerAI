"""Merge a LoRA adapter into the base weights (Phase 12).

Ollama serves merged weights, not adapters, so this is a required step between
training and deployment.

    python training/merge_lora.py \
        --base Qwen/Qwen3.5-9B \
        --adapter outputs/sft_9b/adapter \
        --output models/khmer-support-9b-merged

**Not executed in this environment** (no GPU / no model weights).  The merge
runs on CPU with enough RAM, or on the training host.

A merge is only correct if the adapter was trained against the *same* base
revision, so the adapter's ``run_manifest.json`` is checked against the
requested base model and the merge is refused on a mismatch - silently merging
mismatched weights produces a model that appears to work and is subtly broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io import read_json, write_json
from common.logging import get_logger

log = get_logger(__name__)

__all__ = ["merge_adapter", "verify_compatibility", "main"]


def verify_compatibility(base_model: str, adapter_dir: str | Path) -> dict[str, Any]:
    """Check that the adapter was trained on the requested base model."""
    adapter = Path(adapter_dir)
    report: dict[str, Any] = {"adapter": str(adapter), "base_model": base_model, "warnings": []}

    config_path = adapter / "adapter_config.json"
    if config_path.is_file():
        adapter_config = read_json(config_path)
        trained_on = adapter_config.get("base_model_name_or_path", "")
        report["adapter_base_model"] = trained_on
        report["lora_rank"] = adapter_config.get("r")
        report["lora_alpha"] = adapter_config.get("lora_alpha")
        report["target_modules"] = adapter_config.get("target_modules")
        if trained_on and trained_on != base_model:
            report["warnings"].append(
                f"adapter was trained on {trained_on!r} but the merge targets {base_model!r}"
            )
    else:
        report["warnings"].append("adapter_config.json not found; cannot verify the base model")

    manifest_path = adapter.parent / "run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        report["run"] = {
            "stage": manifest.get("stage"),
            "code_commit": manifest.get("code_commit"),
            "seed": manifest.get("seed"),
            "model_revision": manifest.get("model_revision"),
            "config_fingerprint": manifest.get("config_fingerprint"),
        }
    else:
        report["warnings"].append(
            "run_manifest.json not found next to the adapter; provenance cannot be recorded"
        )

    report["compatible"] = not any("trained on" in w for w in report["warnings"])
    return report


def merge_adapter(
    base_model: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    model_revision: str = "main",
    dtype: str = "bfloat16",
    device_map: str = "cpu",
    force: bool = False,
) -> dict[str, Any]:
    """Load base + adapter, merge, and save the result plus a provenance record."""
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    compatibility = verify_compatibility(base_model, adapter_dir)
    if not compatibility["compatible"] and not force:
        raise RuntimeError(
            f"refusing to merge: {'; '.join(compatibility['warnings'])}. "
            "Pass --force only if you are certain the weights match."
        )
    for warning in compatibility["warnings"]:
        log.warning("merge.warning", extra={"detail": warning})

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    log.info("merge.loading_base", extra={"model": base_model, "dtype": dtype})
    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=model_revision, dtype=torch_dtype, device_map=device_map
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    log.info("merge.merging")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(target), safe_serialization=True)

    tokenizer_source = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.save_pretrained(str(target))

    provenance = {
        "base_model": base_model,
        "model_revision": model_revision,
        "adapter": str(adapter_dir),
        "output": str(target),
        "dtype": dtype,
        "compatibility": compatibility,
        "files": sorted(p.name for p in target.iterdir()),
    }
    write_json(target / "merge_provenance.json", provenance)
    log.info("merge.complete", extra={"output": str(target)})
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python training/merge_lora.py")
    parser.add_argument("--base", required=True, help="base model id or local path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="merged model output directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="cpu", help="cpu (default) or auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_only:
        report = verify_compatibility(args.base, args.adapter)
        print(json.dumps(report, indent=2))
        return 0 if report["compatible"] else 1

    try:
        provenance = merge_adapter(
            args.base,
            args.adapter,
            args.output,
            model_revision=args.revision,
            dtype=args.dtype,
            device_map=args.device_map,
            force=args.force,
        )
    except ImportError as exc:
        print(
            f"\ntorch/transformers/peft are required to merge ({exc}).\n"
            "    pip install -r requirements/training.txt",
            file=sys.stderr,
        )
        return 3
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(json.dumps(provenance, indent=2))
    print(
        f"\nNext: convert to GGUF and build the Ollama model:\n"
        f"    python training/export_model.py --merged {args.output} --quantize Q4_K_M,Q5_K_M,Q8_0\n"
        f"    bash ollama/create_model.sh"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
