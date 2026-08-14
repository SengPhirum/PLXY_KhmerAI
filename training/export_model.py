"""Export a merged model to GGUF and quantize it for Ollama (Phase 12).

    python training/export_model.py --merged models/khmer-support-9b-merged \
        --outdir models/gguf --quantize Q4_K_M,Q5_K_M,Q8_0

**Not executed in this environment**: conversion needs `llama.cpp` and the model
weights, neither of which is present here.  The script locates `llama.cpp`,
prints the exact commands, runs them when the tooling is available, and writes a
quantization comparison table.

Why multiple quantizations
--------------------------
§Phase 12 forbids automatically choosing the smallest file.  Khmer is
particularly sensitive to aggressive quantization: the script emits several
levels so that ``ollama/benchmark.sh`` and ``evaluation/benchmark_model.py`` can
measure Khmer integrity, memory and throughput for each, and the choice is made
from the measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.io import write_json
from common.logging import get_logger

log = get_logger(__name__)

__all__ = ["convert_to_gguf", "export", "find_llama_cpp", "main", "quantize"]

# Practical levels for a 48 GB Mac Studio, smallest last.
DEFAULT_QUANTIZATIONS = ("Q8_0", "Q5_K_M", "Q4_K_M")

_SEARCH_PATHS = (
    "llama.cpp",
    "../llama.cpp",
    "~/llama.cpp",
    "/opt/llama.cpp",
    "/usr/local/opt/llama.cpp",
)


def find_llama_cpp(explicit: str | None = None) -> Path | None:
    """Locate a llama.cpp checkout containing the conversion script."""
    candidates = [explicit] if explicit else []
    candidates += [os.environ.get("LLAMA_CPP_DIR", "")]
    candidates += list(_SEARCH_PATHS)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "convert_hf_to_gguf.py").is_file() or (path / "convert-hf-to-gguf.py").is_file():
            return path
    return None


def _convert_script(llama_cpp: Path) -> Path:
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        if (llama_cpp / name).is_file():
            return llama_cpp / name
    raise FileNotFoundError(f"no conversion script found in {llama_cpp}")


def _quantize_binary(llama_cpp: Path) -> Path | None:
    for candidate in (
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "llama-quantize",
        llama_cpp / "build" / "bin" / "quantize",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("llama-quantize")
    return Path(found) if found else None


def _run(argv: list[str], *, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(str(a) for a in argv)
    if dry_run:
        log.info("export.command", extra={"command": printable, "executed": False})
        return {"command": printable, "executed": False, "returncode": None}
    log.info("export.running", extra={"command": printable})
    result = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        log.error(
            "export.failed",
            extra={
                "command": printable,
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:],
            },
        )
    return {
        "command": printable,
        "executed": True,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-1000:] if result.returncode != 0 else "",
    }


def convert_to_gguf(
    merged_dir: str | Path,
    outdir: str | Path,
    *,
    llama_cpp: Path,
    name: str,
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Convert HF safetensors to an F16 GGUF."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    output = target / f"{name}-f16.gguf"
    command = [
        sys.executable,
        str(_convert_script(llama_cpp)),
        str(merged_dir),
        "--outfile",
        str(output),
        "--outtype",
        "f16",
    ]
    return output, _run(command, dry_run=dry_run)


def quantize(
    source_gguf: Path,
    outdir: str | Path,
    level: str,
    *,
    llama_cpp: Path,
    name: str,
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any]]:
    binary = _quantize_binary(llama_cpp)
    output = Path(outdir) / f"{name}-{level}.gguf"
    if binary is None:
        return output, {
            "command": f"<llama-quantize> {source_gguf} {output} {level}",
            "executed": False,
            "error": (
                "llama-quantize not found. Build it with:\n"
                "    cmake -B build && cmake --build build --config Release -j"
            ),
        }
    return output, _run([str(binary), str(source_gguf), str(output), level], dry_run=dry_run)


def export(
    merged_dir: str | Path,
    outdir: str | Path,
    *,
    name: str,
    quantizations: tuple[str, ...] = DEFAULT_QUANTIZATIONS,
    llama_cpp_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full export: HF -> F16 GGUF -> one file per quantization level."""
    llama_cpp = find_llama_cpp(llama_cpp_dir)
    report: dict[str, Any] = {
        "merged_dir": str(merged_dir),
        "outdir": str(outdir),
        "name": name,
        "quantizations": list(quantizations),
        "llama_cpp": str(llama_cpp) if llama_cpp else None,
        "steps": [],
        "artifacts": [],
    }

    if llama_cpp is None:
        report["blocked"] = True
        report["instructions"] = (
            "llama.cpp was not found. Install it, then re-run:\n"
            "    git clone https://github.com/ggml-org/llama.cpp\n"
            "    cd llama.cpp && cmake -B build && cmake --build build --config Release -j\n"
            "    export LLAMA_CPP_DIR=$PWD\n"
            f"    python training/export_model.py --merged {merged_dir} --outdir {outdir}"
        )
        log.warning("export.llama_cpp_missing")
        return report

    f16_path, step = convert_to_gguf(
        merged_dir, outdir, llama_cpp=llama_cpp, name=name, dry_run=dry_run
    )
    report["steps"].append({"stage": "convert_f16", **step})
    if f16_path.is_file():
        report["artifacts"].append(
            {
                "level": "F16",
                "path": str(f16_path),
                "size_gb": round(f16_path.stat().st_size / 1e9, 2),
            }
        )

    for level in quantizations:
        path, step = quantize(
            f16_path, outdir, level, llama_cpp=llama_cpp, name=name, dry_run=dry_run
        )
        report["steps"].append({"stage": f"quantize_{level}", **step})
        if path.is_file():
            report["artifacts"].append(
                {"level": level, "path": str(path), "size_gb": round(path.stat().st_size / 1e9, 2)}
            )

    report["blocked"] = False
    report["next_steps"] = [
        "Benchmark EVERY level before choosing one - do not pick the smallest file:",
        "    bash ollama/create_model.sh --gguf <path> --name khmer-support-9b-<level>",
        "    python -m evaluation.benchmark_model --compare khmer-support-9b-Q4_K_M,khmer-support-9b-Q5_K_M",
        "    python -m evaluation.evaluate_language --backend ollama --model khmer-support-9b-<level>",
        "Record quality, Khmer integrity, memory, tokens/sec and TTFT per level in",
        "reports/final_model_evaluation.md before selecting the production artifact.",
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python training/export_model.py")
    parser.add_argument("--merged", required=True, help="merged HF model directory")
    parser.add_argument("--outdir", default="models/gguf")
    parser.add_argument("--name", default=None, help="artifact base name")
    parser.add_argument("--quantize", default=",".join(DEFAULT_QUANTIZATIONS))
    parser.add_argument("--llama-cpp", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = parser.parse_args(argv)

    name = args.name or Path(args.merged).name.replace("-merged", "")
    levels = tuple(level.strip() for level in args.quantize.split(",") if level.strip())

    report = export(
        args.merged,
        args.outdir,
        name=name,
        quantizations=levels,
        llama_cpp_dir=args.llama_cpp,
        dry_run=args.dry_run,
    )
    write_json(Path(args.outdir) / "export_report.json", report)
    print(json.dumps(report, indent=2))
    return 1 if report.get("blocked") else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
