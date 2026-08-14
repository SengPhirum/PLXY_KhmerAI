#!/usr/bin/env python3
"""Environment doctor (Phase 1).

    make doctor
    python scripts/doctor.py --json
    python scripts/doctor.py --strict     # non-zero exit on any warning

Checks Python, RAM, OS, architecture, Ollama availability/version/connectivity,
required directories, free disk, environment variables, Apple Metal information
on macOS, and CUDA availability on Linux/Colab.

Every check reports the exact command that fixes it.  A doctor that says
"something is wrong" without saying what to do is a worse experience than no
doctor at all.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"

REQUIRED_DIRECTORIES = (
    "configs",
    "prompts",
    "data/raw/public",
    "data/raw/company",
    "data/interim",
    "data/cleaned",
    "data/sft",
    "data/manifests",
    "data/index",
    "evaluation/golden",
    "reports",
)
# (name, required, description)
ENVIRONMENT_VARIABLES = (
    ("KHMERAI_ENV", False, "development | staging | production"),
    (
        "KHMERAI_ADMIN_API_KEY",
        False,
        "required in production; admin endpoints refuse to run without it",
    ),
    ("KHMERAI_OLLAMA_BASE_URL", False, "defaults to http://127.0.0.1:11434"),
    ("KHMERAI_OLLAMA_MODEL", False, "defaults to khmer-support-9b"),
    ("KHMERAI_EMBEDDING_BACKEND", False, "ollama | sentence_transformers | hashing"),
    ("HF_TOKEN", False, "needed only for gated Hugging Face datasets/models"),
)

MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info
    detail: str
    fix: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _run(argv: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout or result.stderr).strip()


# --- checks -----------------------------------------------------------------
def check_python() -> Check:
    version = sys.version_info[:3]
    text = ".".join(str(v) for v in version)
    if MIN_PYTHON <= version[:2] <= MAX_PYTHON:
        return Check("python", "ok", f"{text} ({sys.executable})")
    return Check(
        "python",
        "fail" if version[:2] < MIN_PYTHON else "warn",
        f"{text} - validated range is 3.11-3.12",
        "brew install python@3.11   # macOS\n    sudo apt install python3.11  # Linux",
    )


def check_virtualenv() -> Check:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return Check("virtualenv", "ok", sys.prefix)
    return Check(
        "virtualenv",
        "warn",
        "not running inside a virtual environment",
        "make setup && source .venv/bin/activate",
    )


def check_os() -> Check:
    return Check(
        "os",
        "ok",
        f"{platform.system()} {platform.release()} ({platform.machine()})",
        data={"system": platform.system(), "machine": platform.machine()},
    )


def check_memory() -> Check:
    total_gb = 0.0
    try:
        if platform.system() == "Darwin":
            code, out = _run(["sysctl", "-n", "hw.memsize"])
            total_gb = int(out) / 1e9 if code == 0 and out.isdigit() else 0.0
        else:
            meminfo = Path("/proc/meminfo")
            if meminfo.is_file():
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1e6
                        break
    except (OSError, ValueError):
        total_gb = 0.0

    if total_gb == 0:
        return Check("memory", "warn", "could not determine total RAM")
    if total_gb >= 40:
        return Check("memory", "ok", f"{total_gb:.0f} GB", data={"total_gb": round(total_gb, 1)})
    if total_gb >= 16:
        return Check(
            "memory",
            "warn",
            f"{total_gb:.0f} GB - the 9B profile assumes 48 GB",
            "Use the 4B model, or lower OLLAMA_NUM_PARALLEL and OLLAMA_CONTEXT_LENGTH.",
            {"total_gb": round(total_gb, 1)},
        )
    return Check(
        "memory",
        "fail",
        f"{total_gb:.0f} GB is not enough to serve a 4B model",
        data={"total_gb": round(total_gb, 1)},
    )


def check_disk() -> Check:
    usage = shutil.disk_usage(_REPO_ROOT)
    free_gb = usage.free / 1e9
    data = {"free_gb": round(free_gb, 1), "total_gb": round(usage.total / 1e9, 1)}
    if free_gb >= 60:
        return Check("disk", "ok", f"{free_gb:.0f} GB free", data=data)
    if free_gb >= 20:
        return Check(
            "disk",
            "warn",
            f"{free_gb:.0f} GB free - GGUF conversion needs roughly 60 GB",
            "Free space, or export on a different host.",
            data,
        )
    return Check("disk", "fail", f"{free_gb:.0f} GB free", "Free at least 20 GB.", data)


def check_ollama() -> list[Check]:
    checks: list[Check] = []
    binary = shutil.which("ollama")
    if binary is None:
        checks.append(
            Check(
                "ollama.binary",
                "warn",
                "not installed",
                "brew install ollama   # macOS\n    curl -fsSL https://ollama.com/install.sh | sh",
            )
        )
    else:
        code, out = _run([binary, "--version"])
        checks.append(
            Check(
                "ollama.binary",
                "ok" if code == 0 else "warn",
                out.splitlines()[0] if out else binary,
            )
        )

    host = (
        os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        .replace("http://", "")
        .replace("https://", "")
    )
    hostname, _, port = host.partition(":")
    port_number = int(port) if port.isdigit() else 11434
    try:
        with socket.create_connection((hostname or "127.0.0.1", port_number), timeout=2):
            reachable = True
    except OSError:
        reachable = False

    if not reachable:
        checks.append(
            Check(
                "ollama.daemon",
                "warn",
                f"not reachable on {host}",
                "bash ollama/start_server.sh",
            )
        )
        return checks

    checks.append(Check("ollama.daemon", "ok", f"reachable on {host}"))
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://{host}/api/tags", timeout=3) as response:
            models = [m.get("name", "") for m in json.load(response).get("models", [])]
    except Exception as exc:
        checks.append(Check("ollama.models", "warn", f"could not list models: {exc}"))
        return checks

    support = [m for m in models if m.startswith("khmer-support")]
    if support:
        checks.append(Check("ollama.models", "ok", ", ".join(support), data={"models": models}))
    else:
        checks.append(
            Check(
                "ollama.models",
                "warn",
                f"no khmer-support model ({len(models)} other model(s) present)",
                "bash ollama/create_model.sh",
                {"models": models},
            )
        )
    return checks


def check_directories() -> Check:
    missing = [d for d in REQUIRED_DIRECTORIES if not (_REPO_ROOT / d).is_dir()]
    if not missing:
        return Check("directories", "ok", f"all {len(REQUIRED_DIRECTORIES)} present")
    return Check(
        "directories",
        "warn",
        f"missing: {', '.join(missing)}",
        "mkdir -p " + " ".join(missing),
    )


def check_environment() -> list[Check]:
    checks: list[Check] = []
    env_file = _REPO_ROOT / ".env"
    if env_file.is_file():
        mode = oct(env_file.stat().st_mode)[-3:]
        if mode in ("600", "400"):
            checks.append(Check("env.file", "ok", f".env present (mode {mode})"))
        else:
            checks.append(
                Check(
                    "env.file",
                    "warn",
                    f".env is mode {mode} - it contains secrets",
                    "chmod 600 .env",
                )
            )
    else:
        checks.append(Check("env.file", "warn", ".env not found", "cp .env.example .env"))

    for name, required, description in ENVIRONMENT_VARIABLES:
        value = os.environ.get(name, "")
        if value and not value.startswith("CHANGE_ME"):
            shown = "***set***" if any(k in name for k in ("KEY", "TOKEN", "SECRET")) else value
            checks.append(Check(f"env.{name}", "ok", shown))
        elif required:
            checks.append(Check(f"env.{name}", "fail", f"not set - {description}"))
        else:
            checks.append(Check(f"env.{name}", "info", f"not set - {description}"))
    return checks


def check_accelerator() -> Check:
    system = platform.system()
    if system == "Darwin":
        code, out = _run(["system_profiler", "SPDisplaysDataType"], timeout=15)
        if code == 0 and out:
            chipset = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in out.splitlines()
                    if "Chipset Model" in line
                ),
                "",
            )
            cores = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in out.splitlines()
                    if "Total Number of Cores" in line
                ),
                "",
            )
            detail = f"Metal: {chipset or 'Apple GPU'}" + (f", {cores} GPU cores" if cores else "")
            return Check("accelerator", "ok", detail)
        return Check("accelerator", "info", "Apple silicon; Metal details unavailable")

    try:
        import torch

        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            memory = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            return Check(
                "accelerator",
                "ok",
                f"CUDA {torch.version.cuda}: {', '.join(names)} ({memory} GB)",
                data={"gpus": names, "vram_gb": memory},
            )
        return Check(
            "accelerator",
            "info",
            "torch is installed but no CUDA device is visible",
            "Training needs a GPU. On Colab: Runtime -> Change runtime type -> GPU.",
        )
    except ImportError:
        return Check(
            "accelerator",
            "info",
            "torch not installed (serving does not need it)",
            "pip install -r requirements/training.txt   # only on a training host",
        )


def check_dependencies() -> list[Check]:
    checks: list[Check] = []
    groups = {
        "core": ["pydantic", "yaml", "numpy", "httpx"],
        "server": ["fastapi", "uvicorn"],
        "rag": ["pypdf", "docx", "openpyxl", "bs4"],
        "dev": ["pytest", "ruff"],
    }
    for group, modules in groups.items():
        missing = []
        for module in modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if not missing:
            checks.append(Check(f"deps.{group}", "ok", f"{len(modules)} module(s)"))
        else:
            requirement = "base" if group == "core" else group
            checks.append(
                Check(
                    f"deps.{group}",
                    "warn" if group != "core" else "fail",
                    f"missing: {', '.join(missing)}",
                    f"pip install -r requirements/{requirement}.txt",
                )
            )
    return checks


def check_index() -> Check:
    pointer = _REPO_ROOT / "data" / "index" / "ACTIVE"
    if not (pointer.exists() or pointer.is_symlink()):
        return Check(
            "knowledge_index",
            "warn",
            "no active index",
            "python -m company_data.validate --input data/raw/company "
            "--output data/interim/company_records.jsonl --report data/manifests/company_validation.json\n"
            "    python -m rag.reindex --input data/interim/company_records.jsonl --activate",
        )
    try:
        version = os.readlink(pointer) if pointer.is_symlink() else pointer.read_text().strip()
    except OSError as exc:
        return Check("knowledge_index", "warn", f"unreadable ACTIVE pointer: {exc}")

    manifest = _REPO_ROOT / "data" / "index" / version / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return Check(
            "knowledge_index",
            "ok",
            f"{version}: {data.get('chunks', 0)} chunks from {data.get('documents', 0)} documents",
            data={"index_version": version},
        )
    return Check("knowledge_index", "warn", f"ACTIVE points at {version}, which has no manifest")


def run_all() -> list[Check]:
    checks: list[Check] = [
        check_python(),
        check_virtualenv(),
        check_os(),
        check_memory(),
        check_disk(),
    ]
    checks += check_dependencies()
    checks += check_ollama()
    checks.append(check_accelerator())
    checks.append(check_directories())
    checks += check_environment()
    checks.append(check_index())
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/doctor.py")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    args = parser.parse_args(argv)

    checks = run_all()

    if args.as_json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        symbols = {
            "ok": f"{GREEN}[ok]{RESET}  ",
            "warn": f"{YELLOW}[warn]{RESET}",
            "fail": f"{RED}[fail]{RESET}",
            "info": f"{DIM}[info]{RESET}",
        }
        print("Khmer Customer-Support LLM - environment doctor\n")
        for check in checks:
            print(f"  {symbols[check.status]} {check.name:24s} {check.detail}")
            if check.fix and check.status in ("warn", "fail"):
                for line in check.fix.splitlines():
                    print(f"           {DIM}-> {line.strip()}{RESET}")

        failures = sum(1 for c in checks if c.status == "fail")
        warnings = sum(1 for c in checks if c.status == "warn")
        print(
            f"\n  {len(checks)} checks: "
            f"{sum(1 for c in checks if c.status == 'ok')} ok, {warnings} warnings, {failures} failures"
        )
        if failures:
            print(f"\n  {RED}Resolve the failures above before continuing.{RESET}")
        elif warnings:
            print(
                f"\n  {YELLOW}Warnings are expected on a fresh checkout - see the fixes above.{RESET}"
            )
        else:
            print(f"\n  {GREEN}Everything is ready.{RESET}")

    failures = sum(1 for c in checks if c.status == "fail")
    warnings = sum(1 for c in checks if c.status == "warn")
    if failures:
        return 1
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
