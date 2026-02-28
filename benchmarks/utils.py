from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any


def clean_method_name(name: str) -> str:
    """Clean and shorten method names for report display."""
    if not name:
        return "N/A"
    return (
        name.replace("-Apply", "")
        .replace("Torch-", "T-")
        .replace("-ReuseFactor", "-Reuse")
        .replace("-Dual-Gram-RHS", "-Dual-Gram")
        .replace("-Primal-Gram-RHS", "-Primal-Gram")
    )


def format_scientific(val: float) -> str:
    """Standardize scientific notation for reports."""
    if val == 0:
        return "0.0"
    if 0.01 <= abs(val) <= 1000:
        return f"{val:.4f}"
    return f"{val:.1e}"


def get_git_metadata(repo_root: str) -> dict[str, Any]:
    """Gather git commit and branch information."""

    def _run(cmd: list[str]) -> str | None:
        try:
            out = subprocess.check_output(
                cmd,
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or None
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": (_run(["git", "status", "--porcelain"]) or "") != "",
    }


def get_repro_context(repo_root: str, args: Any) -> dict[str, Any]:
    """Standardize reproducibility context for manifests."""
    import torch

    return {
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "git": get_git_metadata(repo_root),
    }


def stable_json_sha256(payload: Any) -> str:
    """Generate a stable SHA256 hash for a JSON-serializable object."""
    dump = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def write_text_file(path: str, text: str) -> None:
    """Write text to a file, ensuring directories exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json_file(path: str, payload: dict[str, Any]) -> None:
    """Write JSON payload to a file."""
    write_text_file(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_sha256_sidecar(path: str) -> str:
    """Write a .sha256 sidecar for the given file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    sidecar = f"{path}.sha256"
    write_text_file(sidecar, f"{digest}  {os.path.basename(path)}\n")
    return digest


def repo_relative(path: str, repo_root: str) -> str:
    """Return a repo-relative path with forward slashes."""
    return os.path.relpath(path, repo_root).replace("\\", "/")


def sha256_text(text: str) -> str:
    """SHA256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """SHA256 of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def format_timestamp() -> str:
    """Return a standard ISO timestamp."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_repro_fingerprint_sidecar(manifest_path: str, fingerprint: str) -> str:
    """Write a .repro.sha256 sidecar."""
    sidecar = f"{manifest_path}.repro.sha256"
    write_text_file(sidecar, f"{fingerprint}  reproducibility_fingerprint\n")
    return sidecar
