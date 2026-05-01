#!/usr/bin/env python3
"""
Script to ensure all Kronos models are cached locally before starting the server.
Downloads missing models and then launches the server.

Compatible with Windows, macOS, and Linux.
"""

import sys
import os
import logging
from pathlib import Path

# Force UTF-8 output on Windows so Unicode status chars render correctly.
# Must happen before any print/logging calls.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
SERVER_DIR = REPO_ROOT
sys.path.insert(0, str(SERVER_DIR))

from constants.available_models import AVAILABLE_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_cache_dir() -> Path:
    """
    Return the HuggingFace hub cache directory, respecting the
    HF_HOME / HUGGINGFACE_HUB_CACHE env vars that users may have set.

    Priority (mirrors huggingface_hub's own resolution):
      1. HUGGINGFACE_HUB_CACHE
      2. HF_HOME / hub
      3. ~/.cache/huggingface/hub   (default on all platforms)
    """
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_snapshot_present(cache_dir: Path, model_id: str) -> bool:
    """
    Return True only when the model directory contains at least one
    complete snapshot (i.e. a non-empty snapshots/<revision> sub-dir).

    A model directory that exists but has no snapshots means a previous
    download was interrupted — treat it as not cached.
    """
    # HuggingFace stores models under  models--org--name/snapshots/<sha>/
    safe_id = model_id.replace("/", "--")
    model_dir = cache_dir / f"models--{safe_id}"

    if not model_dir.is_dir():
        return False

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False

    # At least one revision directory must be non-empty
    for revision_dir in snapshots_dir.iterdir():
        if revision_dir.is_dir() and any(revision_dir.iterdir()):
            return True

    return False


def check_and_download_model(model_id: str, config_name: str) -> dict:
    """Check if model is cached locally, download if not, and return status."""
    from huggingface_hub import snapshot_download

    result = {
        "model_id": model_id,
        "config_name": config_name,
        "cached": False,
        "status": "unknown",
    }

    cache_dir = get_hf_cache_dir()

    if is_model_snapshot_present(cache_dir, model_id):
        result["cached"] = True
        result["status"] = "already_cached"
        logger.info(f"[OK] [{config_name}] Model already cached: {model_id}")
        return result

    logger.info(f"[DL] [{config_name}] Downloading {model_id} ...")
    try:
        download_path = snapshot_download(model_id, local_files_only=False)
        result["cached"] = True
        result["status"] = "downloaded"
        logger.info(f"[OK] [{config_name}] Download complete: {download_path}")
    except Exception as e:
        result["status"] = f"failed: {e}"
        logger.error(f"[ERR] [{config_name}] Failed to download {model_id}: {e}")

    return result


def ensure_models_cached() -> dict:
    """Ensure all Kronos models are cached locally, download if needed."""
    results = {}

    logger.info("=" * 60)
    logger.info("Checking Kronos models availability ...")
    logger.info("=" * 60)

    for key, config in AVAILABLE_MODELS.items():
        model_id = config["model_id"]
        tokenizer_id = config["tokenizer_id"]
        param_size = config["params"]

        logger.info("")
        logger.info(f"[{key}]  ({param_size} params)")
        logger.info(f"   Model:     {model_id}")
        logger.info(f"   Tokenizer: {tokenizer_id}")

        model_result = check_and_download_model(model_id, f"{key}_model")
        tokenizer_result = check_and_download_model(tokenizer_id, f"{key}_tokenizer")

        results[key] = {
            "model": model_result,
            "tokenizer": tokenizer_result,
            "ready": model_result["cached"] and tokenizer_result["cached"],
        }

    return results


def start_server() -> None:
    """Start the uvicorn server.  Handles Ctrl-C gracefully on all platforms."""
    import subprocess

    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting server ...")
    logger.info("=" * 60)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload",
    ]

    try:
        # Using sys.executable guarantees we pick the same Python interpreter
        # (and therefore the same venv) on every platform — no reliance on PATH.
        subprocess.run(cmd, cwd=str(SERVER_DIR), check=True)
    except KeyboardInterrupt:
        # Ctrl-C on Windows raises KeyboardInterrupt from subprocess.run;
        # on Unix the signal is forwarded directly to uvicorn — either way,
        # we want a clean exit rather than a traceback.
        logger.info("")
        logger.info("Server stopped by user.")
    except subprocess.CalledProcessError as exc:
        logger.error(f"Server exited with code {exc.returncode}.")
        sys.exit(exc.returncode)


def main() -> None:
    logger.info("")
    logger.info("+" + "-" * 58 + "+")
    logger.info("|" + " " * 15 + "Kronos Model Downloader" + " " * 20 + "|")
    logger.info("+" + "-" * 58 + "+")
    logger.info("")

    results = ensure_models_cached()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)

    all_ready = True
    for key, info in results.items():
        status = "[OK] Ready" if info["ready"] else "[ERR] Missing"
        logger.info(f"  {key}: {status}")
        if not info["ready"]:
            all_ready = False

    if not all_ready:
        logger.error("")
        logger.error("Some models failed to download. Check errors above.")
        sys.exit(1)

    logger.info("")
    logger.info("All models are ready!")
    logger.info("")

    start_server()


if __name__ == "__main__":
    main()