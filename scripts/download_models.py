"""Download embedding and reranker models to a local models/ directory.

Uses ModelScope first (works well in this region), then falls back to
HuggingFace with the hf-mirror.com endpoint.

Usage:
    python scripts/download_models.py                  # download all models
    python scripts/download_models.py --list           # list configured models and local status
    python scripts/download_models.py --missing        # download only missing models
    python scripts/download_models.py --model all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple


MODELS: List[Tuple[str, str]] = [
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2",
    ),
    (
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "cross-encoder-ms-marco-MiniLM-L6-v2",
    ),
    (
        "BAAI/bge-m3",
        "BAAI--bge-m3",
    ),
    (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ),
    (
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "Alibaba-NLP--gte-multilingual-reranker-base",
    ),
]


def models_dir() -> Path:
    return Path(__file__).parent.parent / "models"


def local_path(local_name: str) -> Path:
    return models_dir() / local_name


def is_downloaded(local_name: str) -> bool:
    """Check whether a model directory looks complete."""
    target = local_path(local_name)
    if not target.exists() or not target.is_dir():
        return False
    # A complete model should have at least a config.json
    return (target / "config.json").exists()


def dir_size(path: Path) -> str:
    """Return human-readable size of a directory."""
    if not path.exists():
        return "-"
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def list_models() -> None:
    """Print a table of configured models and their local status."""
    print(f"{'Local name':45s} {'Status':10s} {'Size':>10s}  {'Remote ID'}")
    print("-" * 110)
    for model_id, local_name in MODELS:
        status = "present" if is_downloaded(local_name) else "missing"
        size = dir_size(local_path(local_name))
        print(f"{local_name:45s} {status:10s} {size:>10s}  {model_id}")


def download_model(model_id: str, local_name: str) -> Path:
    target = local_path(local_name)
    target.mkdir(parents=True, exist_ok=True)

    # Try ModelScope first.
    try:
        from modelscope import snapshot_download

        print(f"Downloading {model_id} via ModelScope...")
        snapshot_download(model_id, local_dir=str(target))
        print(f"  -> saved to {target}")
        return target
    except Exception as e:
        print(f"  ModelScope failed: {e}")

    # Fallback: HuggingFace / hf-mirror.
    try:
        from huggingface_hub import snapshot_download

        mirrors = [
            "https://hf-mirror.com",
            None,
        ]
        for mirror in mirrors:
            if mirror:
                os.environ["HF_ENDPOINT"] = mirror
                print(f"Downloading {model_id} via {mirror}...")
            else:
                os.environ.pop("HF_ENDPOINT", None)
                print(f"Downloading {model_id} via HuggingFace...")

            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                )
                print(f"  -> saved to {target}")
                return target
            except Exception as e2:
                print(f"  failed: {e2}")
    except ImportError:
        print("huggingface_hub not installed, skipping HF fallback.")

    raise RuntimeError(f"Could not download {model_id}")


def resolve_model(query: str) -> Tuple[str, str]:
    """Resolve a user query to a (model_id, local_name) pair."""
    for model_id, local_name in MODELS:
        if query == model_id or query == local_name:
            return model_id, local_name
    raise ValueError(
        f"Unknown model '{query}'. Use --list to see available models."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models for rag-knowledge-base"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured models and their local status",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Download only models that are not already present",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="NAME",
        help="Download only the specified model(s) by local name or remote ID",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model:
        to_download = [resolve_model(q) for q in args.model]
    elif args.missing:
        to_download = [
            (mid, ln) for mid, ln in MODELS if not is_downloaded(ln)
        ]
    else:
        to_download = MODELS

    if not to_download:
        print("All requested models are already present.")
        return

    for model_id, local_name in to_download:
        if args.missing and is_downloaded(local_name):
            continue
        download_model(model_id, local_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
