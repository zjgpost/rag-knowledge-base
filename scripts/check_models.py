"""Check which models are present locally and which are missing.

Usage:
    python scripts/check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/download_models importable
sys.path.insert(0, str(Path(__file__).parent))

from download_models import MODELS, dir_size, is_downloaded, local_path


def main() -> None:
    print(f"{'Local name':45s} {'Status':10s} {'Size':>10s}  {'Remote ID'}")
    print("-" * 110)
    missing = []
    for model_id, local_name in MODELS:
        present = is_downloaded(local_name)
        status = "present" if present else "missing"
        size = dir_size(local_path(local_name))
        print(f"{local_name:45s} {status:10s} {size:>10s}  {model_id}")
        if not present:
            missing.append(local_name)

    print("-" * 110)
    if missing:
        print(f"\nMissing models: {len(missing)}")
        print("Run one of:")
        print("  python scripts/download_models.py --missing")
        for name in missing:
            print(f"  python scripts/download_models.py --model {name}")
    else:
        print("\nAll configured models are present.")


if __name__ == "__main__":
    main()
