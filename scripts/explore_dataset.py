"""Explore the SEA e-commerce customer support dataset."""

from __future__ import annotations

from datasets import load_dataset


def main():
    ds = load_dataset("nwchang/sea-ecommerce-customer-support-sample", trust_remote_code=True)
    print("Splits:", list(ds.keys()))
    for split in ds.keys():
        print(f"\nSplit: {split}, size: {len(ds[split])}")
        print("Columns:", ds[split].column_names)
        print("First example:")
        for k, v in ds[split][0].items():
            print(f"  {k}: {v[:200] if isinstance(v, str) else v}")


if __name__ == "__main__":
    main()
