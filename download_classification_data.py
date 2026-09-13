"""
download_classification_data.py — run this once on a login node before submitting
finetune_classification.py jobs — compute nodes on most clusters have no internet, and
`aeon`'s load_classification() downloads UCR/UEA data from timeseriesclassification.com on
first use.

Usage:
    python download_classification_data.py
    python download_classification_data.py --data_path /path/to/data/aeon_data
    python download_classification_data.py --datasets Chinatown ECG200
"""

import os
import argparse

from finetune_classification import DEFAULT_DATASETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='./data/aeon_data')
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    args = p.parse_args()

    from aeon.datasets import load_classification

    os.makedirs(args.data_path, exist_ok=True)
    print(f"Downloading classification datasets to: {args.data_path}")

    for name in args.datasets:
        try:
            X_train, y_train = load_classification(name, split='train', extract_path=args.data_path)
            X_test, y_test = load_classification(name, split='test', extract_path=args.data_path)
            print(f"[OK]   {name}: train={X_train.shape} test={X_test.shape} "
                  f"classes={sorted(set(y_train.tolist()))}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
