"""
download_datasets.py — run this once on a login node before submitting jobs.

Downloads all benchmark datasets from HuggingFace (thuml/Time-Series-Library)
into ./data/ using huggingface_hub, which handles redirects and auth correctly.

Usage:
    python download_datasets.py
    python download_datasets.py --data_path /path/to/data
    python download_datasets.py --datasets Weather Exchange  # subset
"""

import os
import argparse
import shutil

DATASETS = {
    # ETT family — already available via raw GitHub, kept here for completeness
    "ETTh1": ("ETT-small/ETTh1.csv", "ETTh1.csv"),
    "ETTh2": ("ETT-small/ETTh2.csv", "ETTh2.csv"),
    "ETTm1": ("ETT-small/ETTm1.csv", "ETTm1.csv"),
    "ETTm2": ("ETT-small/ETTm2.csv", "ETTm2.csv"),
    # New benchmark datasets
    "Weather":  ("weather/weather.csv",           "Weather.csv"),
    "Exchange": ("exchange_rate/exchange_rate.csv","Exchange.csv"),
    "ECL":      ("electricity/electricity.csv",   "ECL.csv"),
    "Traffic":  ("traffic/traffic.csv",           "Traffic.csv"),
}

REPO_ID = "thuml/Time-Series-Library"


def download_all(data_path: str, datasets: list):
    os.makedirs(data_path, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub", "--quiet"])
        from huggingface_hub import hf_hub_download

    for name in datasets:
        if name not in DATASETS:
            print(f"[SKIP] Unknown dataset '{name}'. "
                  f"Known: {sorted(DATASETS.keys())}")
            continue

        repo_path, local_name = DATASETS[name]
        local_path = os.path.join(data_path, local_name)

        if os.path.exists(local_path):
            print(f"[SKIP] {name} already exists at {local_path}")
            continue

        print(f"[DOWN] {name}  ←  {REPO_ID}/{repo_path}")
        try:
            cached = hf_hub_download(
                repo_id   = REPO_ID,
                filename  = repo_path,
                repo_type = "dataset",
                local_dir = data_path,
            )
            # hf_hub_download may nest the file under the repo sub-path;
            # move it to the flat data_path/<name>.csv location our code expects
            nested = os.path.join(data_path, repo_path)
            if os.path.exists(nested) and nested != local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.move(nested, local_path)
                print(f"       moved {nested} → {local_path}")
            elif cached != local_path and os.path.exists(cached):
                shutil.copy2(cached, local_path)
                print(f"       copied {cached} → {local_path}")

            # Verify
            import csv
            with open(local_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                row1   = next(reader)
            print(f"[OK]   {name}: {len(header)} columns, "
                  f"first row sample: {row1[:3]}...")

        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            # Clean up partial
            for p in [local_path, os.path.join(data_path, repo_path)]:
                if os.path.exists(p):
                    os.remove(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data")
    p.add_argument("--datasets",  nargs="+",
                   default=list(DATASETS.keys()),
                   help="Which datasets to download (default: all)")
    args = p.parse_args()

    print(f"Downloading to: {os.path.abspath(args.data_path)}")
    print(f"Datasets: {args.datasets}\n")
    download_all(args.data_path, args.datasets)
    print("\nDone.")


if __name__ == "__main__":
    main()