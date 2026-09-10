"""
Post-training benchmark report for the NE-Time foundation model.

Aggregates, per dataset x horizon:
  - NE-Time Foundation   (in-domain or zero-shot, from outputs_foundation/)
  - NE-Time Specialist   (per-dataset ceiling, from outputs/, if it exists)
  - Literature baselines (PatchTST / iTransformer, from baselines_specialist.json —
    transcribed from ne-time-results.png)
  - FM literature        (Chronos / TimesFM / Moirai zero-shot, from
    fm_literature_baselines.json if you've filled it in — see the .template.json)

Writes a markdown table (paper-ready) and a long-format CSV.

Usage:
  python benchmark_report.py --exp_name foundation_medium
  python benchmark_report.py --exp_name foundation_medium --metric MAE
"""

import os
import json
import argparse
from typing import Dict, Optional


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_specialist(specialist_dir: str, dataset: str, size: str) -> Optional[dict]:
    """Load the specialist per-horizon ceiling for a dataset, if it exists.
    Matches the naming convention in outputs/ (see train.py's per_horizon+CI
    exp_name construction)."""
    path = os.path.join(specialist_dir, f"netime_{dataset}_{size}_perH_CI_CI_per_horizon_results.json")
    return load_json(path)


def cell(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "—"


def main():
    p = argparse.ArgumentParser(description="NE-Time foundation model benchmark report")
    p.add_argument('--foundation_dir', type=str, default='./outputs_foundation')
    p.add_argument('--exp_name',       type=str, default='foundation_medium')
    p.add_argument('--specialist_dir', type=str, default='./outputs')
    p.add_argument('--specialist_size', type=str, default='medium')
    p.add_argument('--baselines_file',  type=str, default='./baselines_specialist.json')
    p.add_argument('--fm_literature_file', type=str, default='./fm_literature_baselines.json')
    p.add_argument('--metric', type=str, default='MSE', choices=['MSE', 'MAE', 'RMSE'])
    p.add_argument('--horizons', type=int, nargs='+', default=[96, 192, 336, 720])
    p.add_argument('--output_dir', type=str, default='./outputs_foundation')
    args = p.parse_args()

    indomain = load_json(os.path.join(args.foundation_dir, f"{args.exp_name}_indomain_results.json")) or {}
    zeroshot = load_json(os.path.join(args.foundation_dir, f"{args.exp_name}_zeroshot_results.json")) or {}
    baselines = load_json(args.baselines_file) or {"datasets": {}}
    fm_lit = load_json(args.fm_literature_file) or {"datasets": {}}

    if not indomain and not zeroshot:
        print(f"No foundation results found under {args.foundation_dir} for "
              f"exp_name='{args.exp_name}'. Run train_foundation.py (or "
              f"eval_foundation.py) first.")
        return

    all_datasets = [(name, 'in-domain') for name in indomain] + \
                   [(name, 'zero-shot') for name in zeroshot]

    md_lines = [
        f"# NE-Time Foundation Model — Benchmark Report ({args.metric})",
        "",
        f"Foundation exp: `{args.exp_name}`  |  Specialist size: `{args.specialist_size}`",
        "",
    ]
    csv_rows = ["dataset,protocol,horizon,model,metric,value"]

    for name, protocol in all_datasets:
        results = indomain[name] if protocol == 'in-domain' else zeroshot[name]
        specialist = get_specialist(args.specialist_dir, name, args.specialist_size)
        base_models = baselines.get('datasets', {}).get(name, {}).get('models', {})
        fm_models = fm_lit.get('datasets', {}).get(name, {}).get('models', {})

        # Skip NE-Time entries in the baselines file itself — those are the
        # same specialist numbers already shown in the "NE-Time Specialist"
        # column (pulled live from outputs/), just transcribed from the png
        # as a fallback for when outputs/ isn't present in this checkout.
        other_model_names = sorted(
            n for n in (set(base_models) | set(fm_models))
            if n.lower() != 'ne-time'
        )
        header = ["Horizon", f"NE-Time Foundation ({protocol})", "NE-Time Specialist"] + other_model_names
        md_lines.append(f"## {name} ({protocol})")
        md_lines.append("")
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "---|" * len(header))

        # Fallback specialist source: the png-transcribed "NE-Time" row in
        # baselines_specialist.json, used only when outputs/ (live specialist
        # checkpoints) isn't present in this checkout.
        base_ne_time = {k: v for k, v in base_models.items() if k.lower() == 'ne-time'}
        base_ne_time = next(iter(base_ne_time.values()), {})

        for H in args.horizons:
            Hs = str(H)
            found_val = results.get(Hs, {}).get(args.metric)
            spec_val = None
            if specialist and Hs in specialist:
                spec_val = specialist[Hs].get(args.metric)
            elif args.metric == 'MSE' and Hs in base_ne_time:
                spec_val = base_ne_time[Hs]

            row = [Hs, cell(found_val), cell(spec_val)]
            csv_rows.append(f"{name},{protocol},{H},NE-Time-Foundation,{args.metric},"
                             f"{found_val if found_val is not None else ''}")
            csv_rows.append(f"{name},{protocol},{H},NE-Time-Specialist,{args.metric},"
                             f"{spec_val if spec_val is not None else ''}")

            for m in other_model_names:
                src = base_models.get(m) or fm_models.get(m) or {}
                v = src.get(Hs) if isinstance(src, dict) else None
                row.append(cell(v))
                csv_rows.append(f"{name},{protocol},{H},{m},{args.metric},{v if v is not None else ''}")

            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

    md_report = "\n".join(md_lines)
    md_path = os.path.join(args.output_dir, f"{args.exp_name}_benchmark_report.md")
    csv_path = os.path.join(args.output_dir, f"{args.exp_name}_benchmark_report.csv")

    with open(md_path, 'w') as f:
        f.write(md_report)
    with open(csv_path, 'w') as f:
        f.write("\n".join(csv_rows))

    print(md_report)
    print(f"\nSaved: {md_path}")
    print(f"Saved: {csv_path}")
    if not os.path.exists(args.fm_literature_file):
        print(f"\nNote: {args.fm_literature_file} not found — FM literature columns "
              f"(Chronos/TimesFM/Moirai) omitted. Copy fm_literature_baselines.template.json "
              f"and fill in published numbers to include them.")


if __name__ == "__main__":
    main()
