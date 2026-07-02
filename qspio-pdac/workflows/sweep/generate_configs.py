#!/usr/bin/env python
"""Generate per-cell JSON configs from grid.yaml for the NPE sweep driver."""
import argparse
import itertools
import json
from pathlib import Path

import yaml


def _short_k(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(root / "grid.yaml"))
    parser.add_argument("--out-dir", default=str(root / "configs"))
    args = parser.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text())
    fixed = grid["fixed"]
    axes = grid["axes"]

    configs_dir = Path(args.out_dir)
    configs_dir.mkdir(exist_ok=True, parents=True)

    keys = list(axes.keys())
    # Short-name prefix per grid axis — extend when new axes are introduced.
    tag_prefix = {
        'hidden_features': 'h',
        'num_transforms': 't',
        'n_train_subsample': 's',
    }
    written = []
    for values in itertools.product(*(axes[k] for k in keys)):
        cell = dict(zip(keys, values))
        parts = []
        for k in keys:
            v = cell[k]
            prefix = tag_prefix.get(k, k + '_')
            parts.append(f"{prefix}{_short_k(v) if isinstance(v, int) and k == 'n_train_subsample' else v}")
        tag = '_'.join(parts)
        payload = {**fixed, **cell}
        (configs_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
        written.append(tag)

    tags_txt = "\n".join(written) + "\n"
    (configs_dir / "_tags.txt").write_text(tags_txt)
    print(f"Wrote {len(written)} configs + _tags.txt to {configs_dir}/")


if __name__ == "__main__":
    main()
