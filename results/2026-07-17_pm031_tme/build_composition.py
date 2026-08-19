#!/usr/bin/env python3
"""Build the ABM abundance-fraction matrix for the PM-031 presim-TME sweep.

For every theta's snapshot, compute the 19 matched-category composition fractions
(the ``frac__<cat>`` block of the SBI observable) at presim=0 and presim=stop, and
cache them to composition.npz. This is the input to the histogram comparison against
PM-031's real IMC composition.

Only the 19 fractions are computed -- the expensive FunCN KDE is skipped -- but the
category mapping and the mid-z-slice restriction are byte-for-byte the same as
observables._abm_positions_2d, so these match compute_observables(...)[:19] exactly.

Robust to a PARTIAL pull and RESUMABLE: it reads whatever snaps_shard_*.tar are present
and appends only thetas not already cached. Re-run as more shards land.

Usage:
  python build_composition.py \
      --snaps ../../abm-inference/analysis/pm031_tme \
      --out composition.npz
"""
from __future__ import annotations

import argparse
import io as _io
import tarfile
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(_REPO / "PDAC"))

from tools.imc_mapping import ABM_TO_MATCHED, MATCHED_CATEGORIES  # noqa: E402
from tools import io as _IO  # noqa: E402
_NCAT = len(MATCHED_CATEGORIES)
_CAT_IX = {c: i for i, c in enumerate(MATCHED_CATEGORIES)}

# the two endpoints kept per theta (see run_shard_pm031.py)
SNAP_ZERO = "abm/agents_presim_000001.abm.lz4"
SNAP_STOP = "abm/agents_step_000000.abm.lz4"


def composition_from_lz4(path: str) -> tuple[np.ndarray, int]:
    """(19,) fractions on the mid-z slice + total matched-cell count. NaN if empty.

    Mirrors observables._abm_positions_2d exactly: add_category(type_state) ->
    ABM_TO_MATCHED -> drop None -> keep z == gz//2 -> counts / total.
    """
    df = _IO.load_abm_lz4(path)
    gz = int(df["z"].max()) + 1 if len(df) else 1
    cat = _IO.add_category(df, mode="type_state")
    cat = cat.assign(_m=cat["category"].map(ABM_TO_MATCHED)).dropna(subset=["_m"])
    cat = cat[cat["z"] == gz // 2]
    counts = np.zeros(_NCAT, dtype=np.float64)
    for m, n in cat["_m"].value_counts().items():
        counts[_CAT_IX[str(m)]] = n
    total = counts.sum()
    if total == 0:
        return np.full(_NCAT, np.nan), 0
    return counts / total, int(total)


def theta_idx_of(member_name: str) -> int | None:
    # members look like ./theta_<gi>/abm/agents_step_000000.abm.lz4
    for part in member_name.split("/"):
        if part.startswith("theta_"):
            try:
                return int(part[len("theta_"):])
            except ValueError:
                return None
    return None


def _extract_to_tmp(tar: tarfile.TarFile, member: tarfile.TarInfo, tmp: Path) -> Path | None:
    f = tar.extractfile(member)
    if f is None:
        return None
    out = tmp / f"snap_{abs(hash(member.name)) % (10**9)}.abm.lz4"
    out.write_bytes(f.read())
    return out


def process_tar(tar_path: Path, seen: set[int], tmp: Path) -> dict[int, dict]:
    """{gi: {'zero': (19,), 'stop': (19,), 'n_stop': int}} for new thetas in this tar."""
    rows: dict[int, dict] = {}
    with tarfile.open(tar_path, "r") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        # group the two endpoint files by theta index
        by_theta: dict[int, dict[str, tarfile.TarInfo]] = {}
        for name, m in members.items():
            gi = theta_idx_of(name)
            if gi is None or gi in seen:
                continue
            if name.endswith(SNAP_ZERO):
                by_theta.setdefault(gi, {})["zero"] = m
            elif name.endswith(SNAP_STOP):
                by_theta.setdefault(gi, {})["stop"] = m
        for gi, mm in by_theta.items():
            rec: dict = {"zero": np.full(_NCAT, np.nan), "stop": np.full(_NCAT, np.nan),
                         "n_stop": 0}
            try:
                if "zero" in mm:
                    p = _extract_to_tmp(tar, mm["zero"], tmp)
                    if p:
                        rec["zero"], _ = composition_from_lz4(str(p))
                        p.unlink(missing_ok=True)
                if "stop" in mm:
                    p = _extract_to_tmp(tar, mm["stop"], tmp)
                    if p:
                        rec["stop"], rec["n_stop"] = composition_from_lz4(str(p))
                        p.unlink(missing_ok=True)
                rows[gi] = rec
            except Exception as e:
                print(f"    theta {gi}: ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", type=Path, required=True, help="dir of snaps_shard_*.tar")
    ap.add_argument("--out", type=Path, default=Path("composition.npz"))
    args = ap.parse_args()

    idx: list[int] = []
    comp_zero: list[np.ndarray] = []
    comp_stop: list[np.ndarray] = []
    n_stop: list[int] = []
    if args.out.exists():
        d = np.load(args.out, allow_pickle=True)
        idx = list(d["theta_idx"])
        comp_zero = list(d["comp_zero"])
        comp_stop = list(d["comp_stop"])
        n_stop = list(d["n_stop"])
        print(f"resuming: {len(idx)} theta already cached")
    seen = set(idx)

    tars = sorted(args.snaps.glob("snaps_shard_*.tar"))
    print(f"found {len(tars)} shard tars in {args.snaps}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k, tp in enumerate(tars):
            rows = process_tar(tp, seen, tmp)
            for gi, rec in rows.items():
                idx.append(gi); comp_zero.append(rec["zero"])
                comp_stop.append(rec["stop"]); n_stop.append(rec["n_stop"]); seen.add(gi)
            if rows:
                print(f"  [{k+1}/{len(tars)}] {tp.name}: +{len(rows)} theta "
                      f"(total {len(idx)})", flush=True)
            if (k + 1) % 10 == 0:
                _save(args.out, idx, comp_zero, comp_stop, n_stop)

    _save(args.out, idx, comp_zero, comp_stop, n_stop)
    print(f"\nwrote {args.out}: {len(idx)} theta x {_NCAT} categories")


def _save(out, idx, cz, cs, ns):
    np.savez(out,
             theta_idx=np.array(idx, dtype=np.int64),
             comp_zero=np.array(cz, dtype=np.float64),
             comp_stop=np.array(cs, dtype=np.float64),
             n_stop=np.array(ns, dtype=np.int64),
             categories=np.array(MATCHED_CATEGORIES))


if __name__ == "__main__":
    main()
