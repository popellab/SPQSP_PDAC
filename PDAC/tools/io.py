"""ABM snapshot loaders + category helpers.

Schema of .abm.lz4 files is documented in outputs/abm_lz4_def.txt:
  int32 columns = [type_id, agent_id, x, y, z, cell_state, life, extra]

type_id: 0=CANCER 1=TCELL 2=TREG 3=MDSC 4=MAC 5=FIB 6=VAS 7=BCELL 8=DC

DCs additionally encode subtype (cDC1/cDC2) in the `extra` column; BCELLs
encode is_breg in `extra`.
"""
from __future__ import annotations

import glob
import os
import struct
from typing import Iterator

import lz4.block
import numpy as np
import pandas as pd

COLUMNS = ["type", "id", "x", "y", "z", "state", "life", "extra"]

TYPE_CANCER = 0
TYPE_TCELL  = 1
TYPE_TREG   = 2
TYPE_MDSC   = 3
TYPE_MAC    = 4
TYPE_FIB    = 5
TYPE_VAS    = 6
TYPE_BCELL  = 7
TYPE_DC     = 8

TYPE_NAMES = {
    TYPE_CANCER: "cancer",
    TYPE_TCELL:  "tcell",
    TYPE_TREG:   "treg",
    TYPE_MDSC:   "mdsc",
    TYPE_MAC:    "mac",
    TYPE_FIB:    "fib",
    TYPE_VAS:    "vas",
    TYPE_BCELL:  "bcell",
    TYPE_DC:     "dc",
}

# (type, state) → category label. Keep lowercase snake-case for consistency.
_STATE_NAMES = {
    TYPE_CANCER: {0: "stem", 1: "prog", 2: "sen"},
    TYPE_TCELL:  {0: "eff",  1: "cyt",  2: "sup", 3: "naive"},
    TYPE_TREG:   {0: "th",   1: "treg", 2: "tfh", 3: "tcd4_naive"},
    TYPE_MDSC:   {0: ""},  # single state
    TYPE_MAC:    {0: "M1",  1: "M2"},
    TYPE_FIB:    {0: "quiescent", 1: "myCAF", 2: "iCAF", 3: "FRC"},
    TYPE_VAS:    {0: "tip", 2: "phalanx", 3: "collapsed", 4: "hev"},
    TYPE_BCELL:  {0: "naive", 1: "act", 2: "plasma"},
    TYPE_DC:     {0: "immature", 1: "mature"},
}

# DC subtype is encoded in the `extra` column (not `state`).
_DC_SUBTYPES = {0: "cDC1", 1: "cDC2"}

# PDE substrate order — must match enum ChemicalSubstrate in PDAC/pde/pde_solver.cuh.
# O2 is [mM]; all others are [nM].
SUBSTRATE_NAMES = [
    "O2", "IFN", "IL2", "IL10", "TGFB", "CCL2", "ArgI", "NO", "IL12", "VEGFA",
    "IL1", "IL6", "CXCL13", "MMP", "Antibody", "CCL21", "CXCL12", "CCL5",
]
SUBSTRATE_UNITS = {name: ("mM" if name == "O2" else "nM") for name in SUBSTRATE_NAMES}


def load_abm_lz4(path: str) -> pd.DataFrame:
    """Decode a single .abm.lz4 snapshot into a DataFrame."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ABM1":
            raise ValueError(f"{path}: bad magic {magic!r}, expected b'ABM1'")
        n, nc, raw_sz, _ = struct.unpack("<4i", f.read(16))
        data = np.frombuffer(lz4.block.decompress(f.read(), raw_sz), dtype=np.int32)
    arr = data.reshape(n, nc)
    # Some snapshots may have n_cols > 8 in future — keep only the defined ones.
    arr = arr[:, :len(COLUMNS)]
    return pd.DataFrame(arr, columns=COLUMNS)


def iter_abm_series(directory: str, pattern: str = "agents_step_*.abm.lz4") -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (step_idx, df) pairs in step-order from a directory of snapshots."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    for p in paths:
        base = os.path.basename(p)
        # Accept both 'agents_step_000012.abm.lz4' and 'agents_presim_000007.abm.lz4'
        stem = base.split(".")[0]
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            continue
        yield step, load_abm_lz4(p)


def load_pde_lz4(path: str) -> np.ndarray:
    """Decode a single .pde.lz4 snapshot into a (ns, gz, gy, gx) float32 array.

    Substrate order matches ``SUBSTRATE_NAMES``.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"PDE1":
            raise ValueError(f"{path}: bad magic {magic!r}, expected b'PDE1'")
        gx, gy, gz, ns, raw_sz, _ = struct.unpack("<6i", f.read(24))
        data = np.frombuffer(lz4.block.decompress(f.read(), raw_sz), dtype=np.float32)
    return data.reshape(ns, gz, gy, gx)


def iter_pde_series(directory: str, pattern: str = "pde_step_*.pde.lz4") -> Iterator[tuple[int, np.ndarray]]:
    """Yield (step_idx, arr) pairs in step-order from a directory of PDE snapshots."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    for p in paths:
        stem = os.path.basename(p).split(".")[0]
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            continue
        yield step, load_pde_lz4(p)


def add_category(df: pd.DataFrame, mode: str = "type_state") -> pd.DataFrame:
    """Add a 'category' column classifying each agent.

    mode='type'        → one label per agent type ('cancer', 'tcell', ...)
    mode='type_state'  → (type, state) combos ('cancer_stem', 'tcell_eff', 'mdsc', ...)
    """
    out = df.copy()
    if mode == "type":
        out["category"] = out["type"].map(TYPE_NAMES).fillna("unknown")
        return out
    if mode != "type_state":
        raise ValueError(f"Unknown mode {mode!r}; expected 'type' or 'type_state'")

    def _label(row):
        t = int(row["type"])
        s = int(row["state"])
        t_name = TYPE_NAMES.get(t, "unknown")
        s_name = _STATE_NAMES.get(t, {}).get(s)
        if t == TYPE_DC:
            sub = _DC_SUBTYPES.get(int(row["extra"]), f"sub{int(row['extra'])}")
            state_tag = s_name if s_name is not None else f"state{s}"
            return f"{t_name}_{sub}_{state_tag}"
        if s_name is None:
            return f"{t_name}_state{s}"
        return f"{t_name}_{s_name}" if s_name else t_name

    out["category"] = out.apply(_label, axis=1)
    return out


def category_list(mode: str = "type_state") -> list[str]:
    """Canonical category labels for a given mode, sorted for stable column order."""
    if mode == "type":
        return sorted(TYPE_NAMES.values())
    if mode == "type_state":
        labels = []
        for t, states in _STATE_NAMES.items():
            t_name = TYPE_NAMES[t]
            if t == TYPE_DC:
                for sub in _DC_SUBTYPES.values():
                    for s_name in states.values():
                        tag = s_name if s_name else "state0"
                        labels.append(f"{t_name}_{sub}_{tag}")
                continue
            for s_name in states.values():
                labels.append(f"{t_name}_{s_name}" if s_name else t_name)
        return sorted(labels)
    raise ValueError(f"Unknown mode {mode!r}")
