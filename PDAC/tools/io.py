"""ABM snapshot loaders + category helpers.

Schema of .abm.lz4 files is documented in outputs/abm_lz4_def.txt:
  int32 columns = [type_id, agent_id, x, y, z, cell_state, life, extra]

type_id: 0=CANCER 1=TCELL 2=TREG 3=MDSC 4=MAC 5=FIB 6=VAS
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

TYPE_CANCER, TYPE_TCELL, TYPE_TREG, TYPE_MDSC, TYPE_MAC, TYPE_FIB, TYPE_VAS = 0, 1, 2, 3, 4, 5, 6

TYPE_NAMES = {
    TYPE_CANCER: "cancer",
    TYPE_TCELL:  "tcell",
    TYPE_TREG:   "treg",
    TYPE_MDSC:   "mdsc",
    TYPE_MAC:    "mac",
    TYPE_FIB:    "fib",
    TYPE_VAS:    "vas",
}

# (type, state) → category label. Keep lowercase snake-case for consistency.
_STATE_NAMES = {
    TYPE_CANCER: {0: "stem", 1: "prog", 2: "sen"},
    TYPE_TCELL:  {0: "eff",  1: "cyt",  2: "sup"},
    TYPE_TREG:   {0: "th",   1: "treg"},
    TYPE_MDSC:   {0: ""},  # single state
    TYPE_MAC:    {0: "M1",  1: "M2"},
    TYPE_FIB:    {0: "normal", 1: "caf"},
    TYPE_VAS:    {0: "tip", 2: "phalanx"},  # note: state==1 is unused
}


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
        t_name = TYPE_NAMES.get(int(row["type"]), "unknown")
        s_name = _STATE_NAMES.get(int(row["type"]), {}).get(int(row["state"]))
        if s_name is None:
            return f"{t_name}_state{int(row['state'])}"
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
            for s_name in states.values():
                labels.append(f"{t_name}_{s_name}" if s_name else t_name)
        return sorted(labels)
    raise ValueError(f"Unknown mode {mode!r}")
