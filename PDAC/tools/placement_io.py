"""The placement → ABM handoff: write a placed configuration the simulator can load (§C6.2).

Until now the placement samplers (``seeding_potts``) produced ``(pos, cat)`` in memory and
nothing consumed them — there was no file format and no ABM-side loader, so "wire up seeding"
was really "build the handoff". This module is the writer half; ``PDAC/sim/initialization.cu``
(``initializeFromPlacement``) is the reader half, and ``PLC_VERSION`` is the contract between
them. Change one, change both.

What crosses the boundary
-------------------------
Two things, and deliberately only two:

1. **An agent table** — one row per placed cell, ``(type_id, x, y, z, cell_state, subtype)``.
   ``type_id`` reuses the ``abm_lz4_def.txt`` convention already used for ABM *output*, so a
   placement file and a simulation snapshot speak the same vocabulary and can be diffed
   directly. Per-agent internal state (division counters, lifespans, exhaustion, persistence
   directions) is deliberately NOT carried: the loader draws it from the same distributions the
   recruitment path uses, so a placed cell and a recruited cell are statistically
   indistinguishable in everything except position. That is workplan D-11, resolved by
   deferring to the existing code rather than inventing a second convention.

2. **Two field seeds** — ``V_vas(t=0)`` from ``seeding_vvas.seed_vvas`` and ``ECM(t=0)`` from
   ``seeding_ecm.seed_ecm``. D-4 dropped vascular agents but kept perfusion as a field, so
   ``vas_phalanx`` is placed, rasterized, smoothed and floored into ``V_vas`` and then **no
   vascular agent is emitted**; it is the one category that becomes a field instead of cells.
   ECM is different — it is built from placed ``fib_myCAF``, which ALSO become agents, because
   the ECM field is what those cells would have deposited had a presim run.

   Both fields are load-bearing and coupled: without the ECM seed there is no desmoplastic
   compression, ``V_vas`` grows unopposed toward its carrying capacity, and the O2
   heterogeneity the perfusion seed just created collapses within a few steps (measured: O2 cv
   0.28 → 0.057 over 8 steps at ECM ≈ 0). Seeding one without the other is worse than seeding
   neither, because it looks like it worked.

Format (plain, uncompressed, little-endian)
-------------------------------------------
The output snapshots are LZ4 because they are written every step and there are thousands of
them. A placement file is written once and read once, at a few MB, so it is left uncompressed:
the reader stays a dozen lines of ``ifstream`` with no compression dependency on the load path.

    magic     char[4]  'PLC1'
    version   i32      PLC_VERSION
    grid_x    i32
    grid_y    i32
    grid_z    i32
    n_agents  i32
    n_cols    i32      = 6
    has_vvas  i32      0 | 1
    has_ecm   i32      0 | 1
    agents    i32[n_agents][6]
    vvas      f32[grid_z][grid_y][grid_x]   (only if has_vvas)
    ecm       f32[grid_z][grid_y][grid_x]   (only if has_ecm)

⚠️ Both fields are written in **(z, y, x)** order to match the simulator's voxel index
``z*(nx*ny) + y*nx + x``. The sampling code works in ``(x, y, z)``, so the transpose here is
load-bearing — getting it wrong silently rotates the field rather than failing.

``PLC_VERSION`` went 1 → 2 when ECM was added. The reader rejects a mismatch outright: a v1
file loaded as v2 would be read as "no ECM" and would silently reproduce exactly the
compression-free failure described above.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PLC_MAGIC = b"PLC1"
PLC_VERSION = 2
PLC_N_COLS = 6

# ── agent type ids — MUST match core/common.cuh and outputs/abm_lz4_def.txt ─────────────────
T_CANCER, T_TCELL, T_TREG, T_MDSC, T_MAC, T_FIB, T_VAS, T_BCELL, T_DC = range(9)

# ── placement category → (type_id, cell_state, subtype) ─────────────────────────────────────
# States are the enums in core/common.cuh. Note FIB_APCAF = 4, not 3 — 3 is FIB_FRC, which the
# IMC panel cannot identify and which SEED_DROP therefore forbids inventing.
CATEGORY_TO_AGENT: dict[str, tuple[int, int, int]] = {
    "cancer_stem":     (T_CANCER, 0, 0),   # CANCER_STEM
    "cancer_prog":     (T_CANCER, 1, 0),   # CANCER_PROGENITOR
    "cancer_sen":      (T_CANCER, 2, 0),   # CANCER_SENESCENT
    "tcell_eff":       (T_TCELL,  0, 0),   # T_CELL_EFF
    "treg_th":         (T_TREG,   0, 0),   # TCD4_TH
    "treg_treg":       (T_TREG,   1, 0),   # TCD4_TREG
    "mdsc":            (T_MDSC,   0, 0),   # single state
    "mac_M1":          (T_MAC,    0, 0),   # MAC_M1
    "mac_M2":          (T_MAC,    1, 0),   # MAC_M2
    "fib_quiescent":   (T_FIB,    0, 0),   # FIB_QUIESCENT
    "fib_myCAF":       (T_FIB,    1, 0),   # FIB_MYCAF
    "fib_iCAF":        (T_FIB,    2, 0),   # FIB_ICAF
    "fib_apCAF":       (T_FIB,    4, 0),   # FIB_APCAF
    "bcell_naive":     (T_BCELL,  0, 0),   # BCELL_NAIVE
    "bcell_act":       (T_BCELL,  1, 0),   # BCELL_ACTIVATED
    "dc_immature_any": (T_DC,     0, 0),   # DC_IMMATURE, subtype filled below
    "dc_mature_any":   (T_DC,     1, 0),   # DC_MATURE,   subtype filled below
}

# vas_phalanx becomes the V_vas field, never an agent (D-4). Listed explicitly so that a
# category which is neither mapped nor listed here is an ERROR rather than a silent drop —
# the same discipline stage 1 applies to unclaimed HALO columns.
CATEGORY_AS_FIELD = {"vas_phalanx"}


def split_dc_subtypes(n: int, cdc1_frac: float, rng: np.random.Generator) -> np.ndarray:
    """cDC1/cDC2 assignment for placed DCs.

    The IMC panel cannot distinguish cDC1 from cDC2 (which is why both DC categories are
    ``*_any``), but the ABM's DC agent requires a subtype. This is the one place the writer
    must supply something the data does not contain, so it is an explicit, named argument
    rather than a buried default — and it is a *within-category* split, not a new category, so
    SEED_DROP's "never invent a state the data did not measure" is not violated: the DCs
    themselves were measured.
    """
    n_cdc1 = int(round(n * cdc1_frac))
    sub = np.ones(n, dtype=np.int32)          # DC_CDC2 = 1
    sub[:n_cdc1] = 0                          # DC_CDC1 = 0
    rng.shuffle(sub)
    return sub


def build_agent_table(pos: np.ndarray, cat: np.ndarray, shape: tuple[int, int, int],
                      cdc1_frac: float = 0.5, seed: int = 0,
                      substate_frac: dict[str, dict[int, float]] | None = None,
                      ) -> tuple[np.ndarray, dict]:
    """``(pos, cat)`` → the ``(N, 6)`` int32 agent table, dropping field-only categories.

    ``substate_frac`` optionally overrides the single ``CATEGORY_TO_AGENT`` state for a
    category with a distribution over states, e.g.
    ``{"tcell_eff": {0: 0.76, 2: 0.24}}`` to seed 24% of placed CD8 as exhausted. Use it for
    sub-states the IMC panel cannot call but the QSP tracks separately — the fractions come
    from the QSP onset dump, so nothing is invented. Assignment is at random WITHIN the
    category, which deliberately leaves the sub-state spatially unstructured: the placement
    targets are per-category FunCN, and the data carries no exhausted-vs-active geometry to
    match. If that assumption is ever tested and fails, this is the line to change.
    """
    pos = np.asarray(pos)
    cat = np.asarray(cat)
    gx, gy, gz = shape

    unknown = sorted(set(cat.tolist()) - set(CATEGORY_TO_AGENT) - CATEGORY_AS_FIELD)
    if unknown:
        raise ValueError(
            f"categories with no agent mapping and not declared field-only: {unknown}. "
            "Add them to CATEGORY_TO_AGENT or CATEGORY_AS_FIELD — silently dropping a placed "
            "category is the failure mode this check exists to prevent."
        )

    keep = ~np.isin(cat, list(CATEGORY_AS_FIELD))
    p, c = pos[keep].astype(np.int32), cat[keep]
    if len(p) == 0:
        return np.zeros((0, PLC_N_COLS), dtype=np.int32), {"n_agents": 0}

    np.clip(p[:, 0], 0, gx - 1, out=p[:, 0])
    np.clip(p[:, 1], 0, gy - 1, out=p[:, 1])
    np.clip(p[:, 2], 0, gz - 1, out=p[:, 2])

    table = np.zeros((len(p), PLC_N_COLS), dtype=np.int32)
    table[:, 1:4] = p
    for name, (tid, state, sub) in CATEGORY_TO_AGENT.items():
        m = c == name
        if not m.any():
            continue
        table[m, 0] = tid
        table[m, 4] = state
        table[m, 5] = sub

    dc = table[:, 0] == T_DC
    if dc.any():
        table[dc, 5] = split_dc_subtypes(int(dc.sum()), cdc1_frac,
                                         np.random.default_rng(seed))

    substates: dict[str, dict[int, int]] = {}
    if substate_frac:
        rng = np.random.default_rng(seed + 1)
        for name, frac in substate_frac.items():
            m = c == name
            n = int(m.sum())
            if n == 0 or not frac:
                continue
            states = np.array(sorted(frac), dtype=np.int32)
            w = np.array([frac[int(s)] for s in states], dtype=np.float64)
            if w.sum() <= 0:
                continue
            # Largest-remainder allocation, so a rare sub-state is not rounded away and the
            # assigned counts sum to n exactly.
            exact = n * w / w.sum()
            k = np.floor(exact).astype(np.int64)
            k[np.argsort(-(exact - k))[:int(n - k.sum())]] += 1
            assign = np.repeat(states, k)
            rng.shuffle(assign)
            table[m, 4] = assign
            substates[name] = {int(s): int(v) for s, v in zip(states, k)}

    counts = {n: int((c == n).sum()) for n in sorted(set(c.tolist()))}
    return table, {"n_agents": len(table), "per_category": counts,
                   "n_field_only": int((~keep).sum()), "substates": substates}


def _to_zyx(field: np.ndarray, shape: tuple[int, int, int], name: str) -> np.ndarray:
    """(x,y,z) → (z,y,x), the simulator's ``z*(nx*ny) + y*nx + x`` layout."""
    gx, gy, gz = shape
    field = np.asarray(field, dtype=np.float32)
    if field.shape != (gx, gy, gz):
        raise ValueError(f"{name} must be (gx, gy, gz) = {(gx, gy, gz)}, got {field.shape}")
    return np.ascontiguousarray(np.transpose(field, (2, 1, 0)), dtype=np.float32)


def write_placement(path: str | Path, table: np.ndarray, shape: tuple[int, int, int],
                    vvas: np.ndarray | None = None,
                    ecm: np.ndarray | None = None) -> dict:
    """Write the ``.plc`` file. Fields are indexed ``(x, y, z)`` and transposed on the way out."""
    gx, gy, gz = shape
    table = np.ascontiguousarray(table, dtype=np.int32)
    if table.ndim != 2 or table.shape[1] != PLC_N_COLS:
        raise ValueError(f"agent table must be (N, {PLC_N_COLS}), got {table.shape}")

    has_vvas, has_ecm = vvas is not None, ecm is not None
    vvas_out = _to_zyx(vvas, shape, "vvas") if has_vvas else None
    ecm_out = _to_zyx(ecm, shape, "ecm") if has_ecm else None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(PLC_MAGIC)
        f.write(struct.pack("<8i", PLC_VERSION, gx, gy, gz,
                            len(table), PLC_N_COLS, int(has_vvas), int(has_ecm)))
        f.write(table.tobytes())
        if has_vvas:
            f.write(vvas_out.tobytes())
        if has_ecm:
            f.write(ecm_out.tobytes())

    return {"path": str(path), "bytes": path.stat().st_size, "n_agents": len(table),
            "has_vvas": bool(has_vvas), "has_ecm": bool(has_ecm)}


def read_placement(path: str | Path):
    """Read back a ``.plc`` → ``(table, vvas, ecm, meta)``.

    For round-trip tests and for inspecting exactly what the ABM will load.
    """
    with open(path, "rb") as f:
        if f.read(4) != PLC_MAGIC:
            raise ValueError(f"{path}: not a PLC file")
        ver, gx, gy, gz, n, ncols, has_vvas, has_ecm = struct.unpack("<8i", f.read(32))
        if ver != PLC_VERSION:
            raise ValueError(
                f"{path}: version {ver}, expected {PLC_VERSION} — regenerate it. A v1 file has "
                "no ECM field, and loading it as though it did would silently disable "
                "desmoplastic compression."
            )
        table = np.frombuffer(f.read(n * ncols * 4), dtype=np.int32).reshape(n, ncols)

        def _read_field():
            raw = np.frombuffer(f.read(gx * gy * gz * 4), dtype=np.float32)
            return np.transpose(raw.reshape(gz, gy, gx), (2, 1, 0))

        vvas = _read_field() if has_vvas else None
        ecm = _read_field() if has_ecm else None
    return table, vvas, ecm, {"version": ver, "shape": (gx, gy, gz), "n_agents": n}
