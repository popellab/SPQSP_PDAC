#!/usr/bin/env python3
"""Auto-layout for the QSP schematic.

Five layout engines:
  A) dot        — graphviz hierarchical layout with zone clusters
  B) topo       — topological layering + barycenter ordering (pure Python)
  C) force      — force-directed with zone constraints (scipy)
  D) grandalf   — Sugiyama hierarchical layout (grandalf library)
  E) osage      — graphviz recursive cluster layout

Usage:
    python layout_schematic.py all                    # compare all engines
    python layout_schematic.py all --render           # compile PDFs for each
    python layout_schematic.py <engine>               # print positions
    python layout_schematic.py <engine> --apply       # patch qsp_schematic.tex
    python layout_schematic.py <engine> --render      # compile a PDF
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

COMPARTMENTS = {
    "peripheral": (23, 2),
    "central":    (5, 9),
    "dln":        (5, 15),
    "tumor":      (13, 15),
    "injection":  (10, 3),
}

COMP_TEX_OFFSETS = {
    "peripheral": ("PERICOLSTART", "PERIROWSTART"),
    "central":    ("CENTCOLSTART", "CENTROWSTART"),
    "dln":        ("DLNCOLSTART", "DLNROWSTART"),
    "tumor":      ("TUMCOLSTART", "TUMROWSTART"),
    "injection":  ("IDCOLSTART", "IDROWSTART"),
}

PREFIX_MAP = {"p": "peripheral", "c": "central", "d": "dln", "t": "tumor", "id": "injection"}
COMP_TO_PREFIX = {v: k for k, v in PREFIX_MAP.items()}

# Zone bounding boxes within the tumor compartment: (col_min, row_min, col_max, row_max)
# These constrain where nodes in each zone can be placed.
TUMOR_ZONE_BOUNDS = {
    "immune_top": (0,  0,  6,  5),
    "myeloid":    (0,  2,  6,  6),
    "cancer":     (0,  6, 10,  9),
    "tcell":      (6,  0, 12,  8),
    "stroma":     (6,  9, 12, 14),
    "dc":         (0,  9,  8, 14),
}


@dataclass
class Node:
    name: str
    label: str
    compartment: str
    col: int
    row: int
    style: str = "pop"
    fixed: bool = False
    zone: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    style: str = "act"


def parse_ref(ref: str):
    prefix, name = ref.split(".", 1)
    return PREFIX_MAP[prefix], name


def full_id(comp: str, name: str) -> str:
    return f"{COMP_TO_PREFIX[comp]}.{name}"


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

PERIPHERAL_NODES = [
    ("nCD4",  "nTCD4", 6, 0, "pop", False, ""),
    ("nCD8",  "nTCD8", 5, 1, "pop", False, ""),
    ("Teff",  "Teff",  7, 1, "pop", False, ""),
    ("Th",    "Th",    8, 0, "pop", False, ""),
    ("Treg",  "Treg",  9, 1, "pop", False, ""),
]

CENTRAL_NODES = [
    ("nCD8",  "nTCD8", 0, 0, "pop", False, ""),
    ("nCD4",  "nTCD4", 1, 1, "pop", False, ""),
    ("Th",    "Th",    3, 2, "pop", False, ""),
    ("Treg",  "Treg",  3, 3, "pop", False, ""),
    ("Teff",  "Teff",  2, 5, "pop", False, ""),
    ("Cy",    "Cy",    4, 0, "mol", True, ""),
]

DLN_NODES = [
    ("nCD8",  "nTCD8",  0, 5, "pop", False, "tcell"),
    ("aCD8",  "aTCD8",  2, 5, "pop", False, "tcell"),
    ("Teff",  "Teff",   4, 5, "pop", False, "tcell"),
    ("nCD4",  "nTCD4",  0, 3, "pop", False, "tcell"),
    ("aCD4",  "aTCD4",  2, 3, "pop", False, "tcell"),
    ("IL2",   "IL-2",   3, 4, "mol", False, "tcell"),
    ("Treg",  "Treg",   4, 3, "pop", False, "tcell"),
    ("Th",    "Th",     4, 2, "pop", False, "tcell"),
    ("mcDC1", "mcDC1",  1, 14, "pop", False, "dc"),
    ("mcDC2", "mcDC2",  3, 14, "pop", False, "dc"),
]

TUMOR_NODES = [
    # Zone: immune_top
    ("PDL1",      "PD-L1",       0,  1, "mol", False, "immune_top"),
    ("ifng",      "IFNγ",        2,  1, "mol", False, "immune_top"),
    ("il10_m2",   "IL-10",       4,  1, "mol", False, "immune_top"),
    ("tgfb_m2",   "TGF-β",       5,  1, "mol", False, "immune_top"),
    ("Th_entry",  "Th",          1,  2, "pop", True,  "immune_top"),
    ("Treg_entry","Treg",        1,  3, "pop", True,  "immune_top"),
    ("Teff_entry","Teff",        1,  5, "pop", True,  "immune_top"),
    # Zone: myeloid
    ("m2",        "M2 Mac",      4,  2, "pop", False, "myeloid"),
    ("mdsc",      "MDSC",        5,  3, "pop", False, "myeloid"),
    ("IL12_m1",   "IL-12",       0,  4, "mol", False, "myeloid"),
    ("m1",        "M1 Mac",      2,  4, "pop", False, "myeloid"),
    ("ccl2",      "CCL2",        4,  5, "mol", False, "myeloid"),
    # Zone: cancer
    ("cancer",    "Cancer",      2,  7, "pop", False, "cancer"),
    ("debris",    "Debris",      2,  9, "mol", False, "cancer"),
    ("ccl5",      "CCL5",        4,  6, "mol", False, "cancer"),
    # Zone: tcell (right side)
    ("no",        "NO",          7,  3, "mol", False, "tcell"),
    ("argi",      "ArgI",        7,  4, "mol", False, "tcell"),
    ("tgfb_treg", "TGF-β",       8,  5, "mol", False, "tcell"),
    ("Th",        "Th",         10,  2, "pop", False, "tcell"),
    ("Treg",      "Treg",       10,  4, "pop", False, "tcell"),
    ("Teff",      "Teff",        7,  7, "pop", False, "tcell"),
    ("Texh",      "Texh",       12,  7, "pop", False, "tcell"),
    # Zone: stroma
    ("pdgf",      "PDGF",        6, 10, "mol", False, "stroma"),
    ("ecm",       "ECM",         9, 11, "mol", False, "stroma"),
    ("tgfb_fib",  "TGF-β",      10, 11, "mol", False, "stroma"),
    ("cxcl12",    "CXCL12",     12, 11, "mol", False, "stroma"),
    ("il6",       "IL-6",        8, 12, "mol", False, "stroma"),
    ("mycaf",     "myCAF",       9, 13, "pop", False, "stroma"),
    ("icaf",      "iCAF",       10, 13, "pop", False, "stroma"),
    ("qpsc",      "qPSC",       12, 13, "pop", False, "stroma"),
    ("apcaf",     "apCAF",      11, 14, "pop", False, "stroma"),
    # Zone: dc
    ("il12_mapc", "IL-12",       4, 12, "mol", False, "dc"),
    ("il10_mapc", "IL-10",       5, 12, "mol", False, "dc"),
    ("cdc1",      "cDC1",        6, 13, "pop", False, "dc"),
    ("cdc2",      "cDC2",        8, 13, "pop", False, "dc"),
    ("mcdc1",     "mcDC1",       2, 14, "pop", False, "dc"),
    ("mcdc2",     "mcDC2",       4, 14, "pop", False, "dc"),
]

INJECTION_NODES = [
    ("GVAX",  "GVAX",   2, 2, "pop", True,  ""),
    ("GMCSF", "GM-CSF", 5, 2, "mol", False, ""),
    ("P1",    "Debris", 2, 1, "mol", False, ""),
    ("APC",   "APC",    6, 0, "pop", False, ""),
    ("mAPC",  "mAPC",   2, 0, "pop", False, ""),
]

EDGE_DEFS = [
    ("p.nCD8",  "c.nCD8",  "bidir"),
    ("p.nCD4",  "c.nCD4",  "bidir"),
    ("p.Th",    "c.Th",    "bidir"),
    ("p.Treg",  "c.Treg",  "bidir"),
    ("p.Teff",  "c.Teff",  "bidir"),
    ("p.nCD4",  "d.nCD4",  "bidir"),
    ("p.nCD8",  "d.nCD8",  "bidir"),
    ("d.nCD8",  "d.aCD8",  "act"),
    ("d.aCD8",  "d.Teff",  "act"),
    ("d.nCD4",  "d.aCD4",  "act"),
    ("d.aCD4",  "d.Treg",  "act"),
    ("d.aCD4",  "d.IL2",   "act"),
    ("d.mcDC1", "d.nCD8",  "branch"),
    ("d.mcDC2", "d.nCD4",  "branch"),
    ("d.Teff",  "c.Teff",  "act"),
    ("d.Treg",  "c.Treg",  "act"),
    ("d.Th",    "c.Th",    "act"),
    ("c.Th",    "t.Th_entry",   "act"),
    ("c.Treg",  "t.Treg_entry", "act"),
    ("c.Teff",  "t.Teff_entry", "act"),
    ("t.ifng",      "t.PDL1",     "act"),
    ("t.m2",        "t.il10_m2",  "act"),
    ("t.m2",        "t.tgfb_m2",  "act"),
    ("t.Th_entry",  "t.ifng",     "act"),
    ("t.ccl2",  "t.m1",      "act"),
    ("t.m1",    "t.IL12_m1", "act"),
    ("t.m1",    "t.m2",      "bidir"),
    ("t.m1",    "t.cancer",  "inhib"),
    ("t.ccl2",  "t.mdsc",    "act"),
    ("t.mdsc",  "t.no",      "act"),
    ("t.mdsc",  "t.argi",    "act"),
    ("t.no",    "t.argi",    "line"),
    ("t.argi",  "t.tgfb_treg", "line"),
    ("t.cancer", "t.ccl2",   "act"),
    ("t.cancer", "t.debris", "act"),
    ("t.cancer", "t.ccl5",   "act"),
    ("t.cancer", "t.pdgf",   "act"),
    ("t.Teff",   "t.cancer", "inhib"),
    ("t.Th",     "t.Treg",      "act"),
    ("t.Treg",   "t.tgfb_treg", "act"),
    ("t.Th",     "t.Texh",      "act"),
    ("t.Teff",   "t.Texh",      "act"),
    ("t.ccl5",   "t.Treg_entry","branch"),
    ("t.Treg",   "t.Teff",      "inhib"),
    ("t.tgfb_treg", "t.Teff",   "inhib"),
    ("t.qpsc",     "t.mycaf",      "act"),
    ("t.qpsc",     "t.icaf",       "act"),
    ("t.mycaf",    "t.ecm",        "act"),
    ("t.mycaf",    "t.cxcl12",     "act"),
    ("t.icaf",     "t.il6",        "act"),
    ("t.icaf",     "t.mycaf",      "bidir"),
    ("t.tgfb_fib", "t.qpsc",      "branch"),
    ("t.pdgf",     "t.qpsc",      "branch"),
    ("t.cxcl12",   "t.Teff_entry", "inhib"),
    ("t.ecm",      "t.Teff",       "inhib"),
    ("t.ecm",      "t.Texh",       "inhib"),
    ("t.cdc1",      "t.mcdc1",   "act"),
    ("t.cdc2",      "t.mcdc2",   "act"),
    ("t.debris",    "t.mcdc1",   "act"),
    ("t.il12_mapc", "t.cdc1",    "branch"),
    ("t.il10_mapc", "t.cdc2",    "inhib"),
    ("t.mcdc1",     "d.mcDC1",   "act"),
    ("t.mcdc2",     "d.mcDC2",   "act"),
    ("c.Cy", "t.Treg", "inhib"),
    ("id.GVAX",  "id.GMCSF", "act"),
    ("id.GVAX",  "id.P1",    "act"),
    ("id.APC",   "id.mAPC",  "act"),
    ("id.GMCSF", "id.APC",   "branch"),
    ("id.P1",    "id.APC",   "branch"),
    ("id.mAPC",  "d.mcDC1",  "act"),
]


def build_nodes() -> dict[str, list[Node]]:
    raw = {
        "peripheral": PERIPHERAL_NODES,
        "central":    CENTRAL_NODES,
        "dln":        DLN_NODES,
        "tumor":      TUMOR_NODES,
        "injection":  INJECTION_NODES,
    }
    return {
        comp: [Node(n, l, comp, c, r, s, f, z) for n, l, c, r, s, f, z in items]
        for comp, items in raw.items()
    }


def get_internal_edges(comp: str) -> list[Edge]:
    edges = []
    for src, dst, style in EDGE_DEFS:
        sc, _ = parse_ref(src)
        dc, _ = parse_ref(dst)
        if sc == comp and dc == comp:
            edges.append(Edge(src, dst, style))
    return edges


# ---------------------------------------------------------------------------
# Engine A: Graphviz dot with zone clusters
# ---------------------------------------------------------------------------

def layout_dot(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    ncols, nrows = COMPARTMENTS[comp]
    lines = [
        "digraph {",
        "  graph [rankdir=TB, nodesep=0.8, ranksep=0.6, newrank=true];",
        "  node [shape=box, width=0.5, height=0.3];",
    ]
    zones = defaultdict(list)
    for n in nodes:
        zones[n.zone or "default"].append(n)

    for zone_name, zone_nodes in zones.items():
        if zone_name != "default" and len(zone_nodes) > 1:
            lines.append(f"  subgraph cluster_{zone_name} {{")
            lines.append(f'    graph [style=invis];')
            for n in zone_nodes:
                lines.append(f'    "{n.name}" [label="{n.label}"];')
            lines.append("  }")
        else:
            for n in zone_nodes:
                lines.append(f'  "{n.name}" [label="{n.label}"];')

    for e in edges:
        _, src_name = parse_ref(e.src)
        _, dst_name = parse_ref(e.dst)
        attrs = []
        if e.style == "bidir":
            attrs += ["dir=both", "weight=2"]
        elif e.style == "inhib":
            attrs += ["arrowhead=tee", "weight=1", "constraint=false"]
        elif e.style == "branch":
            attrs += ["arrowhead=odot", "style=dashed", "constraint=false"]
        elif e.style == "line":
            attrs += ["dir=none", "weight=1"]
        else:
            attrs.append("weight=3")
        attr_str = f' [{", ".join(attrs)}]' if attrs else ""
        lines.append(f'  "{src_name}" -> "{dst_name}"{attr_str};')
    lines.append("}")

    result = subprocess.run(
        ["dot", "-Tplain"], input="\n".join(lines),
        capture_output=True, text=True, check=True,
    )
    raw_pos = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if parts[0] == "node":
            raw_pos[parts[1].strip('"')] = (float(parts[2]), float(parts[3]))
    if not raw_pos:
        return {n.name: (n.col, n.row) for n in nodes}
    return _normalize_positions(nodes, raw_pos, ncols, nrows, flip_y=True)


# ---------------------------------------------------------------------------
# Engine B: Topological layering + barycenter ordering
# ---------------------------------------------------------------------------

def layout_topo(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    ncols, nrows = COMPARTMENTS[comp]
    node_names = {n.name for n in nodes}

    fwd = defaultdict(list)
    rev = defaultdict(list)
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_names and dst in node_names:
            if e.style in ("act", "line"):
                fwd[src].append(dst)
                rev[dst].append(src)
            elif e.style == "bidir":
                fwd[src].append(dst)
                fwd[dst].append(src)
                rev[src].append(dst)
                rev[dst].append(src)

    layers = {n.name: 0 for n in nodes}
    changed = True
    iters = 0
    while changed and iters < 100:
        changed = False
        iters += 1
        for name in node_names:
            for dst in fwd.get(name, []):
                if layers[dst] < layers[name] + 1:
                    layers[dst] = layers[name] + 1
                    changed = True

    max_layer = max(layers.values()) if layers else 0
    layer_groups = defaultdict(list)
    for name, layer in layers.items():
        layer_groups[layer].append(name)

    x_pos = {n.name: float(n.col) for n in nodes}
    neighbors = defaultdict(list)
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_names and dst in node_names:
            neighbors[src].append(dst)
            neighbors[dst].append(src)

    for _ in range(20):
        for layer in range(max_layer + 1):
            for name in layer_groups[layer]:
                nbrs = [n for n in neighbors[name] if n in x_pos]
                if nbrs:
                    x_pos[name] = sum(x_pos[n] for n in nbrs) / len(nbrs)

    row_scale = (nrows - 1) / max_layer if max_layer > 0 else 1
    xs = list(x_pos.values())
    x_min, x_max = min(xs), max(xs)
    x_range = x_max - x_min if x_max > x_min else 1

    grid = {}
    for n in nodes:
        if n.fixed:
            grid[n.name] = (n.col, n.row)
            continue
        row = round(layers[n.name] * row_scale)
        col = round((x_pos[n.name] - x_min) / x_range * (ncols - 1))
        grid[n.name] = (max(0, min(ncols - 1, col)), max(0, min(nrows - 1, row)))
    return _resolve_collisions(grid, nodes, ncols, nrows)


# ---------------------------------------------------------------------------
# Engine C: Force-directed with ZONE constraints (scipy)
# ---------------------------------------------------------------------------

def layout_force(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    import numpy as np
    from scipy.optimize import minimize as sp_minimize

    ncols, nrows = COMPARTMENTS[comp]
    N = len(nodes)
    if N == 0:
        return {}

    name_to_idx = {n.name: i for i, n in enumerate(nodes)}

    # Zone bounds for each node (normalized to [0,1])
    zone_bounds_norm = {}
    for i, n in enumerate(nodes):
        if comp == "tumor" and n.zone in TUMOR_ZONE_BOUNDS:
            cmin, rmin, cmax, rmax = TUMOR_ZONE_BOUNDS[n.zone]
            zone_bounds_norm[i] = (
                cmin / max(ncols - 1, 1), rmin / max(nrows - 1, 1),
                cmax / max(ncols - 1, 1), rmax / max(nrows - 1, 1),
            )
        else:
            zone_bounds_norm[i] = (0, 0, 1, 1)

    # Initial positions
    x0 = np.zeros(2 * N)
    for i, n in enumerate(nodes):
        x0[2 * i] = n.col / max(ncols - 1, 1)
        x0[2 * i + 1] = n.row / max(nrows - 1, 1)

    edge_pairs = []
    node_set = {n.name for n in nodes}
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_set and dst in node_set:
            edge_pairs.append((name_to_idx[src], name_to_idx[dst], e.style))

    fixed_mask = np.array([n.fixed for n in nodes])

    def energy(x):
        pos = x.reshape(N, 2)
        E = 0.0
        # Repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                E += 2.0 / (d @ d + 1e-4)
        # Springs
        for i, j, style in edge_pairs:
            d = pos[i] - pos[j]
            E += 3.0 * (d @ d)
        # Downward flow
        for i, j, style in edge_pairs:
            if style == "act":
                dy = pos[j, 1] - pos[i, 1]
                if dy < 0:
                    E += 5.0 * dy * dy
        # Orthogonality: penalize diagonal edges (prefer vertical/horizontal)
        # |dx|*|dy| is zero for perfectly axis-aligned edges
        for i, j, style in edge_pairs:
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            E += 4.0 * dx * dy  # diagonal penalty
        # Zone constraints (soft)
        for i in range(N):
            if fixed_mask[i]:
                continue
            xmin, ymin, xmax, ymax = zone_bounds_norm[i]
            px, py = pos[i]
            if px < xmin: E += 200.0 * (px - xmin) ** 2
            if px > xmax: E += 200.0 * (px - xmax) ** 2
            if py < ymin: E += 200.0 * (py - ymin) ** 2
            if py > ymax: E += 200.0 * (py - ymax) ** 2
        return E

    # Build per-variable bounds from zone constraints
    bounds = []
    for i, n in enumerate(nodes):
        xmin, ymin, xmax, ymax = zone_bounds_norm[i]
        if n.fixed:
            bounds.append((x0[2*i], x0[2*i]))
            bounds.append((x0[2*i+1], x0[2*i+1]))
        else:
            bounds.append((xmin, xmax))
            bounds.append((ymin, ymax))

    result = sp_minimize(
        energy, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-12},
    )

    pos = result.x.reshape(N, 2)
    grid = {}
    for i, n in enumerate(nodes):
        if n.fixed:
            grid[n.name] = (n.col, n.row)
        else:
            col = round(pos[i, 0] * (ncols - 1))
            row = round(pos[i, 1] * (nrows - 1))
            grid[n.name] = (max(0, min(ncols-1, col)), max(0, min(nrows-1, row)))
    return _resolve_collisions(grid, nodes, ncols, nrows)


# ---------------------------------------------------------------------------
# Engine D: Grandalf Sugiyama layout
# ---------------------------------------------------------------------------

def layout_grandalf(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    """Grandalf Sugiyama: runs on the whole compartment (no zone constraints)."""
    return _grandalf_layout_impl(comp, nodes, edges, use_zones=False)


def layout_grandalf_zoned(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    """Grandalf Sugiyama per-zone: runs separately within each zone bounding box."""
    return _grandalf_layout_impl(comp, nodes, edges, use_zones=True)


def _grandalf_layout_impl(comp, nodes, edges, use_zones=False):
    from grandalf.graphs import Vertex, Edge as GEdge, Graph
    from grandalf.layouts import SugiyamaLayout

    ncols, nrows = COMPARTMENTS[comp]
    node_set = {n.name for n in nodes}
    node_map = {n.name: n for n in nodes}

    if not use_zones or comp != "tumor":
        # Original: run on entire compartment
        return _grandalf_run(nodes, edges, node_set, ncols, nrows)

    # Per-zone layout: group nodes by zone, run Sugiyama within each zone's bbox
    zones = defaultdict(list)
    for n in nodes:
        zones[n.zone or "default"].append(n)

    grid = {}
    for zone_name, zone_nodes in zones.items():
        if zone_name in TUMOR_ZONE_BOUNDS:
            cmin, rmin, cmax, rmax = TUMOR_ZONE_BOUNDS[zone_name]
            zone_ncols = cmax - cmin + 1
            zone_nrows = rmax - rmin + 1
        else:
            cmin, rmin = 0, 0
            zone_ncols, zone_nrows = ncols, nrows

        zone_node_set = {n.name for n in zone_nodes}
        # Get edges internal to this zone
        zone_edges = []
        for e in edges:
            _, src = parse_ref(e.src)
            _, dst = parse_ref(e.dst)
            if src in zone_node_set and dst in zone_node_set:
                zone_edges.append(e)

        zone_pos = _grandalf_run(zone_nodes, zone_edges, zone_node_set, zone_ncols, zone_nrows)

        # Offset to zone's position within compartment
        for name, (col, row) in zone_pos.items():
            grid[name] = (col + cmin, row + rmin)

    return _resolve_collisions(grid, nodes, ncols, nrows)


def _grandalf_run(nodes, edges, node_set, ncols, nrows):
    from grandalf.graphs import Vertex, Edge as GEdge, Graph
    from grandalf.layouts import SugiyamaLayout

    verts = {}
    for n in nodes:
        v = Vertex(n.name)
        v.view = type("View", (), {"w": 40, "h": 20})()
        verts[n.name] = v

    gedges = []
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_set and dst in node_set:
            gedges.append(GEdge(verts[src], verts[dst]))

    g = Graph(list(verts.values()), gedges)

    raw_pos = {}
    for component in g.C:
        sug = SugiyamaLayout(component)
        sug.init_all()
        sug.draw(3)
        for v in component.sV:
            if hasattr(v, 'view') and v.view is not None:
                raw_pos[v.data] = (v.view.xy[0], v.view.xy[1])

    if not raw_pos:
        return {n.name: (n.col, n.row) for n in nodes}

    return _normalize_positions(nodes, raw_pos, ncols, nrows, flip_y=False)


# ---------------------------------------------------------------------------
# Engine E: Graphviz osage (recursive cluster layout)
# ---------------------------------------------------------------------------

def layout_osage(comp: str, nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[int, int]]:
    ncols, nrows = COMPARTMENTS[comp]
    lines = [
        "digraph {",
        '  graph [pack=true, packmode="array_tu"];',
        "  node [shape=box, width=0.5, height=0.3];",
    ]

    # Group nodes by zone
    zones = defaultdict(list)
    for n in nodes:
        zones[n.zone or "default"].append(n)

    # Define zone ordering for osage (top to bottom, left to right)
    zone_order = ["immune_top", "myeloid", "tcell", "cancer", "dc", "stroma", "default"]

    for zone_name in zone_order:
        if zone_name not in zones:
            continue
        zone_nodes = zones[zone_name]
        if zone_name != "default":
            lines.append(f"  subgraph cluster_{zone_name} {{")
            lines.append(f'    graph [label="{zone_name}", style=dashed, color=gray60];')
            for n in zone_nodes:
                lines.append(f'    "{n.name}" [label="{n.label}"];')
            lines.append("  }")
        else:
            for n in zone_nodes:
                lines.append(f'  "{n.name}" [label="{n.label}"];')

    for e in edges:
        _, src_name = parse_ref(e.src)
        _, dst_name = parse_ref(e.dst)
        lines.append(f'  "{src_name}" -> "{dst_name}";')
    lines.append("}")

    result = subprocess.run(
        ["osage", "-Tplain"], input="\n".join(lines),
        capture_output=True, text=True, check=True,
    )
    raw_pos = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if parts[0] == "node":
            raw_pos[parts[1].strip('"')] = (float(parts[2]), float(parts[3]))
    if not raw_pos:
        return {n.name: (n.col, n.row) for n in nodes}
    return _normalize_positions(nodes, raw_pos, ncols, nrows, flip_y=True)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _normalize_positions(nodes, raw_pos, ncols, nrows, flip_y=True):
    xs = [v[0] for v in raw_pos.values()]
    ys = [v[1] for v in raw_pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1

    grid = {}
    for n in nodes:
        if n.fixed:
            grid[n.name] = (n.col, n.row)
            continue
        if n.name not in raw_pos:
            grid[n.name] = (n.col, n.row)
            continue
        x, y = raw_pos[n.name]
        col = round((x - x_min) / x_range * (ncols - 1))
        if flip_y:
            row = round((1 - (y - y_min) / y_range) * (nrows - 1))
        else:
            row = round((y - y_min) / y_range * (nrows - 1))
        grid[n.name] = (max(0, min(ncols-1, col)), max(0, min(nrows-1, row)))
    return _resolve_collisions(grid, nodes, ncols, nrows)


def _resolve_collisions(grid, nodes, ncols, nrows):
    occupied = {}
    for n in nodes:
        if n.fixed:
            occupied[grid[n.name]] = n.name
    for n in nodes:
        if n.fixed:
            continue
        pos = grid[n.name]
        if pos not in occupied:
            occupied[pos] = n.name
            continue
        found = False
        for radius in range(1, max(ncols, nrows)):
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if abs(dc) != radius and abs(dr) != radius:
                        continue
                    candidate = (pos[0] + dc, pos[1] + dr)
                    if (0 <= candidate[0] < ncols and 0 <= candidate[1] < nrows
                            and candidate not in occupied):
                        occupied[candidate] = n.name
                        grid[n.name] = candidate
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return grid


def _count_crossings(comp, positions, edges):
    node_set = set(positions.keys())
    segments = []
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_set and dst in node_set:
            segments.append((positions[src], positions[dst]))
    crossings = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if _segments_cross(segments[i], segments[j]):
                crossings += 1
    return crossings


def _segments_cross(seg1, seg2):
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    if (x1, y1) in ((x3, y3), (x4, y4)) or (x2, y2) in ((x3, y3), (x4, y4)):
        return False
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))


def _total_edge_length(comp, positions, edges):
    node_set = set(positions.keys())
    total = 0
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_set and dst in node_set:
            p1, p2 = positions[src], positions[dst]
            total += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    return total


def _orthogonality_score(comp, positions, edges):
    """Fraction of edges that are nearly vertical or horizontal (within 15 deg)."""
    node_set = set(positions.keys())
    total = 0
    ortho = 0
    for e in edges:
        _, src = parse_ref(e.src)
        _, dst = parse_ref(e.dst)
        if src in node_set and dst in node_set:
            p1, p2 = positions[src], positions[dst]
            dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
            total += 1
            # Nearly axis-aligned: one dimension is <=25% of the other
            if dx == 0 or dy == 0 or min(dx, dy) / max(dx, dy) <= 0.25:
                ortho += 1
    return (ortho / total * 100) if total > 0 else 100


def _spread_score(positions, ncols, nrows):
    pts = list(positions.values())
    if len(pts) < 2:
        return float("inf")
    min_dists = []
    for i, p1 in enumerate(pts):
        min_d = min(
            math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            for j, p2 in enumerate(pts) if i != j
        )
        min_dists.append(min_d)
    return sum(min_dists) / len(min_dists)


# ---------------------------------------------------------------------------
# Rendering: apply layout to tex and compile PDF
# ---------------------------------------------------------------------------

def render_layout(engine_name: str, results: dict, base_tex: Path, output_dir: Path):
    """Apply layout to a copy of the tex file and compile to PDF."""
    out_tex = output_dir / f"qsp_schematic_{engine_name}.tex"
    shutil.copy(base_tex, out_tex)

    content = out_tex.read_text()
    comp_nodes = build_nodes()
    updates = {}
    for comp_name, positions in results.items():
        col_offset, row_offset = COMP_TEX_OFFSETS[comp_name]
        prefix = COMP_TO_PREFIX[comp_name]
        for n in comp_nodes[comp_name]:
            col, row = positions[n.name]
            updates[f"{prefix}_{n.name}"] = (col, row, col_offset, row_offset)

    pattern = re.compile(
        r'(\\node\[(?:pop|mol)\]\s*\((\w+)\)\s*at\s*\\gridcenterrel)\{(\d+)\}\{(\d+)\}\{(\\[A-Z]+)\}\{(\\[A-Z]+)\}'
    )
    def replacer(m):
        nid = m.group(2)
        if nid in updates:
            c, r, co, ro = updates[nid]
            return f"{m.group(1)}{{{c}}}{{{r}}}{{\\{co}}}{{\\{ro}}}"
        return m.group(0)

    out_tex.write_text(pattern.sub(replacer, content))

    # Compile
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", out_tex.name],
        capture_output=True, text=True, cwd=output_dir, timeout=30,
    )
    pdf = output_dir / f"qsp_schematic_{engine_name}.pdf"
    if pdf.exists():
        print(f"  {engine_name}: {pdf}")
    else:
        print(f"  {engine_name}: COMPILE FAILED", file=sys.stderr)
        # Show last few lines of log
        log = output_dir / f"qsp_schematic_{engine_name}.log"
        if log.exists():
            for line in log.read_text().split("\n")[-10:]:
                if line.strip():
                    print(f"    {line}", file=sys.stderr)


def render_current(base_tex: Path, output_dir: Path):
    """Compile the current (unmodified) tex as 'current' baseline."""
    out_tex = output_dir / "qsp_schematic_current.tex"
    shutil.copy(base_tex, out_tex)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", out_tex.name],
        capture_output=True, text=True, cwd=output_dir, timeout=30,
    )
    pdf = output_dir / "qsp_schematic_current.pdf"
    if pdf.exists():
        print(f"  current: {pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ENGINES = {
    "dot": layout_dot,
    "topo": layout_topo,
    "force": layout_force,
    "grandalf": layout_grandalf,
    "grandalf_z": layout_grandalf_zoned,
    "osage": layout_osage,
}

ENGINE_NAMES = list(ENGINES.keys())


def run_engine(engine_name: str) -> dict[str, dict[str, tuple[int, int]]]:
    comp_nodes = build_nodes()
    engine_fn = ENGINES[engine_name]
    return {
        comp_name: engine_fn(comp_name, nodes, get_internal_edges(comp_name))
        for comp_name, nodes in comp_nodes.items()
    }


def print_comparison(all_results: dict[str, dict]):
    comp_nodes = build_nodes()
    for comp_name in ["tumor"]:
        nodes = comp_nodes[comp_name]
        edges = get_internal_edges(comp_name)
        ncols, nrows = COMPARTMENTS[comp_name]

        print(f"\n{'='*80}")
        print(f"  {comp_name.upper()} ({ncols}x{nrows})")
        print(f"{'='*80}")

        engines = list(all_results.keys())
        header = f"  {'Node':<14s} {'Cur':>7s}"
        for eng in engines:
            header += f" {eng:>9s}"
        print(header)
        print("  " + "-" * (14 + 9 + 11 * len(engines)))

        for n in nodes:
            cur = f"({n.col},{n.row})"
            line = f"  {n.name:<14s} {cur:>7s}"
            for eng in engines:
                pos = all_results[eng].get(comp_name, {}).get(n.name, (n.col, n.row))
                s = f"({pos[0]},{pos[1]})"
                ch = "*" if pos != (n.col, n.row) else " "
                line += f" {s:>8s}{ch}"
            print(line)

        print()
        metric_header = f"  {'Metric':<20s}"
        for eng in engines:
            metric_header += f" {eng:>9s}"
        metric_header += f" {'current':>9s}"
        print(metric_header)

        current_pos = {n.name: (n.col, n.row) for n in nodes}
        for name, fn in [
            ("Crossings", lambda p: _count_crossings(comp_name, p, edges)),
            ("Edge length", lambda p: _total_edge_length(comp_name, p, edges)),
            ("Spread", lambda p: _spread_score(p, ncols, nrows)),
            ("Orthogonal %", lambda p: _orthogonality_score(comp_name, p, edges)),
        ]:
            line = f"  {name:<20s}"
            for eng in engines:
                line += f" {fn(all_results[eng].get(comp_name, {})):>9.1f}"
            line += f" {fn(current_pos):>9.1f}"
            print(line)


def apply_to_tex(results: dict, tex_path: Path):
    content = tex_path.read_text()
    comp_nodes = build_nodes()
    updates = {}
    for comp_name, positions in results.items():
        col_offset, row_offset = COMP_TEX_OFFSETS[comp_name]
        prefix = COMP_TO_PREFIX[comp_name]
        for n in comp_nodes[comp_name]:
            col, row = positions[n.name]
            updates[f"{prefix}_{n.name}"] = (col, row, col_offset, row_offset)

    pattern = re.compile(
        r'(\\node\[(?:pop|mol)\]\s*\((\w+)\)\s*at\s*\\gridcenterrel)\{(\d+)\}\{(\d+)\}\{(\\[A-Z]+)\}\{(\\[A-Z]+)\}'
    )
    def replacer(m):
        nid = m.group(2)
        if nid in updates:
            c, r, co, ro = updates[nid]
            return f"{m.group(1)}{{{c}}}{{{r}}}{{\\{co}}}{{\\{ro}}}"
        return m.group(0)

    new_content = pattern.sub(replacer, content)
    if new_content != content:
        tex_path.write_text(new_content)
        changes = sum(1 for a, b in zip(content.split('\n'), new_content.split('\n')) if a != b)
        print(f"Updated {changes} node positions in {tex_path}")
    else:
        print("No changes needed.")


def main():
    parser = argparse.ArgumentParser(description="Auto-layout QSP schematic")
    parser.add_argument("engine", choices=ENGINE_NAMES + ["all"],
                        help="Layout engine to use")
    parser.add_argument("--apply", action="store_true",
                        help="Patch qsp_schematic.tex in place")
    parser.add_argument("--render", action="store_true",
                        help="Compile PDFs for visual comparison")
    args = parser.parse_args()

    base_tex = Path(__file__).parent / "qsp_schematic.tex"
    render_dir = Path(__file__).parent / "layout_renders"

    if args.engine == "all":
        all_results = {}
        for eng in ENGINE_NAMES:
            print(f"Running {eng}...", file=sys.stderr)
            all_results[eng] = run_engine(eng)
        print_comparison(all_results)

        if args.render:
            render_dir.mkdir(exist_ok=True)
            print(f"\nRendering PDFs to {render_dir}/")
            render_current(base_tex, render_dir)
            for eng in ENGINE_NAMES:
                render_layout(eng, all_results[eng], base_tex, render_dir)
    else:
        results = run_engine(args.engine)
        comp_nodes = build_nodes()
        # Print tumor positions
        print(f"\n=== {args.engine} ===")
        for n in comp_nodes["tumor"]:
            pos = results["tumor"][n.name]
            cur = f"({n.col},{n.row})"
            new = f"({pos[0]},{pos[1]})"
            ch = " *" if pos != (n.col, n.row) else ""
            print(f"  {n.name:<14s} {cur:>8s} -> {new:>8s}{ch}")

        if args.apply:
            apply_to_tex(results, base_tex)
        if args.render:
            render_dir.mkdir(exist_ok=True)
            render_layout(args.engine, results, base_tex, render_dir)


if __name__ == "__main__":
    main()
