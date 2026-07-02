#!/usr/bin/env python3
"""Parse qsp_schematic.tex and render an ASCII grid with edges drawn."""
import re, sys, math

# Compartment offsets (col_start, row_start)
COMPARTMENTS = {
    'PERI': (0, 0),
    'DLN': (0, 2),
    'CENT': (5, 2),
    'TUM': (10, 2),
    'ID': (0, 17),
}

GRID_COLS = 23
GRID_ROWS = 20
CELL_W = 9
CELL_H = 3  # taller cells for edge visibility

def parse_nodes(tex_path):
    """Extract node positions and labels from tex file."""
    nodes = []
    with open(tex_path) as f:
        text = f.read()
    pat = re.compile(
        r'\\node\[.*?\]\s*\((\w+)\)\s*at\s*\\gridcenterrel\{(\d+)\}\{(\d+)\}'
        r'\{\\(\w+)\}\{\\(\w+)\}\s*\{([^}]*)\}',
        re.DOTALL
    )
    for m in pat.finditer(text):
        name = m.group(1)
        rel_col = int(m.group(2))
        rel_row = int(m.group(3))
        col_start_name = m.group(4)
        comp_map = {
            'PERICOLSTART': 'PERI', 'DLNCOLSTART': 'DLN',
            'CENTCOLSTART': 'CENT', 'TUMCOLSTART': 'TUM',
            'IDCOLSTART': 'ID',
        }
        comp = comp_map.get(col_start_name, '?')
        col_off, row_off = COMPARTMENTS.get(comp, (0, 0))
        abs_col = col_off + rel_col
        abs_row = row_off + rel_row
        label = m.group(6).replace('\\', '').replace('$', '').replace('gamma', 'γ').replace('beta', 'β')
        nodes.append((abs_col, abs_row, label, name))
    return nodes

def parse_edges(tex_path):
    """Extract edges with routing info from tex file."""
    edges = []
    with open(tex_path) as f:
        text = f.read()
    # Match direct edges: \draw[style] (src) ROUTING (dst)
    pat = re.compile(r'\\draw\[([^\]]*)\]\s*\(([^)]+)\)\s*([-|]+)\s*\(([^)]+)\)')
    for m in pat.finditer(text):
        style = m.group(1)
        src = m.group(2).strip()
        routing = m.group(3)
        dst = m.group(4).strip()
        # Classify style
        if 'inhib' in style:
            kind = 'inhib'
        elif 'bidir' in style:
            kind = 'bidir'
        elif 'branch' in style:
            kind = 'branch'
        elif 'act' in style:
            kind = 'act'
        elif 'line' in style:
            kind = 'line'
        else:
            kind = 'other'
        edges.append((src, dst, routing, kind, style))
    return edges

EDGE_CHARS = {
    'act':    '·',
    'inhib':  '×',
    'bidir':  '=',
    'branch': '~',
    'line':   '.',
    'other':  '·',
}

def grid_to_pixel(col, row):
    """Convert grid coordinates to character coordinates (center of cell)."""
    return col * CELL_W + CELL_W // 2, row * CELL_H + CELL_H // 2

def draw_line(grid, x0, y0, x1, y1, ch, width, height):
    """Draw a line on the grid using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            current = grid[y0][x0]
            if current == ' ' or current in '─│┌┐└┘':
                grid[y0][x0] = ch
            elif current != ch and current not in ' ─│┌┐└┘':
                grid[y0][x0] = 'X'  # collision marker
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def render_ascii(nodes, edges):
    """Render nodes and edges on an ASCII grid."""
    width = GRID_COLS * CELL_W + 1
    height = GRID_ROWS * CELL_H + 1
    grid = [[' '] * width for _ in range(height)]

    # Draw compartment boundaries
    compartment_bounds = [
        ('Peripheral', 0, 23, 0, 2),
        ('dLN', 0, 5, 2, 17),
        ('Central', 5, 10, 2, 11),
        ('Tumor', 10, 23, 2, 17),
        ('Injection', 0, 10, 17, 20),
    ]
    for name, c0, c1, r0, r1 in compartment_bounds:
        x0, x1 = c0 * CELL_W, min(c1 * CELL_W, width - 1)
        y0, y1 = r0 * CELL_H, min(r1 * CELL_H, height - 1)
        for x in range(x0, x1 + 1):
            if grid[y0][x] == ' ': grid[y0][x] = '─'
            if grid[y1][x] == ' ': grid[y1][x] = '─'
        for y in range(y0, y1 + 1):
            if grid[y][x0] == ' ': grid[y][x0] = '│'
            if grid[y][x1] == ' ': grid[y][x1] = '│'
        grid[y0][x0] = '┌'; grid[y0][x1] = '┐'
        grid[y1][x0] = '└'; grid[y1][x1] = '┘'
        lbl = f' {name} '
        for i, ch in enumerate(lbl):
            if x0 + 1 + i < width:
                grid[y0][x0 + 1 + i] = ch

    # Build node position lookup
    node_pos = {}
    node_label = {}
    for col, row, label, name in nodes:
        node_pos[name] = (col, row)
        node_label[name] = label

    # Draw edges FIRST (so nodes overwrite them)
    edge_info = []
    for src, dst, routing, kind, style in edges:
        # Only draw edges between known grid nodes
        if src not in node_pos or dst not in node_pos:
            edge_info.append((src, dst, routing, kind, '?', '?'))
            continue
        sc, sr = node_pos[src]
        dc, dr = node_pos[dst]
        sx, sy = grid_to_pixel(sc, sr)
        dx, dy = grid_to_pixel(dc, dr)
        ch = EDGE_CHARS.get(kind, '·')

        if routing == '|-':
            # vertical first, then horizontal
            draw_line(grid, sx, sy, sx, dy, ch, width, height)
            draw_line(grid, sx, dy, dx, dy, ch, width, height)
        elif routing == '-|':
            # horizontal first, then vertical
            draw_line(grid, sx, sy, dx, sy, ch, width, height)
            draw_line(grid, dx, sy, dx, dy, ch, width, height)
        else:
            # straight line
            draw_line(grid, sx, sy, dx, dy, ch, width, height)

        edge_info.append((src, dst, routing, kind, (sc, sr), (dc, dr)))

    # Place nodes ON TOP of edges
    for col, row, label, name in nodes:
        cx, cy = grid_to_pixel(col, row)
        lbl = label[:7]
        # Draw a box around the label
        start_x = cx - len(lbl) // 2 - 1
        end_x = start_x + len(lbl) + 1
        for x in range(start_x, end_x + 1):
            if 0 <= x < width:
                if 0 <= cy - 1 < height: grid[cy - 1][x] = ' '
                if 0 <= cy < height: grid[cy][x] = ' '
                if 0 <= cy + 1 < height: grid[cy + 1][x] = ' '
        # Write label
        for i, ch in enumerate(lbl):
            x = cx - len(lbl) // 2 + i
            if 0 <= x < width and 0 <= cy < height:
                grid[cy][x] = ch

    # Print grid
    for row in grid:
        print(''.join(row))

    # Print edge list
    print("\n=== EDGES ===")
    for src, dst, routing, kind, sp, dp in edge_info:
        src_l = node_label.get(src, src)
        dst_l = node_label.get(dst, dst)
        sym = {'act': '→', 'inhib': '⊣', 'bidir': '↔', 'branch': '○', 'line': '—'}.get(kind, '?')
        print(f"  {src_l:>8} {sym} {dst_l:<8}  {routing:3s}  ({sp} → {dp})")

    # Print collision report
    print("\n=== EDGE COLLISIONS (X marks) ===")
    collisions = []
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 'X':
                # Find which grid cell this is in
                gc = x // CELL_W
                gr = y // CELL_H
                collisions.append((gc, gr))
    if collisions:
        # Group by grid cell
        from collections import Counter
        counts = Counter(collisions)
        for (gc, gr), cnt in counts.most_common(20):
            # Find edges passing through this cell
            print(f"  Grid cell ({gc},{gr}): {cnt} collision pixel(s)")
    else:
        print("  No collisions detected!")

    # Print overlap summary — edges sharing same grid cells
    print("\n=== PARALLEL/OVERLAPPING EDGES ===")
    from collections import defaultdict
    edge_cells = defaultdict(list)
    for src, dst, routing, kind, sp, dp in edge_info:
        if sp == '?' or dp == '?':
            continue
        sc, sr = sp
        dc, dr = dp
        # Trace cells the edge passes through
        cells = set()
        if routing == '|-':
            for r in range(min(sr, dr), max(sr, dr) + 1):
                cells.add((sc, r))
            for c in range(min(sc, dc), max(sc, dc) + 1):
                cells.add((c, dr))
        elif routing == '-|':
            for c in range(min(sc, dc), max(sc, dc) + 1):
                cells.add((c, sr))
            for r in range(min(sr, dr), max(sr, dr) + 1):
                cells.add((dc, r))
        else:
            steps = max(abs(dc - sc), abs(dr - sr), 1)
            for i in range(steps + 1):
                t = i / steps
                c = round(sc + t * (dc - sc))
                r = round(sr + t * (dr - sr))
                cells.add((c, r))
        src_l = node_label.get(src, src)
        dst_l = node_label.get(dst, dst)
        edge_name = f"{src_l}→{dst_l}"
        for cell in cells:
            edge_cells[cell].append(edge_name)

    # Find cells with multiple edges
    overlaps = {cell: names for cell, names in edge_cells.items() if len(names) > 1}
    if overlaps:
        # Group overlapping edge pairs
        from itertools import combinations
        pair_counts = Counter()
        for cell, names in overlaps.items():
            for a, b in combinations(sorted(set(names)), 2):
                pair_counts[(a, b)] += 1
        for (a, b), cnt in pair_counts.most_common(15):
            print(f"  [{cnt} shared cells] {a}  ||  {b}")
    else:
        print("  No overlapping edges detected!")


if __name__ == '__main__':
    tex = sys.argv[1] if len(sys.argv) > 1 else 'qsp_schematic.tex'
    nodes = parse_nodes(tex)
    edges = parse_edges(tex)
    render_ascii(nodes, edges)
