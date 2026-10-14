#!/usr/bin/env node
/**
 * ELK-based QSP Schematic Renderer
 *
 * Reads qsp_schematic.d2, builds an ELK graph with fixed compartment placement
 * and layered auto-layout inside each compartment, then emits a styled SVG.
 *
 * Usage: node render_schematic.mjs [output.svg]
 */

import ELK from "elkjs";
import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const D2_PATH = resolve(__dirname, "..", "qsp_schematic.d2");
const DEFAULT_OUT = resolve(__dirname, "..", "qsp_schematic_elk_rendered.svg");

// ─── Layout constants ────────────────────────────────────────────────────────
const COMP_GAP = 30; // gap between compartments

// ─── D2 parser ───────────────────────────────────────────────────────────────

function stripComment(line) {
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '"') inQuote = !inQuote;
    if (line[i] === '#' && !inQuote) return line.slice(0, i);
  }
  return line;
}

function parseD2(text) {
  const compartments = {}; // id -> { label, fill, stroke, nodes: { id -> {label, shape, fill} } }
  const edges = []; // { src, tgt, stroke, dash, arrowhead, bidir }

  let currentComp = null;
  let braceDepth = 0;
  let pendingEdge = null;
  let edgeBraceDepth = 0;

  for (const raw of text.split("\n")) {
    const line = stripComment(raw).trim();
    if (!line) continue;

    // Track brace depth for multi-line edge blocks
    if (pendingEdge !== null) {
      if (line.includes("{")) edgeBraceDepth++;
      if (line.includes("}")) edgeBraceDepth--;
      // Parse style properties inside edge block
      const strokeMatch = line.match(/style\.stroke:\s*"([^"]+)"/);
      if (strokeMatch) pendingEdge.stroke = strokeMatch[1];
      const dashMatch = line.match(/style\.stroke-dash:\s*(\d+)/);
      if (dashMatch) pendingEdge.dash = true;
      const arrowMatch = line.match(/target-arrowhead\.shape:\s*(\S+)/);
      if (arrowMatch) pendingEdge.arrowhead = arrowMatch[1];
      if (edgeBraceDepth <= 0) {
        edges.push(pendingEdge);
        pendingEdge = null;
      }
      continue;
    }

    // Compartment opening: "id: Label {"
    const compMatch = line.match(/^(\w+):\s*(.+?)\s*\{$/);
    if (compMatch && braceDepth === 0) {
      currentComp = compMatch[1];
      compartments[currentComp] = {
        label: compMatch[2],
        fill: "#ffffff",
        stroke: "#cccccc",
        nodes: {},
      };
      braceDepth = 1;
      continue;
    }

    if (currentComp && braceDepth === 1) {
      // Style lines inside compartment
      const fillMatch = line.match(/^style\.fill:\s*"([^"]+)"/);
      if (fillMatch) {
        compartments[currentComp].fill = fillMatch[1];
        continue;
      }
      const strokeMatch = line.match(/^style\.stroke:\s*"([^"]+)"/);
      if (strokeMatch) {
        compartments[currentComp].stroke = strokeMatch[1];
        continue;
      }
      if (line.startsWith("direction:")) continue;

      // Node def: "id: Label" or "id: Label {shape: circle}" or "id: Label {style.fill: ...}"
      const nodeMatch = line.match(
        /^(\w+):\s*(.+?)(?:\s*\{([^}]*)\})?$/
      );
      if (nodeMatch) {
        const [, nid, nlabel, attrs] = nodeMatch;
        const node = { label: nlabel.trim(), shape: "rectangle", fill: null };
        if (attrs) {
          if (attrs.includes("shape: circle")) node.shape = "circle";
          const nfill = attrs.match(/style\.fill:\s*"([^"]+)"/);
          if (nfill) node.fill = nfill[1];
        }
        // Handle nested brace for multi-line node defs (e.g. Cancer)
        if (line.endsWith("{") && !line.includes("}")) {
          braceDepth = 2;
          // Strip trailing " {" from label since regex couldn't match the closing "}"
          node.label = node.label.replace(/\s*\{$/, "");
          compartments[currentComp].nodes[nid] = node;
          continue;
        }
        compartments[currentComp].nodes[nid] = node;
        continue;
      }

      if (line === "}") {
        braceDepth = 0;
        currentComp = null;
        continue;
      }
    }

    // Inside nested node block (depth 2)
    if (currentComp && braceDepth === 2) {
      if (line === "}") {
        braceDepth = 1;
        continue;
      }
      // Parse style.fill inside nested node
      const nfill = line.match(/style\.fill:\s*"([^"]+)"/);
      if (nfill) {
        const lastNode = Object.keys(compartments[currentComp].nodes).pop();
        if (lastNode) compartments[currentComp].nodes[lastNode].fill = nfill[1];
      }
      continue;
    }

    // Close brace at top level
    if (line === "}" && braceDepth > 0) {
      braceDepth--;
      if (braceDepth === 0) currentComp = null;
      continue;
    }

    // Edge lines: "src -> tgt: {style.stroke: ...}" or multi-line
    const edgeMatch = line.match(
      /^([\w.]+)\s*(<->|->)\s*([\w.]+):\s*(.*)$/
    );
    if (edgeMatch) {
      const [, src, dir, tgt, rest] = edgeMatch;
      const edge = {
        src,
        tgt,
        bidir: dir === "<->",
        stroke: "#333333",
        dash: false,
        arrowhead: "triangle",
      };
      // Inline style
      const sMatch = rest.match(/style\.stroke:\s*"([^"]+)"/);
      if (sMatch) edge.stroke = sMatch[1];
      const dMatch = rest.match(/style\.stroke-dash:\s*\d+/);
      if (dMatch) edge.dash = true;
      const aMatch = rest.match(/target-arrowhead\.shape:\s*(\S+)/);
      if (aMatch) edge.arrowhead = aMatch[1];

      // Multi-line block?
      if (rest.trim().endsWith("{") && !rest.includes("}")) {
        pendingEdge = edge;
        edgeBraceDepth = 1;
      } else {
        edges.push(edge);
      }
      continue;
    }
  }

  return { compartments, edges };
}

// ─── Build ELK graph ─────────────────────────────────────────────────────────

const NODE_W = 80;
const NODE_H = 36;
const CIRCLE_R = 36;

function buildElkGraph(compartments, edges) {
  const children = [];

  for (const [cid, comp] of Object.entries(compartments)) {
    const compNode = {
      id: cid,
      x: 0,
      y: 0,
      labels: [{ text: comp.label }],
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "DOWN",
        "elk.layered.spacing.nodeNodeBetweenLayers": "40",
        "elk.spacing.nodeNode": "20",
        "elk.padding": "[top=40,left=20,bottom=20,right=20]",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      },
      children: [],
      edges: [],
    };

    // Blood: layered RIGHT with orthogonal routing
    if (cid === "blood") {
      compNode.layoutOptions["elk.direction"] = "RIGHT";
      compNode.layoutOptions["elk.layered.edgeRouting"] = "ORTHOGONAL";
      compNode.layoutOptions["elk.layered.spacing.nodeNodeBetweenLayers"] = "30";
      compNode.layoutOptions["elk.spacing.nodeNode"] = "15";
    }

    // dLN: spread edges apart to reduce overlap
    if (cid === "dln") {
      compNode.layoutOptions["elk.layered.edgeRouting"] = "ORTHOGONAL";
      compNode.layoutOptions["elk.layered.spacing.edgeEdgeBetweenLayers"] = "15";
      compNode.layoutOptions["elk.layered.spacing.edgeNodeBetweenLayers"] = "15";
      compNode.layoutOptions["elk.spacing.edgeEdge"] = "10";
      compNode.layoutOptions["elk.spacing.edgeNode"] = "10";
    }

    // Tumor gets tighter spacing
    if (cid === "tumor") {
      compNode.layoutOptions["elk.layered.spacing.nodeNodeBetweenLayers"] = "35";
      compNode.layoutOptions["elk.spacing.nodeNode"] = "15";
      compNode.layoutOptions["elk.padding"] = "[top=40,left=15,bottom=15,right=15]";
      compNode.layoutOptions["elk.layered.edgeRouting"] = "ORTHOGONAL";
    }

    for (const [nid, node] of Object.entries(comp.nodes)) {
      const isCircle = node.shape === "circle";
      compNode.children.push({
        id: `${cid}.${nid}`,
        width: isCircle ? CIRCLE_R : NODE_W,
        height: isCircle ? CIRCLE_R : NODE_H,
        labels: [{ text: node.label }],
      });
    }

    children.push(compNode);
  }

  // Partition edges: intra-compartment go inside compartment, cross-compartment at root
  const rootEdges = [];
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const srcComp = e.src.split(".")[0];
    const tgtComp = e.tgt.split(".")[0];
    const edgeObj = {
      id: `e${i}`,
      sources: [e.src],
      targets: [e.tgt],
    };
    if (srcComp === tgtComp) {
      const compChild = children.find((c) => c.id === srcComp);
      if (compChild) compChild.edges.push(edgeObj);
    } else {
      rootEdges.push(edgeObj);
    }
  }

  return {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "fixed",
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
    },
    children,
    edges: rootEdges,
  };
}

// ─── SVG renderer ────────────────────────────────────────────────────────────

function escapeXml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderSvg(layoutGraph, compartments, edges) {
  // Compute total bounds
  let maxX = 0,
    maxY = 0;
  for (const comp of layoutGraph.children) {
    maxX = Math.max(maxX, (comp.x || 0) + (comp.width || 0));
    maxY = Math.max(maxY, (comp.y || 0) + (comp.height || 0));
  }
  const pad = 40;
  const W = maxX + 2 * pad;
  const H = maxY + 2 * pad;

  const parts = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Helvetica, Arial, sans-serif">`
  );
  parts.push(`<rect width="${W}" height="${H}" fill="white"/>`);

  // Arrowhead defs
  parts.push(`<defs>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#3366cc"/>
  </marker>
  <marker id="arrow-purple" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#9933cc"/>
  </marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#cc3333"/>
  </marker>
  <marker id="arrow-teal" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#009688"/>
  </marker>
  <marker id="tee-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <line x1="9" y1="1" x2="9" y2="9" stroke="#cc3333" stroke-width="3"/>
  </marker>
  <marker id="circle-teal" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto">
    <circle cx="5" cy="5" r="4" fill="none" stroke="#009688" stroke-width="1.5"/>
  </marker>
</defs>`);

  // Build node position lookup
  const nodePos = {};
  for (const comp of layoutGraph.children) {
    const cx = (comp.x || 0) + pad;
    const cy = (comp.y || 0) + pad;
    if (comp.children) {
      for (const n of comp.children) {
        nodePos[n.id] = {
          x: cx + (n.x || 0),
          y: cy + (n.y || 0),
          w: n.width,
          h: n.height,
        };
      }
    }
  }

  // Build circle lookup for edge routing
  const isCircleNode = {};
  for (const [cid, comp] of Object.entries(compartments)) {
    for (const [nid, node] of Object.entries(comp.nodes)) {
      isCircleNode[`${cid}.${nid}`] = node.shape === "circle";
    }
  }

  // Draw compartment backgrounds
  for (const comp of layoutGraph.children) {
    const cx = (comp.x || 0) + pad;
    const cy = (comp.y || 0) + pad;
    const cd = compartments[comp.id];
    parts.push(`<g id="comp-${comp.id}">`);
    parts.push(
      `<rect x="${cx}" y="${cy}" width="${comp.width}" height="${comp.height}" rx="12" ry="12" fill="${cd.fill}" stroke="${cd.stroke}" stroke-width="2" opacity="0.8"/>`
    );
    parts.push(
      `<text x="${cx + 12}" y="${cy + 22}" font-size="15" font-weight="bold" fill="${cd.stroke}">${escapeXml(cd.label)}</text>`
    );
    parts.push(`</g>`);
  }

  // Pre-group cross-compartment edges by compartment pair for fanning
  const crossCompGroups = {};
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const srcComp = e.src.split(".")[0];
    const tgtComp = e.tgt.split(".")[0];
    if (srcComp !== tgtComp) {
      const pairKey = [srcComp, tgtComp].sort().join("<->");
      if (!crossCompGroups[pairKey]) crossCompGroups[pairKey] = [];
      crossCompGroups[pairKey].push(i);
    }
  }
  // Build index map: edgeIndex -> { idx, count } within its group
  const crossCompFan = {};
  for (const group of Object.values(crossCompGroups)) {
    for (let j = 0; j < group.length; j++) {
      crossCompFan[group[j]] = { idx: j, count: group.length };
    }
  }

  // Draw edges (on top of nodes)
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const src = nodePos[e.src];
    const tgt = nodePos[e.tgt];
    if (!src || !tgt) continue;

    let markerEnd = "";
    let markerStart = "";
    const color = e.stroke;
    const sw = 1.8;

    if (e.arrowhead === "cf-one") {
      markerEnd = ' marker-end="url(#tee-red)"';
    } else if (e.dash) {
      markerEnd = ' marker-end="url(#circle-teal)"';
    } else if (e.bidir) {
      markerEnd = ' marker-end="url(#arrow-purple)"';
      markerStart = ' marker-start="url(#arrow-purple)"';
    } else {
      markerEnd = ' marker-end="url(#arrow-blue)"';
    }

    const dashAttr = e.dash ? ' stroke-dasharray="6 4"' : "";

    parts.push(`<g id="edge-${e.src}-to-${e.tgt}">`);
    const elkEdge = findElkEdge(layoutGraph, `e${i}`);
    if (elkEdge && elkEdge.sections && elkEdge.sections.length > 0) {
      for (const section of elkEdge.sections) {
        const points = buildEdgePath(section, layoutGraph, e.src, pad);
        parts.push(
          `<path d="${points}" fill="none" stroke="${color}" stroke-width="${sw}"${dashAttr}${markerStart}${markerEnd}/>`
        );
      }
    } else {
      const srcComp = e.src.split(".")[0];
      const tgtComp = e.tgt.split(".")[0];

      if (srcComp !== tgtComp) {
        const fan = crossCompFan[i] || { idx: 0, count: 1 };
        const pathD = routeOrthogonal(
          src, tgt,
          isCircleNode[e.src] || false,
          isCircleNode[e.tgt] || false,
          fan.idx, fan.count
        );
        parts.push(
          `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="${sw}"${dashAttr}${markerStart}${markerEnd}/>`
        );
      } else {
        const sx = src.x + src.w / 2;
        const sy = src.y + src.h / 2;
        const tx = tgt.x + tgt.w / 2;
        const ty = tgt.y + tgt.h / 2;
        parts.push(
          `<line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="${color}" stroke-width="${sw}"${dashAttr}${markerStart}${markerEnd}/>`
        );
      }
    }
    parts.push(`</g>`);
  }

  // Draw nodes (on top of edges)
  for (const comp of layoutGraph.children) {
    const cx = (comp.x || 0) + pad;
    const cy = (comp.y || 0) + pad;
    const cd = compartments[comp.id];
    if (!comp.children) continue;
    for (const n of comp.children) {
      const nx = cx + (n.x || 0);
      const ny = cy + (n.y || 0);
      const nid = n.id.split(".")[1];
      const nodeData = cd.nodes[nid];
      const isCircle = nodeData && nodeData.shape === "circle";
      const fill = (nodeData && nodeData.fill) || "#ffffff";
      const label = n.labels?.[0]?.text || nid;

      parts.push(`<g id="node-${n.id}">`);
      if (isCircle) {
        const r = n.width / 2;
        parts.push(
          `<circle cx="${nx + r}" cy="${ny + r}" r="${r}" fill="${fill}" stroke="#666" stroke-width="1.5"/>`
        );
        parts.push(
          `<text x="${nx + r}" y="${ny + r + 4}" text-anchor="middle" font-size="11" fill="#333">${escapeXml(label)}</text>`
        );
      } else {
        parts.push(
          `<rect x="${nx}" y="${ny}" width="${n.width}" height="${n.height}" rx="6" ry="6" fill="${fill}" stroke="#666" stroke-width="1.5"/>`
        );
        parts.push(
          `<text x="${nx + n.width / 2}" y="${ny + n.height / 2 + 4}" text-anchor="middle" font-size="12" fill="#333">${escapeXml(label)}</text>`
        );
      }
      parts.push(`</g>`);
    }
  }

  parts.push("</svg>");
  return parts.join("\n");
}

function findElkEdge(graph, edgeId) {
  // Search root edges
  if (graph.edges) {
    const found = graph.edges.find((e) => e.id === edgeId);
    if (found) return found;
  }
  // Search compartment edges
  if (graph.children) {
    for (const child of graph.children) {
      if (child.edges) {
        const found = child.edges.find((e) => e.id === edgeId);
        if (found) return found;
      }
    }
  }
  return null;
}

function buildEdgePath(section, layoutGraph, srcNodeId, pad) {
  // Determine offset: if edge is inside a compartment, add compartment position
  const compId = srcNodeId.split(".")[0];
  const comp = layoutGraph.children.find((c) => c.id === compId);
  let ox = pad,
    oy = pad;
  if (comp) {
    ox += comp.x || 0;
    oy += comp.y || 0;
  }

  const start = section.startPoint;
  const end = section.endPoint;
  const bends = section.bendPoints || [];

  let d = `M ${start.x + ox} ${start.y + oy}`;
  for (const b of bends) {
    d += ` L ${b.x + ox} ${b.y + oy}`;
  }
  d += ` L ${end.x + ox} ${end.y + oy}`;
  return d;
}

// ─── Post-layout compartment repositioning ────────────────────────────────────

/**
 * After ELK lays out each compartment's children, reposition compartments
 * to avoid overlap. Uses a simple grid: top row (peripheral, central, injection)
 * and bottom row (dln, tumor), left-to-right with gaps.
 */
function repositionCompartments(layoutGraph) {
  const gap = COMP_GAP;
  const byId = {};
  for (const c of layoutGraph.children) byId[c.id] = c;

  // Row 1: injection, dln, tumor (left to right)
  const injW = byId.injection ? byId.injection.width || 0 : 0;
  const dlnW = byId.dln ? byId.dln.width || 0 : 0;

  if (byId.injection) { byId.injection.x = 0; byId.injection.y = 0; }
  if (byId.dln) { byId.dln.x = injW + gap; byId.dln.y = 0; }
  if (byId.tumor) { byId.tumor.x = injW + gap + dlnW + gap; byId.tumor.y = 0; }

  // Row 2: blood below dln, left of tumor
  const dlnH = byId.dln ? byId.dln.height || 0 : 0;
  const row2Y = dlnH + gap;

  const dlnLeft = byId.dln ? byId.dln.x : 0;
  if (byId.blood) { byId.blood.x = dlnLeft; byId.blood.y = row2Y; }
}

// ─── Orthogonal edge routing for cross-compartment edges ──────────────────────

/**
 * Compute an anchor point on the border of a node, choosing the side
 * closest to the target point.
 */
function borderAnchor(node, targetX, targetY, isCircle) {
  const cx = node.x + node.w / 2;
  const cy = node.y + node.h / 2;
  const dx = targetX - cx;
  const dy = targetY - cy;

  if (isCircle) {
    const r = node.w / 2;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    return { x: cx + (dx / dist) * r, y: cy + (dy / dist) * r };
  }

  // Rectangle: pick the side with the largest projection
  const hw = node.w / 2;
  const hh = node.h / 2;

  // Scale factors to reach each edge
  const sx = hw / (Math.abs(dx) || 1);
  const sy = hh / (Math.abs(dy) || 1);
  const s = Math.min(sx, sy);

  return { x: cx + dx * s, y: cy + dy * s };
}

/**
 * Route an edge orthogonally (Manhattan-style) between two nodes in different
 * compartments. Fans edges apart using perpendicular offsets so parallel
 * edges between the same compartment pair don't overlap.
 */
function routeOrthogonal(src, tgt, srcIsCircle, tgtIsCircle, fanIdx = 0, fanCount = 1) {
  const EDGE_FAN_SPACING = 12;
  const fanOffset = (fanIdx - (fanCount - 1) / 2) * EDGE_FAN_SPACING;

  const scx = src.x + src.w / 2;
  const scy = src.y + src.h / 2;
  const tcx = tgt.x + tgt.w / 2;
  const tcy = tgt.y + tgt.h / 2;
  const dx = tcx - scx;
  const dy = tcy - scy;

  const verticalFirst = Math.abs(dy) >= Math.abs(dx);

  let points;
  if (verticalFirst) {
    // Primary travel is vertical — fan by offsetting X (perpendicular)
    const exitY = dy > 0 ? src.y + src.h : src.y;
    const enterY = dy > 0 ? tgt.y : tgt.y + tgt.h;
    const midY = (exitY + enterY) / 2;
    const fanX = fanOffset; // shift horizontally

    const srcAnchor = borderAnchor(src, scx + fanX, midY, srcIsCircle);
    const tgtAnchor = borderAnchor(tgt, tcx + fanX, midY, tgtIsCircle);
    points = [
      srcAnchor,
      { x: srcAnchor.x, y: midY },
      { x: tgtAnchor.x, y: midY },
      tgtAnchor,
    ];
  } else {
    // Primary travel is horizontal — fan by offsetting Y (perpendicular)
    const exitX = dx > 0 ? src.x + src.w : src.x;
    const enterX = dx > 0 ? tgt.x : tgt.x + tgt.w;
    const midX = (exitX + enterX) / 2;
    const fanY = fanOffset; // shift vertically

    const srcAnchor = borderAnchor(src, midX, scy + fanY, srcIsCircle);
    const tgtAnchor = borderAnchor(tgt, midX, tcy + fanY, tgtIsCircle);
    points = [
      srcAnchor,
      { x: midX, y: srcAnchor.y },
      { x: midX, y: tgtAnchor.y },
      tgtAnchor,
    ];
  }

  const filtered = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const prev = filtered[filtered.length - 1];
    if (Math.abs(points[i].x - prev.x) > 0.5 || Math.abs(points[i].y - prev.y) > 0.5) {
      filtered.push(points[i]);
    }
  }

  let d = `M ${filtered[0].x} ${filtered[0].y}`;
  for (let i = 1; i < filtered.length; i++) {
    d += ` L ${filtered[i].x} ${filtered[i].y}`;
  }
  return d;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const outPath = process.argv[2] || DEFAULT_OUT;

  console.log("Reading D2 file:", D2_PATH);
  const d2Text = readFileSync(D2_PATH, "utf-8");
  const { compartments, edges } = parseD2(d2Text);

  console.log(
    `Parsed: ${Object.keys(compartments).length} compartments, ${edges.length} edges`
  );
  for (const [cid, c] of Object.entries(compartments)) {
    console.log(`  ${cid}: ${Object.keys(c.nodes).length} nodes`);
  }

  const elkGraph = buildElkGraph(compartments, edges);
  console.log("Running ELK layout...");

  const elk = new ELK();
  const laid = await elk.layout(elkGraph);

  console.log("Layout complete. Repositioning compartments...");
  repositionCompartments(laid);
  console.log("Rendering SVG...");
  const svg = renderSvg(laid, compartments, edges);

  writeFileSync(outPath, svg);
  console.log(`Written: ${outPath} (${(svg.length / 1024).toFixed(1)} KB)`);

  // Also export draw.io XML
  const drawioPath = outPath.replace(/\.svg$/, '.drawio');
  const drawioXml = renderDrawio(laid, compartments, edges);
  writeFileSync(drawioPath, drawioXml);
  console.log(`Written: ${drawioPath} (${(drawioXml.length / 1024).toFixed(1)} KB)`);
}

// ─── Draw.io (mxGraph XML) exporter ──────────────────────────────────────────

function renderDrawio(layoutGraph, compartments, edges) {
  const pad = 40;
  let cellId = 2; // 0 and 1 are reserved for root and default parent

  const nodePos = {};
  const nodeCellIds = {};

  const cells = [];

  // Compartment backgrounds as parent containers
  const compCellIds = {};
  for (const comp of layoutGraph.children) {
    const cx = (comp.x || 0) + pad;
    const cy = (comp.y || 0) + pad;
    const cd = compartments[comp.id];
    const id = cellId++;
    compCellIds[comp.id] = id;
    cells.push(
      `      <mxCell id="${id}" value="${escapeXml(cd.label)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=${cd.fill};strokeColor=${cd.stroke};opacity=80;verticalAlign=top;fontStyle=1;fontSize=13;spacingTop=2;arcSize=8;" vertex="1" parent="1">` +
      `\n        <mxGeometry x="${cx}" y="${cy}" width="${comp.width}" height="${comp.height}" as="geometry"/>` +
      `\n      </mxCell>`
    );
  }

  // Nodes
  for (const comp of layoutGraph.children) {
    const cx = (comp.x || 0) + pad;
    const cy = (comp.y || 0) + pad;
    const cd = compartments[comp.id];
    if (!comp.children) continue;
    for (const n of comp.children) {
      const nx = cx + (n.x || 0);
      const ny = cy + (n.y || 0);
      const nid = n.id.split(".")[1];
      const nodeData = cd.nodes[nid];
      const isCircle = nodeData && nodeData.shape === "circle";
      const fill = (nodeData && nodeData.fill) || "#ffffff";
      const label = n.labels?.[0]?.text || nid;
      const id = cellId++;
      nodeCellIds[n.id] = id;
      nodePos[n.id] = { x: nx, y: ny, w: n.width, h: n.height };

      let style;
      if (isCircle) {
        style = `ellipse;whiteSpace=wrap;html=1;fillColor=${fill};strokeColor=#666666;fontSize=11;`;
      } else {
        style = `rounded=1;whiteSpace=wrap;html=1;fillColor=${fill};strokeColor=#666666;fontSize=12;arcSize=20;`;
      }
      cells.push(
        `      <mxCell id="${id}" value="${escapeXml(label)}" style="${style}" vertex="1" parent="1">` +
        `\n        <mxGeometry x="${nx}" y="${ny}" width="${n.width}" height="${n.height}" as="geometry"/>` +
        `\n      </mxCell>`
      );
    }
  }

  // Build circle lookup
  const isCircleNode = {};
  for (const [cid, comp] of Object.entries(compartments)) {
    for (const [nid, node] of Object.entries(comp.nodes)) {
      isCircleNode[`${cid}.${nid}`] = node.shape === "circle";
    }
  }

  // Pre-group cross-compartment edges for fanning
  const crossCompGroups = {};
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const srcComp = e.src.split(".")[0];
    const tgtComp = e.tgt.split(".")[0];
    if (srcComp !== tgtComp) {
      const pairKey = [srcComp, tgtComp].sort().join("<->");
      if (!crossCompGroups[pairKey]) crossCompGroups[pairKey] = [];
      crossCompGroups[pairKey].push(i);
    }
  }
  const crossCompFan = {};
  for (const group of Object.values(crossCompGroups)) {
    for (let j = 0; j < group.length; j++) {
      crossCompFan[group[j]] = { idx: j, count: group.length };
    }
  }

  // Edges
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const srcId = nodeCellIds[e.src];
    const tgtId = nodeCellIds[e.tgt];
    if (!srcId || !tgtId) continue;

    const id = cellId++;
    const color = e.stroke;
    let endArrow = "classic";
    let startArrow = "none";
    let dashed = e.dash ? "1" : "0";

    if (e.arrowhead === "cf-one") {
      endArrow = "dash"; // tee/inhibition
    } else if (e.dash) {
      endArrow = "oval";
    } else if (e.bidir) {
      endArrow = "classic";
      startArrow = "classic";
    }

    const style = `edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=${color};strokeWidth=1.8;endArrow=${endArrow};startArrow=${startArrow};dashed=${dashed};`;

    cells.push(
      `      <mxCell id="${id}" style="${style}" edge="1" source="${srcId}" target="${tgtId}" parent="1">` +
      `\n        <mxGeometry relative="1" as="geometry"/>` +
      `\n      </mxCell>`
    );
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<mxfile>
  <diagram name="QSP Schematic">
    <mxGraphModel>
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
${cells.join("\n")}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
