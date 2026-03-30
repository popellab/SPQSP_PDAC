#!/usr/bin/env python3
"""
bootstrap_abm_xml.py — One-time script to annotate param_all_test.xml with
name/type/val attributes from the current hand-written gpu_param.h and gpu_param.cu.

Uses text-based insertion to preserve XML comments and formatting.

Usage:
    python bootstrap_abm_xml.py <gpu_param_h> <gpu_param_cu> <xml_file> <output_xml>
"""

import re
import sys


def parse_enums(h_path):
    """Parse GPUParamFloat/Int/Bool enums from gpu_param.h."""
    with open(h_path) as f:
        text = f.read()
    enums = {}
    for label, tag in [('GPUParamFloat', 'float'),
                       ('GPUParamInt', 'int'),
                       ('GPUParamBool', 'bool')]:
        m = re.search(rf'enum\s+{label}\s*\{{(.*?)\}}', text, re.DOTALL)
        if not m:
            print(f"ERROR: could not find enum {label}", file=sys.stderr)
            sys.exit(1)
        names = []
        for line in m.group(1).splitlines():
            line = line.strip().rstrip(',')
            if not line or line.startswith('//') or line.startswith('/*') or '_COUNT' in line:
                continue
            name = line.split('//')[0].strip().rstrip(',')
            if name:
                names.append(name)
        enums[tag] = names
    return enums


def parse_descriptions(cu_path):
    """Parse _gpu_param_description from gpu_param.cu."""
    with open(cu_path) as f:
        text = f.read()
    m = re.search(r'_gpu_param_description\[\]\[3\]\s*=\s*\{(.*?)\};', text, re.DOTALL)
    if not m:
        print("ERROR: could not find _gpu_param_description", file=sys.stderr)
        sys.exit(1)
    entries = []
    for line in m.group(1).splitlines():
        m2 = re.search(r'\{"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\}', line)
        if m2:
            entries.append((m2.group(1), m2.group(2), m2.group(3)))
    return entries


def annotate_xml_text(xml_path, params, output_path):
    """Text-based XML annotation that preserves comments and formatting.

    params: list of (name, type, xml_dotpath, units, val)
    """
    with open(xml_path) as f:
        lines = f.readlines()

    # Build lookup: xml_dotpath -> (name, type, units, val)
    lookup = {}
    for name, ptype, dotpath, units, val in params:
        lookup[dotpath] = (name, ptype, units, val)

    # Track current XML path via a stack
    stack = []
    output = []

    for line in lines:
        stripped = line.strip()

        # Skip pure comments or empty lines
        if stripped.startswith('<!--') and stripped.endswith('-->'):
            output.append(line)
            continue
        if not stripped:
            output.append(line)
            continue

        # Detect closing tags: </Tag>
        close_match = re.match(r'^(\s*)</(\w+)>', stripped)

        # Detect opening tags: <Tag> or <Tag>value</Tag> or self-closing
        # We need to handle: <Tag>value</Tag>, <Tag>, </Tag>
        open_match = re.match(r'<(\w+)', stripped)

        if stripped.startswith('</'):
            # Pure closing tag
            if stack:
                stack.pop()
            output.append(line)
            continue

        if open_match:
            tag = open_match.group(1)
            current_path = '.'.join(stack + [tag])

            if current_path in lookup:
                name, ptype, units, val = lookup[current_path]
                # Build attribute string
                attrs = f' name="{name}" type="{ptype}"'
                if val:
                    attrs += f' val="{val}"'
                if units:
                    attrs += f' units="{units}"'

                # Insert attributes after the tag name
                # Handle: <Tag>value</Tag>  and  <Tag>value</Tag> <!-- comment -->
                line = re.sub(
                    rf'<{re.escape(tag)}(?=[\s>])',
                    f'<{tag}{attrs}',
                    line,
                    count=1
                )

            # Check if this is an opening-only tag (no </Tag> on same line)
            if f'</{tag}>' not in stripped and not stripped.endswith('/>'):
                stack.append(tag)
            # If it's <Tag>val</Tag> on one line, don't push (it's self-contained)

        output.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output)

    # Verify we found everything
    found_paths = set()
    for line in output:
        for m in re.finditer(r'name="(PARAM_\w+)"', line):
            found_paths.add(m.group(1))

    expected = set(p[0] for p in params)
    missing = expected - found_paths
    if missing:
        print(f"WARNING: {len(missing)} parameters not annotated:")
        for name in sorted(missing):
            dotpath = [p[2] for p in params if p[0] == name][0]
            print(f"  {name} -> {dotpath}")

    print(f"Annotated {len(found_paths)}/{len(params)} parameters")
    print(f"Wrote to {output_path}")


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <gpu_param.h> <gpu_param.cu> <input.xml> <output.xml>")
        sys.exit(1)

    h_path, cu_path, xml_in, xml_out = sys.argv[1:5]

    enums = parse_enums(h_path)
    print(f"Parsed enums: float={len(enums['float'])}, int={len(enums['int'])}, bool={len(enums['bool'])}")

    descriptions = parse_descriptions(cu_path)
    total_enum = sum(len(v) for v in enums.values())
    print(f"Parsed {len(descriptions)} description entries (expected {total_enum})")

    # Build flat param list
    params = []
    idx = 0
    for ptype in ['float', 'int', 'bool']:
        for name in enums[ptype]:
            if idx >= len(descriptions):
                print(f"ERROR: ran out of descriptions at {name}", file=sys.stderr)
                break
            xml_path, units, val = descriptions[idx]
            params.append((name, ptype, xml_path, units, val))
            idx += 1

    annotate_xml_text(xml_in, params, xml_out)


if __name__ == '__main__':
    main()
