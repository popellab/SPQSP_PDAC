#!/usr/bin/env python3
"""
Export all Mermaid diagrams to PNG using Mermaid Live API.
Creates PNG versions in png_exports/ subdirectory.
"""

import os
import json
import base64
import subprocess
from pathlib import Path

# Find all .mmd files
diagrams_dir = Path(__file__).parent
mmd_files = sorted(diagrams_dir.glob("*.mmd"))

print(f"Found {len(mmd_files)} Mermaid diagrams to export\n")

# Create output directory
output_dir = diagrams_dir / "png_exports"
output_dir.mkdir(exist_ok=True)

# Method 1: Try using curl with Mermaid Live's render endpoint
for mmd_file in mmd_files:
    png_file = output_dir / mmd_file.with_suffix(".png").name

    with open(mmd_file, 'r') as f:
        content = f.read()

    # Encode diagram in URL format for Mermaid Live
    # Using the kroki.io service as alternative (if available)
    try:
        # Try kroki.io (open-source diagram service)
        encoded = base64.b64encode(content.encode()).decode()
        url = f"https://kroki.io/mermaid/png/{encoded}"

        result = subprocess.run(
            ['curl', '-s', '-o', str(png_file), url],
            capture_output=True,
            timeout=10
        )

        if result.returncode == 0 and png_file.exists() and png_file.stat().st_size > 0:
            print(f"✓ {mmd_file.name} → {png_file.name}")
        else:
            print(f"✗ {mmd_file.name} — curl failed or returned empty")
    except Exception as e:
        print(f"✗ {mmd_file.name} — {str(e)}")

# Copy existing PNG files
png_files = list(diagrams_dir.glob("*.png"))
print(f"\nCopying {len(png_files)} existing PNG files:")

for png_file in png_files:
    dest = output_dir / png_file.name
    try:
        with open(png_file, 'rb') as src:
            with open(dest, 'wb') as dst:
                dst.write(src.read())
        print(f"✓ Copied {png_file.name}")
    except Exception as e:
        print(f"✗ {png_file.name} — {str(e)}")

# Summary
exported_pngs = list(output_dir.glob("*.png"))
print(f"\n{'='*50}")
print(f"Summary: {len(exported_pngs)} PNG files in png_exports/")
print(f"{'='*50}")

for png_file in sorted(exported_pngs):
    size_kb = png_file.stat().st_size / 1024
    print(f"  {png_file.name:<50} {size_kb:>6.1f} KB")

print(f"\nNote: If kroki.io is unavailable, use Mermaid Live Editor:")
print(f"  1. Visit https://mermaid.live")
print(f"  2. Open each .mmd file and export to PNG")
print(f"  3. Save to png_exports/")
