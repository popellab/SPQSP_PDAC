#!/usr/bin/env python3
"""Export submodel posterior medians to parameters/posterior_medians.json.

Skips the write when the on-disk file's ``_meta.source_hash`` already matches
the freshly-computed value. ``source_hash`` is a deterministic fingerprint of
the submodel cache content, so a match means nothing substantive changed and
the only diff would be the ``_meta.generated_at`` timestamp. Writing
unconditionally would produce a spurious file-modified diff on every run, so
we avoid unnecessary writes.
"""

import json
from pathlib import Path

from qsp_inference.audit.report import export_posterior_medians

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "calibration_targets" / "submodel_targets" / ".compare_cache"
OUTPUT = ROOT / "parameters" / "posterior_medians.json"

if __name__ == "__main__":
    if not CACHE_DIR.exists():
        print(f"No cache found at {CACHE_DIR}. Run parameter audit first.")
        raise SystemExit(1)

    data = export_posterior_medians(CACHE_DIR)
    new_source_hash = data["_meta"]["source_hash"]

    # Skip write if existing file already reflects the same source cache state.
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text())
            if existing.get("_meta", {}).get("source_hash") == new_source_hash:
                print(
                    f"posterior_medians.json already up to date "
                    f"(source_hash={new_source_hash}); skipping write."
                )
                raise SystemExit(0)
        except (json.JSONDecodeError, OSError):
            # Fall through to a fresh write if the existing file is unreadable.
            pass

    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {data['_meta']['n_params']} posterior medians to {OUTPUT.name}")
    print(f"  source_hash: {new_source_hash}")
