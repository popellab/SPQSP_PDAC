#!/usr/bin/env python3
"""
Extract original LLM outputs from Logfire traces for CalibrationTarget files.

Queries the `agent run` span's `final_result` attribute, which contains
the structured CalibrationTarget JSON as returned by the LLM before curation.

Usage:
    # Extract originals for all gpt-5.1 files with trace IDs
    python scripts/extract_logfire_originals.py calibration_targets/

    # Extract original for a single file
    python scripts/extract_logfire_originals.py calibration_targets/baseline_no_treatment/m1_m2_ratio_PDAC_deriv001.yaml

    # Extract by trace ID directly
    python scripts/extract_logfire_originals.py --trace-id 019c6764f022bda0c4d9c5bf3116ffd1

    # Write originals to a directory
    python scripts/extract_logfire_originals.py calibration_targets/ --output-dir originals/

    # Also compute diff stats
    python scripts/extract_logfire_originals.py calibration_targets/ --output-dir originals/ --diff
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from logfire.query_client import LogfireQueryClient

RATE_LIMIT_DELAY = 5  # seconds between Logfire API calls

load_dotenv(find_dotenv())


class LogfireOriginalExtractor:
    """Extract original LLM outputs from Logfire traces."""

    def __init__(
        self,
        read_token: Optional[str] = None,
        base_url: str = "https://logfire-us.pydantic.dev",
    ):
        self.read_token = read_token or os.environ.get("LOGFIRE_READ_TOKEN")
        self.client = LogfireQueryClient(
            read_token=self.read_token, base_url=base_url
        )

    def _query_with_retry(self, sql: str, max_retries: int = 3) -> dict:
        """Execute a Logfire query with rate-limit retry."""
        for attempt in range(max_retries):
            try:
                result = self.client.query_json_rows(sql)
                time.sleep(RATE_LIMIT_DELAY)
                return result
            except Exception as e:
                if "Rate limit" in str(e) or "429" in str(e):
                    wait = RATE_LIMIT_DELAY * (attempt + 2)
                    print(f"(rate limited, waiting {wait}s)", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")

    def get_original_json(self, trace_id: str) -> dict:
        """Get the original LLM output (final_result) for a trace.

        Returns the parsed JSON dict from the agent run span's final_result attribute.
        """
        result = self._query_with_retry(f"""
            SELECT attributes
            FROM records
            WHERE trace_id = '{trace_id}'
              AND span_name = 'agent run'
            LIMIT 1
        """)

        rows = result.get("rows", [])
        if not rows:
            raise ValueError(f"No agent run span found for trace_id: {trace_id}")

        attrs = rows[0].get("attributes", {})
        final_result = attrs.get("final_result")

        if not final_result:
            raise ValueError(
                f"No final_result attribute in agent run span for trace_id: {trace_id}"
            )

        if isinstance(final_result, str):
            return json.loads(final_result)
        return final_result

    def get_original_yaml(self, trace_id: str) -> str:
        """Get the original LLM output as YAML string."""
        data = self.get_original_json(trace_id)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, width=120)

    def get_trace_metadata(self, trace_id: str) -> dict:
        """Get extraction metadata (model, tokens, cost, duration)."""
        result = self._query_with_retry(f"""
            SELECT span_name, attributes, duration
            FROM records
            WHERE trace_id = '{trace_id}'
            ORDER BY start_timestamp ASC
        """)

        metadata = {
            "trace_id": trace_id,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "duration_s": 0.0,
            "chat_count": 0,
        }

        for row in result.get("rows", []):
            attrs = row.get("attributes", {})
            span_name = row.get("span_name", "")

            if span_name == "agent run":
                metadata["model"] = attrs.get("model_name")
                metadata["input_tokens"] = attrs.get(
                    "gen_ai.usage.input_tokens", 0
                )
                metadata["output_tokens"] = attrs.get(
                    "gen_ai.usage.output_tokens", 0
                )

            if "chat" in span_name:
                metadata["chat_count"] += 1
                metadata["cost"] += attrs.get("operation.cost", 0)
                metadata["duration_s"] += (row.get("duration", 0) or 0) / 1e9

        return metadata


def find_yaml_files_with_traces(directory: Path) -> list[tuple[Path, str]]:
    """Find all YAML files with populated logfire_trace_id fields."""
    results = []
    for yaml_path in sorted(directory.rglob("*.yaml")):
        if yaml_path.parent.name == "sources":
            continue
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        trace_id = data.get("logfire_trace_id")
        if trace_id:
            results.append((yaml_path, str(trace_id)))
    return results


def compute_diff_stats(original_yaml: str, curated_path: Path) -> dict:
    """Compute diff statistics between original and curated YAML."""
    with open(curated_path) as f:
        curated_yaml = f.read()

    orig_lines = original_yaml.strip().splitlines()
    cur_lines = curated_yaml.strip().splitlines()

    # Simple line-based diff stats
    orig_set = set(orig_lines)
    cur_set = set(cur_lines)

    added = len(cur_set - orig_set)
    removed = len(orig_set - cur_set)
    common = len(orig_set & cur_set)

    return {
        "orig_lines": len(orig_lines),
        "cur_lines": len(cur_lines),
        "added": added,
        "removed": removed,
        "common": common,
        "pct_changed": round(
            100 * (added + removed) / max(len(orig_lines) + len(cur_lines), 1), 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract original LLM outputs from Logfire traces"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="YAML file or directory to process",
    )
    parser.add_argument(
        "--trace-id",
        help="Query a specific trace ID (ignores path for finding trace)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write original YAMLs to this directory",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Compute diff stats between original and curated",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write structured results to JSON file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an original in --output-dir",
    )
    args = parser.parse_args()

    extractor = LogfireOriginalExtractor()
    results = []

    if args.trace_id:
        # Single trace ID mode
        files = [(args.path, args.trace_id)]
    elif args.path.is_dir():
        files = find_yaml_files_with_traces(args.path)
        print(f"Found {len(files)} files with Logfire trace IDs\n")
    elif args.path.is_file():
        with open(args.path) as f:
            data = yaml.safe_load(f)
        trace_id = data.get("logfire_trace_id")
        if not trace_id:
            print(f"No logfire_trace_id in {args.path}")
            sys.exit(1)
        files = [(args.path, str(trace_id))]
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for yaml_path, trace_id in files:
        name = yaml_path.stem

        # Skip if already extracted
        if args.skip_existing and args.output_dir:
            out_path = args.output_dir / f"{name}_original.yaml"
            if out_path.exists():
                print(f"Skipping {name} (already extracted)")
                # Still compute diff for summary
                if args.diff:
                    with open(out_path) as f:
                        original_yaml = f.read()
                    diff = compute_diff_stats(original_yaml, yaml_path)
                    results.append({
                        "file": str(yaml_path),
                        "name": name,
                        "trace_id": trace_id,
                        "orig_lines": len(original_yaml.strip().splitlines()),
                        "status": "ok",
                        "diff": diff,
                        "output_path": str(out_path),
                    })
                continue

        print(f"Processing {name}...", end=" ", flush=True)

        try:
            original_json = extractor.get_original_json(trace_id)
            original_yaml = yaml.dump(
                original_json,
                default_flow_style=False,
                allow_unicode=True,
                width=120,
            )

            result = {
                "file": str(yaml_path),
                "name": name,
                "trace_id": trace_id,
                "orig_lines": len(original_yaml.strip().splitlines()),
                "status": "ok",
            }

            # Get metadata
            try:
                metadata = extractor.get_trace_metadata(trace_id)
                result["metadata"] = metadata
            except Exception:
                pass

            # Write original to output dir
            if args.output_dir:
                out_path = args.output_dir / f"{name}_original.yaml"
                with open(out_path, "w") as f:
                    f.write(original_yaml)
                result["output_path"] = str(out_path)

            # Compute diff
            if args.diff:
                diff = compute_diff_stats(original_yaml, yaml_path)
                result["diff"] = diff
                print(
                    f"OK ({diff['orig_lines']} -> {diff['cur_lines']} lines, "
                    f"{diff['pct_changed']}% changed)"
                )
            else:
                print(f"OK ({result['orig_lines']} lines)")

            results.append(result)

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "file": str(yaml_path),
                "name": name,
                "trace_id": trace_id,
                "status": "error",
                "error": str(e),
            })

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*60}")
    print(f"Extracted {ok_count}/{len(results)} originals")

    if args.diff:
        ok_results = [r for r in results if r.get("diff")]
        if ok_results:
            avg_pct = sum(r["diff"]["pct_changed"] for r in ok_results) / len(
                ok_results
            )
            print(f"Average change: {avg_pct:.1f}%")

            # Print table
            print(f"\n{'File':<55} {'Orig':>5} {'Cur':>5} {'%Chg':>6}")
            print("-" * 75)
            for r in sorted(ok_results, key=lambda x: x["name"]):
                d = r["diff"]
                print(
                    f"{r['name']:<55} {d['orig_lines']:>5} "
                    f"{d['cur_lines']:>5} {d['pct_changed']:>5.1f}%"
                )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON results written to {args.json_output}")


if __name__ == "__main__":
    main()