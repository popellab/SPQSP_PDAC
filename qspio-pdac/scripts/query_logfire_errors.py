#!/usr/bin/env python3
"""
Query Logfire for detailed validation errors from failed extractions.

Usage:
    python scripts/query_logfire_errors.py <trace_id>
"""

import json
import os
import re
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from logfire.query_client import LogfireQueryClient

load_dotenv(find_dotenv())


class LogfireErrorClient:
    """Client for querying detailed errors from Logfire traces."""

    def __init__(
        self,
        read_token: Optional[str] = None,
        base_url: str = "https://logfire-us.pydantic.dev",
    ):
        self.read_token = read_token or os.environ.get("LOGFIRE_READ_TOKEN")
        self.client = LogfireQueryClient(read_token=self.read_token, base_url=base_url)

    def get_all_spans(self, trace_id: str) -> list[dict]:
        """Get all spans for a trace."""
        query = f"""
        SELECT trace_id, span_id, span_name, start_timestamp, duration,
               attributes, message, level
        FROM records
        WHERE trace_id = '{trace_id}'
        ORDER BY start_timestamp ASC
        """
        result = self.client.query_json_rows(query)
        return result["rows"]

    def get_validation_errors(self, trace_id: str) -> list[dict]:
        """Extract validation errors from a trace."""
        spans = self.get_all_spans(trace_id)
        errors = []

        for span in spans:
            attrs = span.get("attributes", {})
            message = span.get("message", "")
            span_name = span.get("span_name", "")

            # Check for validation errors in various places

            # 1. Check message field
            if "validation" in message.lower() or "error" in message.lower():
                errors.append({
                    "source": "message",
                    "span_name": span_name,
                    "content": message,
                })

            # 2. Check input messages for retry context
            input_msgs = attrs.get("gen_ai.input.messages", [])
            if isinstance(input_msgs, list):
                for msg in input_msgs:
                    if isinstance(msg, dict):
                        for part in msg.get("parts", []):
                            if isinstance(part, dict):
                                content = str(part.get("content", ""))
                                if "validation error" in content.lower():
                                    # Extract just the error part
                                    errors.append({
                                        "source": "retry_context",
                                        "span_name": span_name,
                                        "content": content[:2000],  # Truncate
                                    })

            # 3. Check for exception info
            if attrs.get("exception.type"):
                errors.append({
                    "source": "exception",
                    "span_name": span_name,
                    "type": attrs.get("exception.type"),
                    "message": attrs.get("exception.message", ""),
                })

            # 4. Check for pydantic validation details
            if "pydantic" in str(attrs).lower() and "error" in str(attrs).lower():
                errors.append({
                    "source": "pydantic",
                    "span_name": span_name,
                    "attrs": {k: v for k, v in attrs.items() if "error" in k.lower()},
                })

        return errors

    def summarize_trace(self, trace_id: str) -> dict:
        """Get a summary of all spans and errors in a trace."""
        spans = self.get_all_spans(trace_id)

        summary = {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "span_names": [],
            "chat_spans": [],
            "error_spans": [],
            "validation_errors": [],
        }

        for span in spans:
            span_name = span.get("span_name", "unknown")
            summary["span_names"].append(span_name)

            if "chat" in span_name.lower():
                attrs = span.get("attributes", {})
                chat_info = {
                    "span_name": span_name,
                    "timestamp": span.get("start_timestamp"),
                    "input_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
                    "output_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
                }

                # Look for validation errors in input messages (retry context)
                input_msgs = attrs.get("gen_ai.input.messages", [])
                for msg in input_msgs if isinstance(input_msgs, list) else []:
                    if isinstance(msg, dict):
                        for part in msg.get("parts", []):
                            if isinstance(part, dict):
                                content = str(part.get("content", ""))
                                if "validation error" in content.lower():
                                    # Extract error details
                                    chat_info["has_validation_context"] = True
                                    # Try to extract the actual errors
                                    error_match = re.search(
                                        r'(\d+) validation error[s]? for SubmodelTarget\n(.+?)(?=\n\n|\Z)',
                                        content,
                                        re.DOTALL
                                    )
                                    if error_match:
                                        chat_info["validation_preview"] = error_match.group(0)[:500]

                summary["chat_spans"].append(chat_info)

            level = span.get("level", "")
            if level in ("error", "warning"):
                summary["error_spans"].append({
                    "span_name": span_name,
                    "level": level,
                    "message": span.get("message", ""),
                })

        # Deduplicate span names
        summary["unique_span_names"] = list(set(summary["span_names"]))
        del summary["span_names"]

        return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: query_logfire_errors.py <trace_id>")
        sys.exit(1)

    trace_id = sys.argv[1]
    client = LogfireErrorClient()

    print(f"=== Trace: {trace_id[:16]}... ===\n")

    # Get summary
    summary = client.summarize_trace(trace_id)

    print(f"Total spans: {summary['total_spans']}")
    print(f"Chat spans: {len(summary['chat_spans'])}")
    print(f"Error spans: {len(summary['error_spans'])}")
    print(f"\nUnique span types: {len(summary['unique_span_names'])}")
    for name in sorted(summary['unique_span_names']):
        print(f"  - {name}")

    print(f"\n--- Chat Span Details ---")
    for i, chat in enumerate(summary['chat_spans'], 1):
        print(f"\n[{i}] {chat['span_name']}")
        print(f"    Tokens: {chat.get('input_tokens', 0):,} in / {chat.get('output_tokens', 0):,} out")
        if chat.get("has_validation_context"):
            print(f"    Has validation error context: YES")
            if chat.get("validation_preview"):
                print(f"    Preview: {chat['validation_preview'][:200]}...")

    if summary['error_spans']:
        print(f"\n--- Error Spans ---")
        for err in summary['error_spans']:
            print(f"  [{err['level']}] {err['span_name']}: {err['message'][:100]}")

    # Get detailed validation errors
    print(f"\n--- Validation Errors ---")
    errors = client.get_validation_errors(trace_id)
    if errors:
        for i, err in enumerate(errors[:10], 1):  # Limit to first 10
            print(f"\n[{i}] Source: {err['source']}")
            print(f"    Span: {err.get('span_name', 'N/A')}")
            if err.get('content'):
                # Print first 300 chars of content
                content = err['content'][:300]
                print(f"    Content: {content}...")
            if err.get('type'):
                print(f"    Type: {err['type']}")
                print(f"    Message: {err.get('message', '')[:200]}")
    else:
        print("  No validation errors captured in span attributes.")
        print("  Check Logfire UI for detailed message content.")


if __name__ == "__main__":
    main()
