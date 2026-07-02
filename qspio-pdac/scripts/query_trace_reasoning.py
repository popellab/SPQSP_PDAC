#!/usr/bin/env python3
"""
Query Logfire for the full reasoning trace of a specific extraction.

Extracts chat span input/output messages and tool call results to understand
why the agent chose a particular paper during extraction.

Usage:
    python scripts/query_trace_reasoning.py
"""

import json
import os
import sys
import textwrap
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv
from logfire.query_client import LogfireQueryClient

load_dotenv(find_dotenv())

TRACE_ID = "019c67a72f92cae9984f6043c8f044f8"


def get_client() -> LogfireQueryClient:
    token = os.environ.get("LOGFIRE_READ_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_READ_TOKEN not found in environment.")
        print("Make sure .env file exists and contains LOGFIRE_READ_TOKEN=...")
        sys.exit(1)
    return LogfireQueryClient(
        read_token=token,
        base_url="https://logfire-us.pydantic.dev",
    )


def query_all_spans(client: LogfireQueryClient, trace_id: str) -> list[dict]:
    """Get ALL spans for a trace, ordered by time."""
    query = f"""
    SELECT trace_id, span_id, parent_span_id, span_name, start_timestamp,
           duration, attributes, message, level
    FROM records
    WHERE trace_id = '{trace_id}'
    ORDER BY start_timestamp ASC
    """
    result = client.query_json_rows(query)
    return result["rows"]


def format_message_parts(parts: list[dict], max_content_len: int = 5000) -> str:
    """Format message parts for display."""
    lines = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "unknown")

        if part_type == "text":
            content = part.get("content", "")
            if len(content) > max_content_len:
                content = content[:max_content_len] + f"\n... [TRUNCATED, total {len(part.get('content', ''))} chars]"
            lines.append(f"    [TEXT] {content}")

        elif part_type == "tool_call":
            tool_name = part.get("name", "unknown")
            args = part.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            lines.append(f"    [TOOL_CALL] {tool_name}")
            if isinstance(args, dict):
                for k, v in args.items():
                    v_str = str(v)
                    if len(v_str) > 500:
                        v_str = v_str[:500] + "..."
                    lines.append(f"      {k}: {v_str}")
            else:
                lines.append(f"      args: {str(args)[:1000]}")

        elif part_type == "tool_call_response":
            tool_name = part.get("name", "unknown")
            result = part.get("result", part.get("content", ""))
            result_str = str(result)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + f"\n... [TRUNCATED, total {len(str(result))} chars]"
            lines.append(f"    [TOOL_RESULT] {tool_name}")
            lines.append(f"      {result_str}")

        elif part_type == "tool_use":
            # Alternative format used by some agents
            tool_name = part.get("name", part.get("tool_name", "unknown"))
            tool_input = part.get("input", part.get("args", {}))
            lines.append(f"    [TOOL_USE] {tool_name}")
            lines.append(f"      input: {str(tool_input)[:1000]}")

        elif part_type == "tool_result":
            content = part.get("content", "")
            lines.append(f"    [TOOL_RESULT]")
            result_str = str(content)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + "..."
            lines.append(f"      {result_str}")

        else:
            content = str(part)
            if len(content) > 1000:
                content = content[:1000] + "..."
            lines.append(f"    [{part_type.upper()}] {content}")

    return "\n".join(lines)


def print_chat_span_detail(span: dict, index: int):
    """Print detailed input/output for a chat span."""
    attrs = span.get("attributes", {})
    span_name = span.get("span_name", "unknown")
    timestamp = span.get("start_timestamp", "")

    print(f"\n{'='*80}")
    print(f"CHAT SPAN [{index}]: {span_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Input tokens: {attrs.get('gen_ai.usage.input_tokens', 'N/A')}")
    print(f"Output tokens: {attrs.get('gen_ai.usage.output_tokens', 'N/A')}")
    print(f"{'='*80}")

    # Input messages
    input_msgs = attrs.get("gen_ai.input.messages", [])
    if input_msgs:
        print(f"\n--- INPUT MESSAGES ({len(input_msgs)} messages) ---")
        for i, msg in enumerate(input_msgs):
            if not isinstance(msg, dict):
                print(f"  [{i}] (non-dict): {str(msg)[:500]}")
                continue

            role = msg.get("role", "unknown")
            parts = msg.get("parts", [])
            print(f"\n  [{i}] Role: {role} ({len(parts)} parts)")

            if parts:
                print(format_message_parts(parts))
    else:
        print("\n--- INPUT MESSAGES: NONE ---")

    # Output message
    output_msg = attrs.get("gen_ai.output.message", {})
    if output_msg:
        print(f"\n--- OUTPUT MESSAGE ---")
        if isinstance(output_msg, dict):
            role = output_msg.get("role", "unknown")
            parts = output_msg.get("parts", [])
            print(f"  Role: {role} ({len(parts)} parts)")
            if parts:
                print(format_message_parts(parts))
        else:
            print(f"  {str(output_msg)[:3000]}")
    else:
        print("\n--- OUTPUT MESSAGE: NONE ---")


def print_tool_spans(spans: list[dict]):
    """Print tool-related spans (web_search, etc.)."""
    tool_spans = [
        s for s in spans
        if any(kw in s.get("span_name", "").lower()
               for kw in ["tool", "web_search", "search", "fetch", "http"])
        and "chat" not in s.get("span_name", "").lower()
    ]

    if tool_spans:
        print(f"\n{'='*80}")
        print(f"TOOL SPANS ({len(tool_spans)} total)")
        print(f"{'='*80}")
        for i, span in enumerate(tool_spans):
            attrs = span.get("attributes", {})
            print(f"\n  [{i}] {span['span_name']} @ {span.get('start_timestamp', '')}")
            print(f"      Message: {span.get('message', 'N/A')}")
            # Print relevant attributes (skip huge ones)
            for key, val in sorted(attrs.items()):
                val_str = str(val)
                if len(val_str) > 500:
                    val_str = val_str[:500] + "..."
                print(f"      {key}: {val_str}")


def print_agent_messages(spans: list[dict]):
    """Print pydantic_ai.all_messages from agent run spans."""
    agent_spans = [s for s in spans if s.get("span_name") == "agent run"]

    for span in agent_spans:
        attrs = span.get("attributes", {})
        all_messages = attrs.get("pydantic_ai.all_messages", [])
        if not all_messages:
            continue

        print(f"\n{'='*80}")
        print(f"AGENT RUN MESSAGES (pydantic_ai.all_messages)")
        print(f"Total messages: {len(all_messages)}")
        print(f"{'='*80}")

        for i, msg in enumerate(all_messages):
            if not isinstance(msg, dict):
                continue
            kind = msg.get("kind", msg.get("type", "unknown"))
            parts = msg.get("parts", [])

            # Summarize each message
            print(f"\n--- Message [{i}] kind={kind}, {len(parts)} parts ---")

            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", part.get("part_kind", "unknown"))

                if part_type in ("text", "system-prompt", "user-prompt"):
                    content = part.get("content", "")
                    if len(content) > 2000:
                        content = content[:2000] + f"\n... [TRUNCATED from {len(part.get('content', ''))} chars]"
                    print(f"  [{part_type}] {content}")

                elif part_type == "tool_call":
                    tool_name = part.get("name", part.get("tool_name", "unknown"))
                    args = part.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    print(f"  [TOOL_CALL] {tool_name}")
                    args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                    if len(args_str) > 1000:
                        args_str = args_str[:1000] + "..."
                    print(f"    args: {args_str}")

                elif part_type in ("tool_call_response", "tool-return", "tool-result"):
                    tool_name = part.get("name", part.get("tool_name", ""))
                    result = part.get("result", part.get("content", ""))
                    result_str = str(result)
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + f"\n... [TRUNCATED from {len(str(result))} chars]"
                    print(f"  [TOOL_RESULT] {tool_name}")
                    print(f"    {result_str}")

                else:
                    content = str(part)
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    print(f"  [{part_type}] {content}")


def main():
    print(f"Querying Logfire for trace: {TRACE_ID}")
    print(f"Looking for: why extraction chose Li et al. 2022 instead of Zheng et al. 2022\n")

    client = get_client()

    # Get all spans
    spans = query_all_spans(client, TRACE_ID)
    print(f"Total spans found: {len(spans)}")

    # Summarize span types
    span_names = [s.get("span_name", "unknown") for s in spans]
    from collections import Counter
    name_counts = Counter(span_names)
    print(f"\nSpan type summary:")
    for name, count in name_counts.most_common():
        print(f"  {name}: {count}")

    # Find chat spans
    chat_spans = [s for s in spans if "chat" in s.get("span_name", "").lower()]
    print(f"\nChat spans: {len(chat_spans)}")

    # Print FIRST chat span in full detail
    if chat_spans:
        print(f"\n{'#'*80}")
        print(f"# FIRST CHAT SPAN (initial prompt + response)")
        print(f"{'#'*80}")
        print_chat_span_detail(chat_spans[0], 0)

        # Print subsequent chat spans more briefly (just tool calls and key content)
        if len(chat_spans) > 1:
            print(f"\n{'#'*80}")
            print(f"# SUBSEQUENT CHAT SPANS (retries / continuations)")
            print(f"{'#'*80}")
            for i, span in enumerate(chat_spans[1:], 1):
                print_chat_span_detail(span, i)

    # Print tool-related spans
    print_tool_spans(spans)

    # Print agent messages (contains full conversation including tool results)
    print_agent_messages(spans)

    # Search for DOI mentions across all spans
    print(f"\n{'='*80}")
    print(f"DOI MENTIONS ACROSS ALL SPANS")
    print(f"{'='*80}")
    doi_pattern_zheng = "10.1016/j.ccell.2022.10.024"
    doi_pattern_li = "10.1016/j.ccell.2022.10.001"
    for span in spans:
        span_str = json.dumps(span, default=str)
        has_zheng = doi_pattern_zheng in span_str
        has_li = doi_pattern_li in span_str
        if has_zheng or has_li:
            dois = []
            if has_zheng:
                dois.append("Zheng2022")
            if has_li:
                dois.append("Li2022")
            print(f"  Span '{span.get('span_name', '')}' @ {span.get('start_timestamp', '')}: {', '.join(dois)}")


if __name__ == "__main__":
    main()
