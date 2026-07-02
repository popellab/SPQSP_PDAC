#!/usr/bin/env python3
"""
Query Logfire for LLM extraction metrics.

Usage:
    python scripts/query_logfire.py target.yaml
    python scripts/query_logfire.py <trace_id>

Library usage:
    from query_logfire import LogfireClient

    client = LogfireClient()
    metrics = client.get_trace_metrics("0af7651916cd43dd...")
    print(metrics.summary())
"""

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from logfire.query_client import LogfireQueryClient

load_dotenv(find_dotenv())


# Exception type to category mapping (mirrors qsp_llm_workflows.core.calibration.exceptions)
EXCEPTION_CATEGORIES = {
    # Hallucination detection
    "SnippetValueMismatchError": "hallucination",
    # Fabrication detection
    "DOIResolutionError": "fabrication",
    "DOIMetadataMismatchError": "fabrication",
    "PaperTitleMismatchError": "fabrication",
    # Reference consistency
    "InputReferenceError": "reference",
    "SourceRefError": "reference",
    "ParameterReferenceError": "reference",
    "StateVariableReferenceError": "reference",
    "SpeciesNotFoundError": "reference",
    "CalibrationReferenceError": "reference",
    # Code validation
    "CodeSyntaxError": "code",
    "CodeStructureError": "code",
    "CodeExecutionError": "code",
    "CodeSignatureError": "code",
    "HardcodedConstantError": "code",
    "ReturnValueError": "code",
    "ScalarReturnError": "code",
    "ArrayLengthError": "code",
    "ReturnStructureError": "code",
    # Unit validation
    "UnitParsingError": "units",
    "UnitValidationError": "units",
    "MissingUnitsError": "units",
    "DimensionalityMismatchError": "units",
    "UnitConversionError": "units",
    # Structural validation
    "SpanOrderingError": "structural",
    "MissingFieldError": "structural",
    "ObservableConfigError": "structural",
    "EvaluationPointsError": "structural",
    "SampleSizeError": "structural",
    "DataConsistencyError": "structural",
    "ComputedValueMismatchError": "structural",
    "ScaleMismatchError": "structural",
    "ContentValidationError": "structural",
    "EmptyScenarioError": "structural",
    # Prior validation
    "PriorParameterError": "prior",
    "PriorScaleError": "prior",
    # Source quality
    "SourceQualityError": "source_quality",
    "TranslationUncertaintyError": "translation",
    # Encoding
    "ControlCharacterError": "encoding",
}


def categorize_exception_type(exception_type: str) -> str:
    """Get category from exception type name."""
    for exc_name, category in EXCEPTION_CATEGORIES.items():
        if exc_name in exception_type:
            return category
    return "other"


@dataclass
class TraceMetrics:
    """Metrics extracted from a single Logfire trace."""

    trace_id: str
    timestamp: str
    chat_count: int
    duration: float
    input_tokens: int
    output_tokens: int
    cost: float
    validation_errors: list[str] = field(default_factory=list)
    exception_types: list[str] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)  # Tool name -> call count
    spans: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def retries(self) -> int:
        return max(0, self.chat_count - 1)

    @property
    def first_attempt_success(self) -> bool:
        return self.chat_count == 1

    def categorize_errors(self) -> Counter:
        """Categorize validation errors by type.

        Prefers exception_type from Logfire attributes when available,
        falls back to string parsing for backward compatibility.
        """
        categories = Counter()

        # First, categorize by exception type (most reliable)
        for exc_type in self.exception_types:
            category = categorize_exception_type(exc_type)
            categories[category] += 1

        # If no exception types captured, fall back to string parsing
        if not self.exception_types:
            for err in self.validation_errors:
                err_lower = err.lower()
                if "snippetvaluemismatch" in err_lower or ("snippet" in err_lower and "not found" in err_lower):
                    categories["hallucination"] += 1
                elif "doiresolution" in err_lower or ("doi" in err_lower and "resolve" in err_lower):
                    categories["fabrication"] += 1
                elif "doimetadatamismatch" in err_lower:
                    categories["fabrication"] += 1
                elif "pint" in err_lower or ("unit" in err_lower and "valid" in err_lower):
                    categories["units"] += 1
                elif "sourceref" in err_lower or "source_ref" in err_lower or "source tag" in err_lower:
                    categories["reference"] += 1
                elif "inputreference" in err_lower or ("input" in err_lower and "reference" in err_lower):
                    categories["reference"] += 1
                elif "syntax" in err_lower:
                    categories["code"] += 1
                elif "span" in err_lower and "ordering" in err_lower:
                    categories["structural"] += 1
                elif "field required" in err_lower or '"type": "missing"' in err_lower:
                    categories["structural"] += 1
                else:
                    categories["other"] += 1

        return categories

    def logfire_url(self, org: str = "popel-lab", project: str = "qsp-llm-workflows") -> str:
        """Generate Logfire UI URL for this trace."""
        return f"https://logfire-us.pydantic.dev/{org}/{project}/traces/{self.trace_id}"

    def summary(self, include_url: bool = True) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Trace ID: {self.trace_id}",
            f"Timestamp: {self.timestamp}",
            f"LLM calls: {self.chat_count} ({self.retries} retries)",
            f"Duration: {self.duration:.1f}s",
            f"Tokens: {self.total_tokens:,} (in: {self.input_tokens:,}, out: {self.output_tokens:,})",
            f"Cost: ${self.cost:.4f}",
        ]

        if include_url:
            lines.append(f"Logfire UI: {self.logfire_url()}")

        error_cats = self.categorize_errors()
        if error_cats:
            lines.append(f"\nValidation errors caught: {sum(error_cats.values())}")
            for cat, count in error_cats.most_common():
                lines.append(f"  {cat}: {count}")

        if self.exception_types:
            lines.append(f"\nException types: {', '.join(self.exception_types)}")

        if self.tool_calls:
            lines.append(f"\nTool usage:")
            for tool, count in sorted(self.tool_calls.items()):
                lines.append(f"  {tool}: {count}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "chat_count": self.chat_count,
            "retries": self.retries,
            "duration": self.duration,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "first_attempt_success": self.first_attempt_success,
            "validation_error_count": len(self.validation_errors),
            "exception_types": self.exception_types,
            "validation_error_categories": dict(self.categorize_errors()),
            "tool_calls": self.tool_calls,
        }


class LogfireClient:
    """Client for querying Logfire extraction metrics."""

    def __init__(
        self,
        read_token: Optional[str] = None,
        base_url: str = "https://logfire-us.pydantic.dev",
    ):
        self.read_token = read_token or os.environ.get("LOGFIRE_READ_TOKEN")
        self.client = LogfireQueryClient(read_token=self.read_token, base_url=base_url)

    def get_trace_metrics(self, trace_id: str) -> TraceMetrics:
        """Get metrics for a single trace by ID."""
        # Query chat spans for token/cost metrics
        chat_query = f"""
        SELECT trace_id, span_name, start_timestamp, duration, attributes
        FROM records
        WHERE trace_id = '{trace_id}'
          AND span_name LIKE 'chat%'
        ORDER BY start_timestamp ASC
        """

        result = self.client.query_json_rows(chat_query)
        rows = result["rows"]

        if not rows:
            raise ValueError(f"No spans found for trace_id: {trace_id}")

        spans = sorted(rows, key=lambda x: x["start_timestamp"])

        input_tokens = 0
        output_tokens = 0
        total_cost = 0.0
        total_duration = 0.0
        validation_errors = []
        exception_types = []

        for span in spans:
            attrs = span.get("attributes", {})
            input_tokens += attrs.get("gen_ai.usage.input_tokens", 0)
            output_tokens += attrs.get("gen_ai.usage.output_tokens", 0)
            total_cost += attrs.get("operation.cost", 0)
            total_duration += span.get("duration", 0) or 0

            # Extract validation errors from retry messages
            for msg in attrs.get("gen_ai.input.messages", []):
                if isinstance(msg, dict):
                    for part in msg.get("parts", []):
                        if isinstance(part, dict):
                            content = str(part.get("content", ""))
                            if "validation error" in content.lower():
                                if '"type":' in content or "Value error" in content:
                                    validation_errors.append(content)

        # Query all spans to find exception types
        exception_query = f"""
        SELECT attributes
        FROM records
        WHERE trace_id = '{trace_id}'
          AND attributes['exception.type'] IS NOT NULL
        """

        try:
            exc_result = self.client.query_json_rows(exception_query)
            for row in exc_result.get("rows", []):
                attrs = row.get("attributes", {})
                exc_type = attrs.get("exception.type", "")
                if exc_type and exc_type not in exception_types:
                    exception_types.append(exc_type)
        except Exception:
            # Exception query failed, continue without exception types
            pass

        # Query for tool calls and validation errors from pydantic_ai.all_messages
        tool_calls: dict[str, int] = {}
        agent_query = f"""
        SELECT attributes
        FROM records
        WHERE trace_id = '{trace_id}'
          AND span_name = 'agent run'
        """

        try:
            agent_result = self.client.query_json_rows(agent_query)
            for row in agent_result.get("rows", []):
                attrs = row.get("attributes", {})
                all_messages = attrs.get("pydantic_ai.all_messages", [])

                for msg in all_messages:
                    if not isinstance(msg, dict):
                        continue
                    parts = msg.get("parts", [])
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type", "")

                        # Count tool calls
                        if part_type == "tool_call":
                            tool_name = part.get("name", "unknown")
                            # Skip final_result as it's the output tool
                            if tool_name != "final_result":
                                tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

                        # Extract validation errors from final_result responses
                        if part_type == "tool_call_response" and part.get("name") == "final_result":
                            result_str = str(part.get("result", ""))
                            if "validation error" in result_str.lower():
                                validation_errors.append(result_str)
                                # Try to extract exception type from error message
                                for exc_name in EXCEPTION_CATEGORIES.keys():
                                    if exc_name.lower() in result_str.lower():
                                        if exc_name not in exception_types:
                                            exception_types.append(exc_name)
                                # Also check for common error patterns
                                if "prior predictive" in result_str.lower():
                                    if "PriorScaleError" not in exception_types:
                                        exception_types.append("PriorScaleError")
                                if "execution failed" in result_str.lower():
                                    if "CodeExecutionError" not in exception_types:
                                        exception_types.append("CodeExecutionError")
                                if "snippet" in result_str.lower() and "not found" in result_str.lower():
                                    if "SnippetValueMismatchError" not in exception_types:
                                        exception_types.append("SnippetValueMismatchError")
                                if "doi" in result_str.lower() and "resolve" in result_str.lower():
                                    if "DOIResolutionError" not in exception_types:
                                        exception_types.append("DOIResolutionError")
        except Exception:
            # Agent query failed, continue without tool calls
            pass

        return TraceMetrics(
            trace_id=trace_id,
            timestamp=spans[0]["start_timestamp"],
            chat_count=len(spans),
            duration=total_duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost,
            validation_errors=validation_errors,
            exception_types=exception_types,
            tool_calls=tool_calls,
            spans=spans,
        )

    def get_trace_metrics_from_yaml(self, yaml_path: Path) -> TraceMetrics:
        """Get metrics for a trace referenced in a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        trace_id = data.get("logfire_trace_id")
        if not trace_id:
            raise ValueError(f"No logfire_trace_id found in {yaml_path}")

        return self.get_trace_metrics(trace_id)


def main():
    if len(sys.argv) < 2:
        print("Usage: query_logfire.py <target.yaml | trace_id>")
        sys.exit(1)

    arg = sys.argv[1]
    client = LogfireClient()

    try:
        if arg.endswith((".yaml", ".yml")):
            metrics = client.get_trace_metrics_from_yaml(Path(arg))
            print(f"=== {Path(arg).name} ===\n")
        else:
            metrics = client.get_trace_metrics(arg)
            print(f"=== {arg[:12]}... ===\n")

        print(metrics.summary())

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
