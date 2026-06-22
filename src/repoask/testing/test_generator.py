from __future__ import annotations

from dataclasses import dataclass, field

from ..review.reviewer import FileContext


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FileTestPlan:
    file_path: str
    feature_summary: str
    feature_flow: str
    unit_tests: str
    integration_tests: str
    flow_tests: str
    regression_risks: str
    side_effects_covered: str


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are a senior software engineer generating test cases for a code change.
You have full repository context — not just the diff.
Your goal is to reconstruct the exact feature behavior and produce meaningful tests
that a developer can run to validate the change before committing.

Prioritize Flow / Scenario Tests that simulate real user behavior end-to-end.
Be specific: name functions, describe inputs/outputs, reference actual method calls.
If a change touches side effects (email, queue, database write, API call), include tests for those explicitly.
"""

_USER_TEMPLATE = """\
## File changed
{file_path}

## Diff (what changed)
```
{diff}
```

## Symbols defined in this file
{symbols}

## Call graph — who calls this file (callers)
{callers}

## Call graph — what this file depends on (callees)
{callees}

## Related code from the repository (RAG context)
{rag_context}

---

Reconstruct the feature flow and generate test cases.
Respond with EXACTLY these section labels (no markdown, no bold, just plain text):

Feature Summary:
[One paragraph: what feature or behavior this file implements and what changed]

Feature Flow:
[Step-by-step execution path from user input to final output, e.g.:
Request received → validate input → call service → query DB → publish event → return response]

Unit Tests:
[List each as a dash + description. Function-level: input/output, edge cases, error handling]

Integration Tests:
[List each as a dash + description. Cross-module: API+service+DB, external dependencies]

Flow / Scenario Tests:
[List each as a dash + description. End-to-end: simulate real user behavior, full workflow validation]

Regression Risks:
[List each as a dash + description. What existing behavior could break due to this change]

Side Effects Covered:
[List each as a dash + description. Side effects being validated: DB writes, emails, events, API calls]
"""


def build_test_prompt(ctx: FileContext) -> str:
    symbols_text = "\n".join(
        f"  {s['type']} {s['name']} (lines {s['lines']})" for s in ctx.symbols
    ) or "  (none found in index)"

    callers_text = "\n".join(f"  {c}" for c in ctx.callers[:20]) or "  (none)"
    callees_text = "\n".join(f"  {c}" for c in ctx.callees[:20]) or "  (none)"

    rag_parts = []
    for hit in ctx.rag_chunks:
        meta = hit.get("metadata", {})
        fp = meta.get("file_path", "?")
        sym = meta.get("symbol_name", "?")
        lines = f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        snippet = hit.get("text", "")[:300]
        rag_parts.append(f"  [{fp} :: {sym} lines {lines}]\n  {snippet}")
    rag_text = "\n\n".join(rag_parts) or "  (none)"

    return _USER_TEMPLATE.format(
        file_path=ctx.file_path,
        diff=ctx.diff[:3000] if ctx.diff else "(no diff — new file or binary)",
        symbols=symbols_text,
        callers=callers_text,
        callees=callees_text,
        rag_context=rag_text,
    )


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #

def parse_test_response(file_path: str, response: str) -> FileTestPlan:
    sections: dict[str, list[str]] = {
        "feature_summary": [],
        "feature_flow": [],
        "unit_tests": [],
        "integration_tests": [],
        "flow_tests": [],
        "regression_risks": [],
        "side_effects_covered": [],
    }

    label_map = {
        "feature summary": "feature_summary",
        "feature flow": "feature_flow",
        "unit tests": "unit_tests",
        "unit test": "unit_tests",
        "integration tests": "integration_tests",
        "integration test": "integration_tests",
        "flow / scenario tests": "flow_tests",
        "flow/scenario tests": "flow_tests",
        "scenario tests": "flow_tests",
        "flow tests": "flow_tests",
        "regression risks": "regression_risks",
        "regression risk": "regression_risks",
        "side effects covered": "side_effects_covered",
        "side effect covered": "side_effects_covered",
    }

    current_key: str | None = None

    for line in response.splitlines():
        stripped = line.strip()
        lower = stripped.rstrip(":").lower()

        if lower in label_map:
            current_key = label_map[lower]
        elif current_key is not None and stripped:
            sections[current_key].append(stripped)

    def _join(key: str, fallback: str) -> str:
        return "\n".join(sections[key]).strip() or fallback

    return FileTestPlan(
        file_path=file_path,
        feature_summary=_join("feature_summary", "No summary available."),
        feature_flow=_join("feature_flow", "Could not reconstruct flow."),
        unit_tests=_join("unit_tests", "- No unit tests generated."),
        integration_tests=_join("integration_tests", "- No integration tests generated."),
        flow_tests=_join("flow_tests", "- No flow tests generated."),
        regression_risks=_join("regression_risks", "- No regression risks detected."),
        side_effects_covered=_join("side_effects_covered", "- No side effects identified."),
    )


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

DIVIDER = "─" * 44


def format_test_report(changed_files: list[str], plans: list[FileTestPlan]) -> str:
    lines = []
    lines.append("GENERATE TEST REPORT")
    lines.append("")
    lines.append("Changed Files:")
    for f in changed_files:
        lines.append(f"  - {f}")
    lines.append("")
    lines.append(DIVIDER)

    for plan in plans:
        lines.append("")
        lines.append(f"FILE: {plan.file_path}")
        lines.append("")
        lines.append("🧠 Feature Summary:")
        lines.append(plan.feature_summary)
        lines.append("")
        lines.append("🔁 Feature Flow:")
        lines.append(plan.feature_flow)
        lines.append("")
        lines.append("🧪 Unit Tests:")
        lines.append(plan.unit_tests)
        lines.append("")
        lines.append("🔗 Integration Tests:")
        lines.append(plan.integration_tests)
        lines.append("")
        lines.append("🎬 Flow / Scenario Tests:")
        lines.append(plan.flow_tests)
        lines.append("")
        lines.append("⚠️  Regression Risks:")
        lines.append(plan.regression_risks)
        lines.append("")
        lines.append("💥 Side Effects Covered:")
        lines.append(plan.side_effects_covered)
        lines.append("")
        lines.append(DIVIDER)

    return "\n".join(lines)
