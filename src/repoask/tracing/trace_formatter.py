from __future__ import annotations

import sys

from .graph_traverser import TraceNode
from .side_effect_detector import SideEffect


def _supports_unicode() -> bool:
    try:
        return sys.stdout.encoding.lower().startswith("utf")
    except Exception:
        return False


def _node_label(node: TraceNode) -> str:
    """Format a node as  file:line :: symbol()  when location is available."""
    sym = f"{node.symbol}()"
    if node.file_path and node.line:
        return f"{node.file_path}:{node.line} :: {sym}"
    return sym


def format_trace(
    query: str,
    entry: str,
    root: TraceNode,
    side_effects: list[SideEffect],
    rich: bool = True,
) -> str:
    """
    Render the full trace output as a string.

    `rich=True`  → tree characters (├──, └──, │)
    `rich=False` → plain indented text
    """
    lines = []
    lines.append(f"TRACE: {query}")
    lines.append("")
    lines.append("Entry Point:")
    entry_label = f"{root.file_path}:{root.line} :: {entry}()" if root.file_path and root.line else f"{entry}()"
    lines.append(f"  {entry_label}")
    lines.append("")
    lines.append("Execution Flow:")

    if rich and _supports_unicode():
        flow_lines = _render_rich_tree(root, prefix="", is_last=True)
    else:
        flow_lines = _render_plain_tree(root, indent=0)

    lines.extend(flow_lines)
    lines.append("")
    lines.append("Side Effects:")

    if side_effects:
        for effect in side_effects:
            lines.append(f"  - {effect.kind}: {effect.detail}")
    else:
        lines.append("  (none detected)")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Rich tree rendering
# ------------------------------------------------------------------ #

def _render_rich_tree(
    node: TraceNode,
    prefix: str,
    is_last: bool,
) -> list[str]:
    connector = "└── " if is_last else "├── "
    suffix = " (cycle)" if node.is_cycle else ""
    base = _node_label(node)
    label = f"{prefix}{connector}{base}{suffix}" if prefix else f"  {base}{suffix}"

    lines = [label]

    if node.is_cycle:
        return lines

    extension = "    " if is_last else "│   "
    child_prefix = (prefix + extension) if prefix else "  "

    for i, child in enumerate(node.children):
        child_is_last = i == len(node.children) - 1
        lines.extend(_render_rich_tree(child, child_prefix, child_is_last))

    return lines


# ------------------------------------------------------------------ #
# Plain text tree rendering
# ------------------------------------------------------------------ #

def _render_plain_tree(node: TraceNode, indent: int) -> list[str]:
    pad = "  " * indent
    arrow = "→ " if indent > 0 else "  "
    suffix = " (cycle)" if node.is_cycle else ""
    lines = [f"{pad}{arrow}{_node_label(node)}{suffix}"]

    if node.is_cycle:
        return lines

    for child in node.children:
        lines.extend(_render_plain_tree(child, indent + 1))

    return lines
