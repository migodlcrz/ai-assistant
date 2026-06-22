from __future__ import annotations

from dataclasses import dataclass, field

from .call_graph import CallGraphStore

MAX_DEPTH = 8

# Prefixes that indicate external / stdlib symbols to skip
_EXTERNAL_PREFIXES = (
    "os.", "sys.", "re.", "json.", "math.", "time.", "datetime.",
    "pathlib.", "typing.", "collections.", "itertools.", "functools.",
    "logging.", "traceback.", "hashlib.", "uuid.", "io.", "abc.",
    "console.", "print", "len", "range", "str", "int", "float",
    "list", "dict", "set", "tuple", "bool", "type", "super",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "open", "next", "iter",
)

_EXTERNAL_EXACT = frozenset({
    "print", "len", "range", "str", "int", "float", "list",
    "dict", "set", "tuple", "bool", "type", "super",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "open", "next", "iter",
})


@dataclass
class TraceNode:
    symbol: str
    children: list[TraceNode] = field(default_factory=list)
    is_cycle: bool = False   # True when this node was already visited above
    file_path: str | None = None
    line: int | None = None


def _is_external(symbol: str) -> bool:
    if symbol in _EXTERNAL_EXACT:
        return True
    return any(symbol.startswith(p) for p in _EXTERNAL_PREFIXES)


class GraphTraverser:
    """Depth-first traversal of the call graph from a given entry point."""

    def __init__(self, call_graph: CallGraphStore, max_depth: int = MAX_DEPTH):
        self._cg = call_graph
        self._max_depth = max_depth

    def traverse(self, entry: str) -> TraceNode:
        """
        Build a TraceNode tree rooted at `entry`.

        Cycle detection uses the current ancestor path (not a global visited set)
        so the same symbol can appear in different branches.
        """
        loc = self._cg.location_of(entry)
        root = TraceNode(
            symbol=entry,
            file_path=loc[0] if loc else None,
            line=loc[1] if loc else None,
        )
        self._dfs(root, depth=0, ancestors=set())
        return root

    def _dfs(self, node: TraceNode, depth: int, ancestors: set[str]) -> None:
        if depth >= self._max_depth:
            return

        callees = self._cg.callees_of(node.symbol)
        seen_in_level: set[str] = set()

        for callee in callees:
            if _is_external(callee):
                continue
            if callee in seen_in_level:
                continue
            seen_in_level.add(callee)

            loc = self._cg.location_of(callee)
            child = TraceNode(
                symbol=callee,
                file_path=loc[0] if loc else None,
                line=loc[1] if loc else None,
            )
            node.children.append(child)

            if callee in ancestors:
                child.is_cycle = True
                continue

            self._dfs(child, depth + 1, ancestors | {node.symbol})

    def collect_all_symbols(self, root: TraceNode) -> list[str]:
        """Flat list of all unique symbols in the tree (BFS order)."""
        result = []
        seen: set[str] = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.symbol not in seen:
                seen.add(node.symbol)
                result.append(node.symbol)
            if not node.is_cycle:
                queue.extend(node.children)
        return result
