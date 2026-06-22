from __future__ import annotations

import re

from .call_graph import CallGraphStore


class EntryPointResolver:
    """
    Resolves a query string to the most likely symbol in the call graph.

    Priority:
      1. Exact symbol match
      2. Route pattern match  (e.g. "POST /api/login")
      3. Semantic retrieval via vector search
    """

    # HTTP route pattern: optional METHOD + path
    _ROUTE_RE = re.compile(
        r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[^\s]*)$",
        re.IGNORECASE,
    )

    def __init__(self, call_graph: CallGraphStore, retriever=None):
        self._cg = call_graph
        self._retriever = retriever  # repoask Retriever, optional

    def resolve(self, query: str) -> str | None:
        """Return the best-matching symbol name, or None if nothing found."""
        # Priority 1: exact match
        result = self._exact_match(query)
        if result:
            return result

        # Priority 2: route match
        result = self._route_match(query)
        if result:
            return result

        # Priority 3: partial symbol search in call graph
        result = self._graph_search(query)
        if result:
            return result

        # Priority 4: semantic retrieval
        if self._retriever is not None:
            result = self._semantic_match(query)
            if result:
                return result

        return None

    # ------------------------------------------------------------------ #

    def _exact_match(self, query: str) -> str | None:
        """Return the symbol if it exists verbatim in the call graph."""
        all_symbols = self._cg.all_symbols()
        # Exact
        if query in all_symbols:
            return query
        # Case-insensitive
        lower = query.lower()
        for sym in all_symbols:
            if sym.lower() == lower:
                return sym
        return None

    def _route_match(self, query: str) -> str | None:
        """Match patterns like 'POST /api/login' against symbol names."""
        m = self._ROUTE_RE.match(query.strip())
        if not m:
            return None

        method = m.group(1).upper()
        path = m.group(2).lower()

        all_symbols = self._cg.all_symbols()
        # Look for a symbol whose name contains the method + a path segment
        path_slug = path.strip("/").replace("/", "_").replace("-", "_")
        candidates = []
        for sym in all_symbols:
            sym_lower = sym.lower()
            # e.g. "login" in "POST /api/login" → match AuthController.login
            if any(seg in sym_lower for seg in path.strip("/").split("/")):
                candidates.append(sym)

        if not candidates:
            return None
        # Prefer symbols that also contain the HTTP method name
        preferred = [s for s in candidates if method.lower() in s.lower()]
        return (preferred or candidates)[0]

    def _graph_search(self, query: str) -> str | None:
        """Search call graph symbols by substring, word-by-word for natural language."""
        words = [w.strip("?.,!") for w in query.lower().split() if len(w.strip("?.,!")) > 2]

        # Try the full query first (works for exact phrases like "createEmployee")
        matches = self._cg.search_symbol(query)

        # Then try each word individually and union the results
        seen: set[str] = set(matches)
        for word in words:
            for sym in self._cg.search_symbol(word):
                if sym not in seen:
                    seen.add(sym)
                    matches.append(sym)

        if not matches:
            return None

        def score(sym: str) -> tuple:
            s = sym.lower()
            word_hits = sum(1 for w in words if w in s)
            # Prefer top-level symbols (no dot = CLI command / module function)
            # and shorter names (less deeply nested)
            depth_penalty = sym.count(".")
            return (word_hits, -depth_penalty, -len(sym))

        matches.sort(key=score, reverse=True)
        return matches[0]

    def _semantic_match(self, query: str) -> str | None:
        """Use vector retrieval to find the best-matching symbol."""
        try:
            hits = self._retriever.retrieve(query, top_k=5)
        except Exception:
            return None

        for hit in hits:
            sym = hit.get("metadata", {}).get("symbol_name", "")
            if sym:
                # Check if this symbol is in our call graph
                all_syms = self._cg.all_symbols()
                # Try exact and suffix match (e.g. "login" → "AuthController.login")
                for known in all_syms:
                    if known == sym or known.endswith(f".{sym}") or sym.endswith(f".{known}"):
                        return known
                # Return the symbol itself even if not in graph (may still be useful)
                return sym

        return None
