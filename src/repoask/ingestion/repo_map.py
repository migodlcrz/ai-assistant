from __future__ import annotations

from pathlib import Path

from .chunker import Chunk

REPO_MAP_FILE = "__repo_map__"
REPO_MAP_SYMBOL = "__repo_map__"


def build_repo_map(all_chunks: list[Chunk], root: Path) -> Chunk:
    """
    Build a single high-level map of the entire repository from the already-extracted
    chunks. Groups symbols by file so the LLM can answer overview questions.
    """
    # Group symbols by file
    file_symbols: dict[str, dict[str, list[str]]] = {}
    for chunk in all_chunks:
        if chunk.symbol_type in ("repo_map",):
            continue
        fp = chunk.file_path
        st = chunk.symbol_type
        if fp not in file_symbols:
            file_symbols[fp] = {}
        if st not in file_symbols[fp]:
            file_symbols[fp][st] = []
        if chunk.symbol_name not in file_symbols[fp][st]:
            file_symbols[fp][st].append(chunk.symbol_name)

    lines: list[str] = ["# Repository Map\n"]
    for fp in sorted(file_symbols):
        parts: list[str] = []
        sym_map = file_symbols[fp]
        for sym_type in ("class", "interface", "function", "block", "module"):
            names = sym_map.get(sym_type, [])
            if names:
                parts.append(f"{sym_type}s: {', '.join(names)}")
        lines.append(f"## {fp}")
        if parts:
            lines.append("  " + " | ".join(parts))

    text = "\n".join(lines)

    return Chunk(
        text=text,
        file_path=REPO_MAP_FILE,
        language="text",
        symbol_name=REPO_MAP_SYMBOL,
        symbol_type="repo_map",
        start_line=1,
        end_line=len(lines),
    )
