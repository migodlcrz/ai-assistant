from __future__ import annotations

from pathlib import Path

import pathspec

from ..config.schema import IndexingConfig

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _build_spec(patterns: list[str]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _load_gitignore(root: Path) -> pathspec.PathSpec:
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return pathspec.PathSpec.from_lines("gitwildmatch", [])


def scan_files(root: Path, config: IndexingConfig) -> list[Path]:
    ignore_spec = _build_spec(config.ignore_patterns)
    gitignore_spec = _load_gitignore(root)
    max_bytes = config.max_file_size_kb * 1024
    supported_langs = set(config.languages)

    result: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        rel_str = str(rel)

        # Check any path component against ignore patterns (handles dir names like node_modules)
        parts_match = any(
            ignore_spec.match_file(part) or ignore_spec.match_file(str(Path(part)))
            for part in rel.parts
        )
        if parts_match or ignore_spec.match_file(rel_str):
            continue

        if gitignore_spec.match_file(rel_str):
            continue

        lang = LANGUAGE_EXTENSIONS.get(path.suffix.lower())
        if lang is None or lang not in supported_langs:
            continue

        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue

        result.append(path)

    return sorted(result)
