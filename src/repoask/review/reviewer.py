from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class FileContext:
    file_path: str
    diff: str
    symbols: list[dict] = field(default_factory=list)   # from vector store metadata
    callers: list[str] = field(default_factory=list)    # call graph: who calls this file
    callees: list[str] = field(default_factory=list)    # call graph: what this file calls
    rag_chunks: list[dict] = field(default_factory=list)  # similar patterns from repo


@dataclass
class FileReview:
    file_path: str
    summary: str
    risks: str
    security: str
    architecture: str
    tests: str
    side_effects: str
    performance: str


# --------------------------------------------------------------------------- #
# Git helpers
# --------------------------------------------------------------------------- #

def get_changed_files(repo_root: Path, base_branch: str = "main") -> list[str]:
    """
    Return all changed/added/deleted file paths relative to repo root.

    Covers four sources:
      1. Committed changes ahead of base branch (git diff main...HEAD)
      2. Staged changes not yet committed (git diff --cached)
      3. Unstaged modifications (git diff)
      4. Untracked new files (git ls-files --others)
    """
    seen: set[str] = set()
    files: list[str] = []

    def _run(*args) -> list[str]:
        try:
            result = subprocess.run(
                list(args),
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except Exception:
            return []

    _SKIP_SUFFIXES = {
        ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
        ".egg-info", ".lock", ".log",
    }
    _SKIP_FRAGMENTS = {"__pycache__", ".egg-info/", "node_modules/", ".DS_Store"}

    def _keep(path: str) -> bool:
        from pathlib import PurePosixPath
        p = PurePosixPath(path)
        if p.suffix in _SKIP_SUFFIXES:
            return False
        if any(frag in path for frag in _SKIP_FRAGMENTS):
            return False
        return True

    for path in (
        # committed ahead of base
        _run("git", "diff", "--name-only", f"{base_branch}...HEAD")
        # staged
        + _run("git", "diff", "--name-only", "--cached")
        # unstaged modifications / deletions
        + _run("git", "diff", "--name-only")
        # untracked new files
        + _run("git", "ls-files", "--others", "--exclude-standard")
    ):
        if path not in seen and _keep(path):
            seen.add(path)
            files.append(path)

    return files


def get_file_diff(repo_root: Path, file_path: str, base_branch: str = "main") -> str:
    """Return the unified diff for a single file, covering all change states."""

    def _run(*args) -> str:
        try:
            result = subprocess.run(
                list(args),
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    # Try committed diff first
    diff = _run("git", "diff", f"{base_branch}...HEAD", "--", file_path)
    if diff:
        return diff

    # Try staged diff
    diff = _run("git", "diff", "--cached", "--", file_path)
    if diff:
        return diff

    # Try unstaged diff
    diff = _run("git", "diff", "--", file_path)
    if diff:
        return diff

    # Untracked file — show full content as a new-file diff
    content = get_file_content(repo_root, file_path)
    if content:
        lines = "\n".join(f"+ {line}" for line in content.splitlines())
        return f"--- /dev/null\n+++ {file_path}\n{lines}"

    return ""


def get_file_content(repo_root: Path, file_path: str) -> str:
    """Return full current content of a file (HEAD)."""
    abs_path = repo_root / file_path
    if abs_path.exists():
        return abs_path.read_text(errors="replace")
    return ""


# --------------------------------------------------------------------------- #
# Context gathering
# --------------------------------------------------------------------------- #

def gather_file_context(
    file_path: str,
    repo_root: Path,
    base_branch: str,
    store,
    retriever,
    call_graph,
) -> FileContext:
    diff = get_file_diff(repo_root, file_path, base_branch)

    # Symbols defined in this file (from vector store metadata)
    symbols = []
    try:
        results = store._col.get(where={"file_path": {"$eq": file_path}}, include=["metadatas"])
        for meta in (results.get("metadatas") or []):
            if meta.get("symbol_type") not in ("repo_map", "block"):
                symbols.append({
                    "name": meta.get("symbol_name", ""),
                    "type": meta.get("symbol_type", ""),
                    "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                })
    except Exception:
        pass

    # Call graph: callers of symbols in this file, and what this file calls
    callers: list[str] = []
    callees: list[str] = []
    if call_graph is not None:
        try:
            all_edges = call_graph._conn.execute(
                "SELECT DISTINCT caller, callee FROM call_graph WHERE file_path = ?",
                (file_path,),
            ).fetchall()
            file_callers_q = call_graph._conn.execute(
                "SELECT DISTINCT caller FROM call_graph WHERE callee IN "
                "(SELECT DISTINCT caller FROM call_graph WHERE file_path = ?)",
                (file_path,),
            ).fetchall()
            callees = list({r[1] for r in all_edges})
            callers = list({r[0] for r in file_callers_q})
        except Exception:
            pass

    # RAG: retrieve related chunks (similar code patterns / business logic)
    rag_chunks: list[dict] = []
    if retriever is not None and diff:
        try:
            query = f"code similar to changes in {file_path}: {diff[:500]}"
            hits = retriever.retrieve(query, top_k=6)
            rag_chunks = [
                h for h in hits
                if h.get("metadata", {}).get("file_path") != file_path
            ][:4]
        except Exception:
            pass

    return FileContext(
        file_path=file_path,
        diff=diff,
        symbols=symbols,
        callers=callers,
        callees=callees,
        rag_chunks=rag_chunks,
    )


# --------------------------------------------------------------------------- #
# LLM prompt construction
# --------------------------------------------------------------------------- #

_REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer performing a thorough code review.
You have full context of the repository — not just the diff.
You understand architecture, business logic, side effects, security, and test coverage.

You must produce a structured review. Be direct, specific, and actionable.
Do not be vague. If something looks fine, say "No issues detected." rather than leaving it empty.
Cite specific function names, line numbers, or patterns when relevant.
"""

_REVIEW_USER_TEMPLATE = """\
## File under review
{file_path}

## Diff (changes made)
```
{diff}
```

## Symbols defined in this file
{symbols}

## Call graph — who calls this file
{callers}

## Call graph — what this file calls (dependencies)
{callees}

## Related code patterns from the rest of the repository (RAG context)
{rag_context}

---

Produce a structured review with EXACTLY these sections.
Use plain text only — no markdown headers, no bold, no bullet symbols beyond simple dashes.

Summary:
[What changed and what it does]

Risks:
[Logical errors, broken flows, null risks, unreachable code, incorrect assumptions]

Security:
[Exposed secrets, unsafe input, SQL injection, sensitive logging, auth issues]

Architecture:
[Wrong layering, duplicated logic, tight coupling, bypassing service layers]

Tests:
[Missing tests, outdated tests, missing edge cases, tests not updated after changes]

Side Effects:
[Removed notifications, missing event/queue publishing, altered business workflows, unintended impacts on callers listed above]

Performance:
[N+1 queries, unnecessary loops, repeated API calls, inefficient processing]
"""


def build_review_prompt(ctx: FileContext) -> str:
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
        lines = f"{meta.get('start_line','?')}-{meta.get('end_line','?')}"
        snippet = hit.get("text", "")[:300]
        rag_parts.append(f"  [{fp} :: {sym} lines {lines}]\n  {snippet}")
    rag_text = "\n\n".join(rag_parts) or "  (none)"

    return _REVIEW_USER_TEMPLATE.format(
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

def parse_review_response(file_path: str, response: str) -> FileReview:
    sections = {
        "summary": "",
        "risks": "",
        "security": "",
        "architecture": "",
        "tests": "",
        "side_effects": "",
        "performance": "",
    }

    # Map label variations → canonical key
    label_map = {
        "summary": "summary",
        "risks": "risks",
        "risk": "risks",
        "security": "security",
        "architecture": "architecture",
        "tests": "tests",
        "test": "tests",
        "side effects": "side_effects",
        "side effect": "side_effects",
        "performance": "performance",
    }

    current_key = None
    buf: list[str] = []

    for line in response.splitlines():
        stripped = line.strip()
        lower = stripped.rstrip(":").lower()

        if lower in label_map:
            if current_key and buf:
                sections[current_key] = "\n".join(buf).strip()
            current_key = label_map[lower]
            buf = []
        elif current_key is not None:
            buf.append(stripped)

    if current_key and buf:
        sections[current_key] = "\n".join(buf).strip()

    return FileReview(
        file_path=file_path,
        summary=sections["summary"] or "No summary.",
        risks=sections["risks"] or "No issues detected.",
        security=sections["security"] or "No issues detected.",
        architecture=sections["architecture"] or "No issues detected.",
        tests=sections["tests"] or "No issues detected.",
        side_effects=sections["side_effects"] or "No issues detected.",
        performance=sections["performance"] or "No issues detected.",
    )


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

DIVIDER = "─" * 44


def format_report(changed_files: list[str], reviews: list[FileReview]) -> str:
    lines = []
    lines.append("REPO REVIEW REPORT")
    lines.append("")
    lines.append("Changed Files:")
    for f in changed_files:
        lines.append(f"  - {f}")
    lines.append("")
    lines.append(DIVIDER)

    for review in reviews:
        lines.append("")
        lines.append(f"FILE: {review.file_path}")
        lines.append("")
        lines.append("Summary:")
        lines.append(review.summary)
        lines.append("")
        lines.append("Risks:")
        lines.append(review.risks)
        lines.append("")
        lines.append("Security:")
        lines.append(review.security)
        lines.append("")
        lines.append("Architecture:")
        lines.append(review.architecture)
        lines.append("")
        lines.append("Tests:")
        lines.append(review.tests)
        lines.append("")
        lines.append("Side Effects:")
        lines.append(review.side_effects)
        lines.append("")
        lines.append("Performance:")
        lines.append(review.performance)
        lines.append("")
        lines.append(DIVIDER)

    return "\n".join(lines)
