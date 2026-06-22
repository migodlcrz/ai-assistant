from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    text: str
    file_path: str          # relative to repo root
    language: str
    symbol_name: str        # function/class/module name
    symbol_type: str        # "function" | "class" | "interface" | "module" | "block"
    start_line: int
    end_line: int
    imports: list[str] = field(default_factory=list)   # identifiers imported by this file


# --------------------------------------------------------------------------- #
# Language parsers
# --------------------------------------------------------------------------- #

_PARSERS: dict[str, Any] = {}


def _get_parser(language: str):
    if language in _PARSERS:
        return _PARSERS[language]

    try:
        import tree_sitter_python as tspython
        import tree_sitter_javascript as tsjavascript
        import tree_sitter_typescript as tstypescript
        from tree_sitter import Language, Parser
    except ImportError:
        raise SystemExit(
            "tree-sitter language packages are required.\n"
            "pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript"
        )

    lang_map = {
        "python": tspython.language(),
        "javascript": tsjavascript.language(),
        "typescript": tstypescript.language_typescript(),
    }

    raw = lang_map.get(language)
    if raw is None:
        return None

    parser = Parser(Language(raw))
    _PARSERS[language] = parser
    return parser


# --------------------------------------------------------------------------- #
# Import extraction helpers
# --------------------------------------------------------------------------- #

def _extract_imports_python(root_node) -> list[str]:
    imports: list[str] = []
    def walk(node):
        if node.type in ("import_statement", "import_from_statement"):
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    name = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                    imports.append(name.split(" as ")[0].strip())
        for child in node.children:
            walk(child)
    walk(root_node)
    return imports


def _extract_imports_js_ts(root_node) -> list[str]:
    imports: list[str] = []
    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "string":
                    val = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                    imports.append(val.strip("'\""))
        for child in node.children:
            walk(child)
    walk(root_node)
    return imports


# --------------------------------------------------------------------------- #
# Node-type sets per language
# --------------------------------------------------------------------------- #

_SYMBOL_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "async_function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "function",  # may wrap class or function
    },
    "javascript": {
        "function_declaration": "function",
        "function_expression": "function",
        "arrow_function": "function",
        "class_declaration": "class",
        "method_definition": "function",
        "lexical_declaration": "block",
        "variable_declaration": "block",
        "export_statement": "block",
    },
    "typescript": {
        "function_declaration": "function",
        "function_expression": "function",
        "arrow_function": "function",
        "class_declaration": "class",
        "method_definition": "function",
        "interface_declaration": "interface",
        "type_alias_declaration": "interface",
        "enum_declaration": "class",
        "lexical_declaration": "block",
        "variable_declaration": "block",
        "export_statement": "block",
    },
}


def _node_name(node, source: bytes) -> str:
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
    # For export_statement, look one level deeper
    for child in node.children:
        for grandchild in child.children:
            if grandchild.type in ("identifier", "type_identifier"):
                return grandchild.text.decode("utf-8") if isinstance(grandchild.text, bytes) else grandchild.text
    snippet = source[node.start_byte:node.start_byte + 40].decode("utf-8", errors="replace")
    return snippet.split("\n")[0].strip()[:30] or "<anonymous>"


# --------------------------------------------------------------------------- #
# Fallback: line-based chunking
# --------------------------------------------------------------------------- #

_FALLBACK_WINDOW = 60
_FALLBACK_OVERLAP = 10


def _fallback_chunks(source: str, rel_path: str, language: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    step = _FALLBACK_WINDOW - _FALLBACK_OVERLAP
    i = 0
    while i < len(lines):
        end = min(i + _FALLBACK_WINDOW, len(lines))
        text = "\n".join(lines[i:end])
        chunks.append(Chunk(
            text=text,
            file_path=rel_path,
            language=language,
            symbol_name=f"lines_{i + 1}_{end}",
            symbol_type="block",
            start_line=i + 1,
            end_line=end,
        ))
        i += step
    return chunks


# --------------------------------------------------------------------------- #
# Main chunking entry point
# --------------------------------------------------------------------------- #

def chunk_file(path: Path, root: Path, language: str) -> list[Chunk]:
    rel_path = str(path.relative_to(root))
    source_bytes = path.read_bytes()
    source_str = source_bytes.decode("utf-8", errors="replace")

    parser = _get_parser(language)
    if parser is None:
        return _fallback_chunks(source_str, rel_path, language)

    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    # Extract file-level imports
    if language == "python":
        file_imports = _extract_imports_python(root_node)
    else:
        file_imports = _extract_imports_js_ts(root_node)

    symbol_map = _SYMBOL_TYPES.get(language, {})
    chunks: list[Chunk] = []

    def collect(node, depth: int = 0):
        node_type = node.type
        sym_type = symbol_map.get(node_type)

        if sym_type and depth <= 2:
            name = _node_name(node, source_bytes)
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            if text.strip():
                chunks.append(Chunk(
                    text=text,
                    file_path=rel_path,
                    language=language,
                    symbol_name=name,
                    symbol_type=sym_type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    imports=file_imports,
                ))
            # Still recurse into classes to get methods, but mark depth
            for child in node.children:
                collect(child, depth + 1)
        else:
            for child in node.children:
                collect(child, depth)

    collect(root_node)

    # If nothing was extracted (e.g. a file with only top-level expressions), fallback
    if not chunks:
        # Emit the whole file as a single module chunk
        if source_str.strip():
            chunks.append(Chunk(
                text=source_str,
                file_path=rel_path,
                language=language,
                symbol_name=path.stem,
                symbol_type="module",
                start_line=1,
                end_line=len(source_str.splitlines()),
                imports=file_imports,
            ))

    return chunks
