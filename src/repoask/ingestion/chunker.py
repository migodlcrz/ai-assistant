from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tracing.call_graph import CallRelationship


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
        # export_statement intentionally excluded — it wraps the above node types
        # and would produce duplicate chunks with identical line ranges
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
        # export_statement intentionally excluded — it wraps the above node types
        # and would produce duplicate chunks with identical line ranges
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


# --------------------------------------------------------------------------- #
# Call graph extraction
# --------------------------------------------------------------------------- #

def _collect_calls_python(node, source: bytes) -> list[str]:
    """Walk an AST node and return all called identifiers (Python)."""
    calls: list[str] = []

    def walk(n):
        if n.type == "call":
            func_node = n.child_by_field_name("function")
            if func_node is not None:
                name = _call_name_python(func_node, source)
                if name:
                    calls.append(name)
        for child in n.children:
            walk(child)

    walk(node)
    return calls


def _call_name_python(node, source: bytes) -> str:
    """Extract a qualified call name from a Python call's function node."""
    if node.type == "identifier":
        return _decode(node.text)
    if node.type == "attribute":
        obj = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if obj is not None and attr is not None:
            obj_name = _decode(obj.text).split("(")[0]
            attr_name = _decode(attr.text)
            return f"{obj_name}.{attr_name}"
    return ""


def _collect_calls_js_ts(node, source: bytes) -> list[str]:
    """Walk an AST node and return all called identifiers (JS/TS)."""
    calls: list[str] = []

    def walk(n):
        if n.type == "call_expression":
            func_node = n.child_by_field_name("function")
            if func_node is not None:
                name = _call_name_js_ts(func_node, source)
                if name:
                    calls.append(name)
        for child in n.children:
            walk(child)

    walk(node)
    return calls


def _call_name_js_ts(node, source: bytes) -> str:
    if node.type == "identifier":
        return _decode(node.text)
    if node.type == "member_expression":
        obj = node.child_by_field_name("object")
        prop = node.child_by_field_name("property")
        if obj is not None and prop is not None:
            obj_name = _decode(obj.text).split("(")[0]
            prop_name = _decode(prop.text)
            return f"{obj_name}.{prop_name}"
    return ""


def _decode(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val) if val is not None else ""


def extract_call_relationships(
    path: Path, root: Path, language: str
) -> tuple[list[CallRelationship], list[tuple[str, str, int]]]:
    """
    Return (relationships, locations).

    relationships: caller→callee edges found in the file.
    locations: (symbol, rel_path, start_line) for every defined symbol.
    """
    from ..tracing.call_graph import CallRelationship

    rel_path = str(path.relative_to(root))
    parser = _get_parser(language)
    if parser is None:
        return [], []

    source_bytes = path.read_bytes()
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    symbol_map = _SYMBOL_TYPES.get(language, {})
    relationships: list[CallRelationship] = []
    locations: list[tuple[str, str, int]] = []

    if language == "python":
        _collect_calls = _collect_calls_python
    else:
        _collect_calls = _collect_calls_js_ts

    def visit(node, enclosing: str | None, depth: int):
        node_type = node.type
        sym_type = symbol_map.get(node_type)

        if sym_type and depth <= 2:
            name = _node_name(node, source_bytes)
            qualified = (
                f"{enclosing}.{name}"
                if enclosing and sym_type == "function"
                else name
            )
            start_line = node.start_point[0] + 1
            locations.append((qualified, rel_path, start_line))

            callees = _collect_calls(node, source_bytes)
            for callee in callees:
                # Normalize self.method → EnclosingClass.method when inside a class
                if callee.startswith("self.") and enclosing:
                    callee = f"{enclosing}.{callee[5:]}"
                if callee and callee != qualified:
                    relationships.append(
                        CallRelationship(
                            caller=qualified,
                            callee=callee,
                            file_path=rel_path,
                        )
                    )
            new_enclosing = name if sym_type == "class" else enclosing
            for child in node.children:
                visit(child, new_enclosing, depth + 1)
        else:
            for child in node.children:
                visit(child, enclosing, depth)

    visit(root_node, None, 0)
    return relationships, locations
