from __future__ import annotations

from dataclasses import dataclass

from .graph_traverser import TraceNode


@dataclass
class SideEffect:
    kind: str    # "Database read" | "Database write" | "External API" | "Messaging"
    detail: str  # The symbol that triggered detection


_DB_READ_PATTERNS = {
    "find", "findbyid", "findone", "findall", "findby",
    "findfirst", "findlast", "get", "fetch", "load", "read",
    "select", "query", "search", "lookup", "retrieve",
    "exists", "count", "list",
}

_DB_WRITE_PATTERNS = {
    "insert", "create", "save", "update", "upsert", "patch",
    "delete", "remove", "destroy", "purge", "truncate",
    "write", "store", "persist", "commit",
}

_REPOSITORY_SUFFIXES = (
    "repository", "repo", "dao", "store", "storage", "mapper",
    "model", "entity", "table", "db", "database",
)

# Substrings that indicate an SDK/HTTP client call → External API
_EXTERNAL_API_FRAGMENTS = (
    "._client.", ".client.", "requests.get", "requests.post", "requests.put",
    "requests.patch", "requests.delete", "urllib", "httpx", "aiohttp",
    "axios", "httpclient", "apiclient",
    # Known third-party SDK constructors / namespaced calls
    "chromadb.", "anthropic.", "openai.", "groq.", "boto3.", "botocore.",
    "stripe.", "twilio.", "sendgrid.", "firebase.", "supabase.",
)

_EXTERNAL_API_PATTERNS = {
    "fetch", "request", "axios", "got", "httpclient", "apiclient",
    "requests.get", "requests.post", "requests.put", "requests.patch",
    "requests.delete", "urllib", "httpx", "aiohttp", "session.get",
    "session.post",
}

# Verbs that are only DB operations when the symbol has clear repo context
_DB_CONTEXT_REQUIRED = {"get", "create", "delete", "exists", "load", "list", "count"}

# Substrings that mark a symbol as NOT a real DB operation
_FILESYSTEM_FRAGMENTS = (".exists", "file.exists", "dir.exists", "path.exists", "_file.exists")

_MESSAGING_PATTERNS = {
    "emailservice", "notificationservice", "mailer", "sendmail",
    "sendemail", "sendnotification", "publish", "send", "dispatch",
    "enqueue", "queue.send", "sns.publish", "sqs.send", "kafka.send",
    "producer.send", "emit", "fire",
}

_MESSAGING_NAME_FRAGMENTS = (
    "emailservice", "notificationservice", "mailerservice",
    "queue", "sns", "sqs", "kafka", "pubsub", "mailer",
    # "message" / "event" / "worker" intentionally excluded — too broad
    # (e.g. "user_messages" is not a messaging side effect)
)


def detect_side_effects(root: TraceNode) -> list[SideEffect]:
    """Walk the trace tree and return detected side effects."""
    effects: list[SideEffect] = []
    seen: set[str] = set()

    def _add(kind: str, detail: str):
        key = f"{kind}:{detail}"
        if key not in seen:
            seen.add(key)
            effects.append(SideEffect(kind=kind, detail=detail))

    def walk(node: TraceNode):
        sym = node.symbol
        sym_lower = sym.lower()
        local_name = sym_lower.split(".")[-1]  # last segment after dot

        # --- Skip filesystem / path ops (not DB) ---
        is_filesystem = any(frag in sym_lower for frag in _FILESYSTEM_FRAGMENTS)

        # --- External API detection (checked before DB to avoid misclassification) ---
        is_sdk_call = any(frag in sym_lower for frag in _EXTERNAL_API_FRAGMENTS)
        is_api_pattern = sym_lower in _EXTERNAL_API_PATTERNS or any(
            sym_lower.startswith(p) for p in _EXTERNAL_API_PATTERNS
        )
        if is_sdk_call or is_api_pattern:
            _add("External API", sym)

        # --- Database detection ---
        elif not is_filesystem:
            is_repo_context = any(frag in sym_lower for frag in _REPOSITORY_SUFFIXES)

            if is_repo_context:
                if local_name in _DB_READ_PATTERNS or any(local_name.startswith(p) for p in _DB_READ_PATTERNS):
                    _add("Database read", sym)
                elif local_name in _DB_WRITE_PATTERNS or any(local_name.startswith(p) for p in _DB_WRITE_PATTERNS):
                    _add("Database write", sym)
            else:
                # Without repo context, only trigger on verbs that are unambiguously DB
                # (exclude generic verbs like get/exists/create/load which appear everywhere)
                if local_name in _DB_READ_PATTERNS and local_name not in _DB_CONTEXT_REQUIRED and not _is_messaging(sym_lower):
                    _add("Database read", sym)
                elif local_name in _DB_WRITE_PATTERNS and local_name not in _DB_CONTEXT_REQUIRED and not _is_messaging(sym_lower):
                    _add("Database write", sym)

        # --- Messaging detection ---
        if _is_messaging(sym_lower):
            _add("Messaging", sym)

        if not node.is_cycle:
            for child in node.children:
                walk(child)

    walk(root)
    return effects


def _is_messaging(sym_lower: str) -> bool:
    if any(frag in sym_lower for frag in _MESSAGING_NAME_FRAGMENTS):
        return True
    local = sym_lower.split(".")[-1]
    return local in _MESSAGING_PATTERNS or sym_lower in _MESSAGING_PATTERNS
