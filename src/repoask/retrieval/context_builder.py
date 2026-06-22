from __future__ import annotations

from ..ingestion.repo_map import REPO_MAP_FILE, REPO_MAP_SYMBOL
from ..store.chroma import VectorStore


_MAX_CONTEXT_CHARS = 12_000
_DEPENDENCY_TOP_K = 3


def _fetch_repo_map(store: VectorStore) -> dict | None:
    try:
        results = store._col.get(
            where={"file_path": {"$eq": REPO_MAP_FILE}},
            include=["documents", "metadatas"],
        )
        if results and results["documents"]:
            return {
                "text": results["documents"][0],
                "metadata": results["metadatas"][0],
            }
    except Exception:
        pass
    return None


def build_context(hits: list[dict], store: VectorStore, query: str) -> str:
    """
    Given top-K retrieval hits, build an enriched context string for the LLM.

    Strategy:
    1. Deduplicate hits by (file_path, symbol_name).
    2. For each hit, pull in sibling chunks from the same file if the hit is
       a function — this surfaces immediately adjacent helper definitions.
    3. Assemble into a structured block with file paths and line numbers so
       the LLM can cite exact locations.
    """
    seen_ids: set[str] = set()
    primary_chunks: list[dict] = []

    # Always inject the repo map first so the LLM has a bird's-eye view
    repo_map = _fetch_repo_map(store)
    if repo_map:
        repo_map_block = f"### Repository Overview\n```\n{repo_map['text']}\n```"
    else:
        repo_map_block = None

    for hit in hits:
        if hit["metadata"].get("file_path") == REPO_MAP_FILE:
            continue
        chunk_id = hit["id"]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            primary_chunks.append(hit)

    # For each primary hit that is a function, attempt to pull in the
    # definition of any symbols mentioned in its imports list.
    enrichment_chunks: list[dict] = []
    for hit in primary_chunks[:5]:  # only enrich top-5 to control context size
        meta = hit["metadata"]
        imports_raw: str = meta.get("imports", "")
        if not imports_raw:
            continue

        import_names = [i.strip() for i in imports_raw.split(",") if i.strip()]
        for name in import_names[:4]:
            try:
                results = store.query(
                    # Re-use the collection but filter by symbol_name
                    embedding=store._col.get(
                        where={"symbol_name": {"$eq": name}},
                        include=["embeddings"],
                        limit=1,
                    ).get("embeddings", [[]])[0] or [],
                    top_k=_DEPENDENCY_TOP_K,
                    where={"symbol_name": {"$eq": name}},
                )
                for r in results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        enrichment_chunks.append(r)
            except Exception:
                pass

    all_chunks = primary_chunks + enrichment_chunks

    # Build final context string — repo map always goes first
    parts: list[str] = []
    total_chars = 0

    if repo_map_block:
        parts.append(repo_map_block)
        total_chars += len(repo_map_block)

    for chunk in all_chunks:
        meta = chunk["metadata"]
        header = (
            f"### {meta['file_path']}  "
            f"[{meta['symbol_type']}: {meta['symbol_name']}]  "
            f"lines {meta['start_line']}–{meta['end_line']}"
        )
        block = f"{header}\n```{meta['language']}\n{chunk['text']}\n```"

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)
