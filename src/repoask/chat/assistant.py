from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..providers.base import LLMProvider, Message
from ..retrieval.context_builder import build_context
from ..retrieval.retriever import Retriever
from ..store.chroma import VectorStore

SYSTEM_PROMPT = """\
You are RepoAsk, an expert repository assistant.
You answer questions about the codebase using ONLY the context provided below.
When answering:
- Cite the exact file path and line numbers from the context (e.g. `src/foo.py:42`).
- If multiple functions are relevant, reference all of them.
- If the answer cannot be found in the provided context, say: "I don't have enough context to answer that — try re-indexing or narrowing your question to a specific file."
- Do NOT invent code or behaviour that is not present in the context.
- Keep answers concise but complete. Use markdown code blocks for code snippets.
"""


class ChatSession:
    """Holds the in-memory message history for a single interactive session."""

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever,
        store: VectorStore,
        history_db: Path | None = None,
        top_k: int = 10,
    ):
        self._llm = llm
        self._retriever = retriever
        self._store = store
        self._top_k = top_k
        self._messages: list[Message] = []
        self._history_db = history_db
        self._session_id = str(int(time.time()))

        if history_db:
            self._init_history_db()

    # ------------------------------------------------------------------ #
    # Persistent history
    # ------------------------------------------------------------------ #

    def _init_history_db(self) -> None:
        assert self._history_db
        self._history_db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._history_db))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, created_at REAL)"
        )
        self._conn.commit()

    def _persist_message(self, message: Message) -> None:
        if not self._history_db:
            return
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self._session_id, message.role, message.content, time.time()),
        )
        self._conn.commit()

    def load_last_session(self, n_messages: int = 10) -> None:
        """Optionally seed RAM context with the tail of the previous session."""
        if not self._history_db or not self._history_db.exists():
            return
        rows = self._conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = (SELECT session_id FROM messages ORDER BY created_at DESC LIMIT 1) "
            "ORDER BY created_at DESC LIMIT ?",
            (n_messages,),
        ).fetchall()
        for role, content in reversed(rows):
            self._messages.append(Message(role=role, content=content))

    # ------------------------------------------------------------------ #
    # Core ask
    # ------------------------------------------------------------------ #

    def ask(self, question: str, stream: bool = True):
        # 1. Retrieve relevant chunks
        hits = self._retriever.retrieve(question, top_k=self._top_k)

        # 2. Build context
        context = build_context(hits, self._store, question)

        # 3. Compose messages
        user_content = f"## Retrieved context\n\n{context}\n\n## Question\n\n{question}"
        user_msg = Message(role="user", content=user_content)
        self._messages.append(user_msg)
        self._persist_message(user_msg)

        full_messages = [Message(role="system", content=SYSTEM_PROMPT)] + self._messages

        # 4. Call LLM
        if stream:
            return self._stream_response(full_messages)
        else:
            response_text = self._llm.chat(full_messages, stream=False)
            assistant_msg = Message(role="assistant", content=response_text)
            self._messages.append(assistant_msg)
            self._persist_message(assistant_msg)
            return response_text

    def _stream_response(self, messages: list[Message]):
        chunks = []
        for chunk in self._llm.chat(messages, stream=True):
            chunks.append(chunk)
            yield chunk
        full_text = "".join(chunks)
        assistant_msg = Message(role="assistant", content=full_text)
        self._messages.append(assistant_msg)
        self._persist_message(assistant_msg)

    def clear(self) -> None:
        self._messages.clear()
