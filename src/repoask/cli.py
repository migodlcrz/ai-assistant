from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import print as rprint

from .config.manager import (
    GLOBAL_CONFIG_FILE,
    get_or_create_global_config,
    load_config,
    save_global_config,
)
from .config.schema import RepoAskConfig

app = typer.Typer(
    name="repoask",
    help="AI-powered repository assistant. Ask questions about any codebase.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    return Path.cwd()


def _store_dir(config: RepoAskConfig, root: Path) -> Path:
    return root / config.store.path / "chroma"


def _tracker_db(config: RepoAskConfig, root: Path) -> Path:
    return root / config.store.path / "tracker.db"


def _history_db(config: RepoAskConfig, root: Path) -> Path:
    return root / config.store.path / "history.db"


def _call_graph_db(config: RepoAskConfig, root: Path) -> Path:
    return root / config.store.path / "call_graph.db"


def _load_providers(config: RepoAskConfig):
    from .providers.embeddings.factory import create_embedding_provider
    from .providers.llm.factory import create_llm_provider

    with console.status("[dim]Loading providers...[/dim]", spinner="dots"):
        embedder = create_embedding_provider(config.embedding)
        llm = create_llm_provider(config.llm)
    return embedder, llm


def _build_session(config: RepoAskConfig, root: Path):
    from .providers.embeddings.factory import create_embedding_provider
    from .providers.llm.factory import create_llm_provider
    from .retrieval.retriever import Retriever
    from .store.chroma import VectorStore
    from .chat.assistant import ChatSession

    embedder = create_embedding_provider(config.embedding)
    llm = create_llm_provider(config.llm)
    store = VectorStore(_store_dir(config, root))
    retriever = Retriever(store, embedder)
    session = ChatSession(
        llm=llm,
        retriever=retriever,
        store=store,
        history_db=_history_db(config, root),
    )
    return session, store


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #

@app.command()
def init():
    """Interactive first-time setup: configure providers and API keys."""
    console.print(Panel("[bold]Welcome to RepoAsk[/bold]", subtitle="First-time setup"))

    config = get_or_create_global_config()

    # --- Embedding provider ---
    console.print("\n[bold]Embedding provider[/bold]")
    console.print("  [dim]1[/dim] huggingface  (local, no API key required)")
    console.print("  [dim]2[/dim] openai       (requires API key)")
    emb_choice = Prompt.ask("Choose", choices=["1", "2"], default="1")
    if emb_choice == "1":
        config.embedding.provider = "huggingface"
        config.embedding.model = Prompt.ask(
            "Model", default=config.embedding.model
        )
        config.embedding.api_key = ""
    else:
        config.embedding.provider = "openai"
        config.embedding.model = Prompt.ask("Model", default="text-embedding-3-small")
        config.embedding.api_key = Prompt.ask("OpenAI API key", password=True)

    # --- LLM provider ---
    console.print("\n[bold]LLM provider[/bold]")
    console.print("  [dim]1[/dim] groq      (fast, free tier available)")
    console.print("  [dim]2[/dim] openai")
    console.print("  [dim]3[/dim] anthropic")
    console.print("  [dim]4[/dim] ollama    (local, no API key required)")
    llm_choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="4")
    provider_map = {"1": "groq", "2": "openai", "3": "anthropic", "4": "ollama"}
    default_models = {
        "groq": "llama-3.1-8b-instant",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "ollama": "llama3.2",
    }
    config.llm.provider = provider_map[llm_choice]
    config.llm.model = Prompt.ask("Model", default=default_models[config.llm.provider])
    if config.llm.provider == "ollama":
        config.llm.base_url = Prompt.ask("Ollama base URL", default="http://localhost:11434")
        config.llm.api_key = ""
    else:
        config.llm.api_key = Prompt.ask(f"{config.llm.provider.capitalize()} API key", password=True)

    save_global_config(config)
    console.print(f"\n[green]Config saved to {GLOBAL_CONFIG_FILE}[/green]")
    console.print("[dim]Run [bold]repoask index[/bold] inside a repository to build the index.[/dim]")


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #

@app.command()
def index(
    full: Annotated[bool, typer.Option("--full", help="Force a complete re-index")] = False,
):
    """Scan and index the current repository (incremental by default)."""
    root = _repo_root()
    config = load_config(root)

    from .ingestion.scanner import scan_files, LANGUAGE_EXTENSIONS
    from .ingestion.chunker import chunk_file, extract_call_relationships
    from .ingestion.tracker import FileTracker
    from .ingestion.repo_map import build_repo_map, REPO_MAP_FILE
    from .providers.embeddings.factory import create_embedding_provider
    from .store.chroma import VectorStore
    from .tracing.call_graph import CallGraphStore

    tracker = FileTracker(_tracker_db(config, root))
    store = VectorStore(_store_dir(config, root))
    call_graph = CallGraphStore(_call_graph_db(config, root))

    console.print(f"[dim]Scanning [bold]{root}[/bold]...[/dim]")
    all_files = scan_files(root, config.indexing)
    console.print(f"[dim]Found {len(all_files)} source files[/dim]")

    # Detect deleted files
    indexed_paths = set(tracker.all_indexed_paths())
    current_paths = {str(f.relative_to(root)) for f in all_files}
    deleted = indexed_paths - current_paths
    if deleted:
        for rel in deleted:
            store.delete_by_file(rel)
            call_graph.delete_by_file(rel)
            tracker.remove(root / rel, root)
        console.print(f"[dim]Removed {len(deleted)} deleted file(s) from index[/dim]")

    # Filter to changed/new files
    if full:
        files_to_index = all_files
        console.print("[yellow]Full re-index requested.[/yellow]")
    else:
        files_to_index = [f for f in all_files if tracker.is_changed(f, root)]

    if not files_to_index:
        console.print("[green]Index is up to date. Nothing to do.[/green]")
        return

    console.print(f"Indexing [bold]{len(files_to_index)}[/bold] file(s)...")

    with console.status("[dim]Loading embedding model...[/dim]", spinner="dots"):
        embedder = create_embedding_provider(config.embedding)

    EMBED_BATCH = 32
    total_chunks = 0
    all_chunks_for_map: list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        file_task = progress.add_task("Indexing files", total=len(files_to_index))

        for path in files_to_index:
            lang_ext = {".py": "python", ".js": "javascript", ".jsx": "javascript",
                        ".ts": "typescript", ".tsx": "typescript"}
            language = lang_ext.get(path.suffix.lower(), "python")

            try:
                chunks = chunk_file(path, root, language)
            except Exception as e:
                console.print(f"[yellow]Warning: could not parse {path.relative_to(root)}: {e}[/yellow]")
                progress.advance(file_task)
                continue

            if not chunks:
                tracker.mark_indexed(path, root)
                progress.advance(file_task)
                continue

            # Remove stale vectors and call edges for this file before upserting
            rel_str = str(path.relative_to(root))
            store.delete_by_file(rel_str)
            call_graph.delete_by_file(rel_str)

            # Extract and store call relationships + symbol locations
            try:
                relationships, locations = extract_call_relationships(path, root, language)
                if relationships:
                    call_graph.upsert(relationships)
                if locations:
                    call_graph.upsert_locations(locations)
            except Exception as e:
                console.print(f"[yellow]Call graph warning for {path.relative_to(root)}: {e}[/yellow]")

            # Embed in batches
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i : i + EMBED_BATCH]
                texts = [c.text for c in batch]
                try:
                    embeddings = embedder.embed(texts)
                    store.upsert_chunks(batch, embeddings)
                    total_chunks += len(batch)
                except Exception as e:
                    console.print(f"[yellow]Embedding error for {path.relative_to(root)}: {e}[/yellow]")

            all_chunks_for_map.extend(chunks)
            tracker.mark_indexed(path, root)
            progress.advance(file_task)

    # Rebuild repo map from all currently indexed chunks (not just new ones)
    with console.status("[dim]Building repo map...[/dim]", spinner="dots"):
        try:
            # Fetch existing chunks to include unchanged files in the map
            existing = store._col.get(include=["metadatas"])
            from .ingestion.chunker import Chunk as _Chunk
            for meta in (existing["metadatas"] or []):
                if meta.get("symbol_type") == "repo_map":
                    continue
                # Only add files not already covered by this run
                if not any(c.file_path == meta["file_path"] for c in all_chunks_for_map):
                    all_chunks_for_map.append(_Chunk(
                        text="", file_path=meta["file_path"], language=meta["language"],
                        symbol_name=meta["symbol_name"], symbol_type=meta["symbol_type"],
                        start_line=meta["start_line"], end_line=meta["end_line"],
                    ))
            store.delete_by_file(REPO_MAP_FILE)
            repo_map_chunk = build_repo_map(all_chunks_for_map, root)
            repo_map_embedding = embedder.embed_one(repo_map_chunk.text)
            store.upsert_chunks([repo_map_chunk], [repo_map_embedding])
        except Exception as e:
            console.print(f"[yellow]Warning: could not build repo map: {e}[/yellow]")

    stats = store.stats()
    console.print(
        f"[green]Done.[/green] Indexed [bold]{total_chunks}[/bold] new chunks "
        f"across [bold]{stats['files']}[/bold] files "
        f"([bold]{stats['chunks']}[/bold] total in store)."
    )


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #

@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask about the repository")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of chunks to retrieve")] = 10,
    language: Annotated[Optional[str], typer.Option("--lang", help="Filter to a specific language")] = None,
):
    """Ask a one-shot question about the repository."""
    root = _repo_root()
    config = load_config(root)

    store_dir = _store_dir(config, root)
    if not store_dir.exists():
        err_console.print("[red]No index found. Run [bold]repoask index[/bold] first.[/red]")
        raise typer.Exit(1)

    with console.status("[dim]Loading...[/dim]", spinner="dots"):
        session, _ = _build_session(config, root)

    console.print()
    console.rule("[dim]Answer[/dim]")
    console.print()

    full_response = []
    try:
        for chunk in session.ask(question, stream=True):
            console.print(chunk, end="", highlight=False)
            full_response.append(chunk)
    except Exception as e:
        err_console.print(f"[red]LLM error: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.rule()


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #

@app.command()
def chat(
    top_k: Annotated[int, typer.Option("--top-k")] = 10,
    no_history: Annotated[bool, typer.Option("--no-history", help="Don't persist this session")] = False,
):
    """Start an interactive chat session about the repository."""
    root = _repo_root()
    config = load_config(root)

    store_dir = _store_dir(config, root)
    if not store_dir.exists():
        err_console.print("[red]No index found. Run [bold]repoask index[/bold] first.[/red]")
        raise typer.Exit(1)

    from .providers.embeddings.factory import create_embedding_provider
    from .providers.llm.factory import create_llm_provider
    from .retrieval.retriever import Retriever
    from .store.chroma import VectorStore
    from .chat.assistant import ChatSession

    with console.status("[dim]Loading...[/dim]", spinner="dots"):
        embedder = create_embedding_provider(config.embedding)
        llm = create_llm_provider(config.llm)
        store = VectorStore(store_dir)
        retriever = Retriever(store, embedder)
        history_db = None if no_history else _history_db(config, root)
        session = ChatSession(
            llm=llm,
            retriever=retriever,
            store=store,
            history_db=history_db,
            top_k=top_k,
        )

    stats = store.stats()
    console.print(
        Panel(
            f"[bold]RepoAsk[/bold]  ·  {stats['chunks']} chunks · {stats['files']} files indexed\n"
            "[dim]Type your question. Commands: [bold]/clear[/bold] · [bold]/quit[/bold][/dim]",
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold]You[/bold]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question.strip():
            continue

        if question.strip().lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if question.strip().lower() == "/clear":
            session.clear()
            console.print("[dim]Session cleared.[/dim]")
            continue

        console.print()
        console.rule("[dim]Assistant[/dim]")
        console.print()

        try:
            for chunk in session.ask(question, stream=True):
                console.print(chunk, end="", highlight=False)
        except Exception as e:
            err_console.print(f"\n[red]Error: {e}[/red]")

        console.print()
        console.rule()


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #

@app.command()
def review(
    base: Annotated[str, typer.Option("--base", help="Base branch to diff against")] = "main",
    files: Annotated[Optional[str], typer.Option("--files", help="Comma-separated list of files to review (skips git detection)")] = None,
):
    """AI-powered review of all changed files in the current branch."""
    root = _repo_root()
    config = load_config(root)

    from .review.reviewer import (
        get_changed_files,
        gather_file_context,
        build_review_prompt,
        parse_review_response,
        format_report,
    )
    from .providers.llm.factory import create_llm_provider
    from .providers.base import Message

    # Detect changed files
    if files:
        changed = [f.strip() for f in files.split(",") if f.strip()]
    else:
        with console.status("[dim]Detecting changed files...[/dim]", spinner="dots"):
            changed = get_changed_files(root, base)

    if not changed:
        console.print(f"[green]No changed files detected against [bold]{base}[/bold].[/green]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Changed files ({len(changed)}):[/bold]")
    for f in changed:
        console.print(f"  [dim]-[/dim] {f}")
    console.print()

    # Load providers
    store_dir = _store_dir(config, root)
    store = None
    retriever = None
    call_graph = None

    if store_dir.exists():
        try:
            from .providers.embeddings.factory import create_embedding_provider
            from .retrieval.retriever import Retriever
            from .store.chroma import VectorStore

            with console.status("[dim]Loading embedding model...[/dim]", spinner="dots"):
                embedder = create_embedding_provider(config.embedding)
            store = VectorStore(store_dir)
            retriever = Retriever(store, embedder)
        except Exception as e:
            console.print(f"[yellow]Warning: could not load vector store: {e}[/yellow]")

    cg_db = _call_graph_db(config, root)
    if cg_db.exists():
        try:
            from .tracing.call_graph import CallGraphStore
            call_graph = CallGraphStore(cg_db)
        except Exception:
            pass

    with console.status("[dim]Loading LLM...[/dim]", spinner="dots"):
        llm = create_llm_provider(config.llm)

    # Review each file
    reviews = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reviewing files", total=len(changed))

        for file_path in changed:
            progress.update(task, description=f"Reviewing [bold]{file_path}[/bold]")

            ctx = gather_file_context(
                file_path=file_path,
                repo_root=root,
                base_branch=base,
                store=store,
                retriever=retriever,
                call_graph=call_graph,
            )

            prompt = build_review_prompt(ctx)
            messages = [
                Message(role="system", content="You are a senior software engineer performing a thorough, repository-aware code review."),
                Message(role="user", content=prompt),
            ]

            try:
                response_text = llm.chat(messages, stream=False)
            except Exception as e:
                response_text = f"Summary:\nLLM error: {e}\n\nRisks:\nCould not complete review."

            reviews.append(parse_review_response(file_path, response_text))
            progress.advance(task)

    report = format_report(changed, reviews)

    console.print()
    console.rule("[dim]Review Report[/dim]")
    console.print()
    console.print(report)
    console.rule()


# --------------------------------------------------------------------------- #
# generate-test
# --------------------------------------------------------------------------- #

@app.command(name="generate-test")
def generate_test(
    base: Annotated[str, typer.Option("--base", help="Base branch to diff against")] = "main",
    files: Annotated[Optional[str], typer.Option("--files", help="Comma-separated list of files to generate tests for")] = None,
):
    """Generate unit, integration, and flow-based tests for all changed files."""
    root = _repo_root()
    config = load_config(root)

    from .review.reviewer import get_changed_files, gather_file_context
    from .testing.test_generator import build_test_prompt, parse_test_response, format_test_report
    from .providers.llm.factory import create_llm_provider
    from .providers.base import Message

    # Detect changed files
    if files:
        changed = [f.strip() for f in files.split(",") if f.strip()]
    else:
        with console.status("[dim]Detecting changed files...[/dim]", spinner="dots"):
            changed = get_changed_files(root, base)

    if not changed:
        console.print(f"[green]No changed files detected against [bold]{base}[/bold].[/green]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Changed files ({len(changed)}):[/bold]")
    for f in changed:
        console.print(f"  [dim]-[/dim] {f}")
    console.print()

    # Load providers
    store_dir = _store_dir(config, root)
    store = None
    retriever = None
    call_graph = None

    if store_dir.exists():
        try:
            from .providers.embeddings.factory import create_embedding_provider
            from .retrieval.retriever import Retriever
            from .store.chroma import VectorStore

            with console.status("[dim]Loading embedding model...[/dim]", spinner="dots"):
                embedder = create_embedding_provider(config.embedding)
            store = VectorStore(store_dir)
            retriever = Retriever(store, embedder)
        except Exception as e:
            console.print(f"[yellow]Warning: could not load vector store: {e}[/yellow]")

    cg_db = _call_graph_db(config, root)
    if cg_db.exists():
        try:
            from .tracing.call_graph import CallGraphStore
            call_graph = CallGraphStore(cg_db)
        except Exception:
            pass

    with console.status("[dim]Loading LLM...[/dim]", spinner="dots"):
        llm = create_llm_provider(config.llm)

    # Generate tests for each file
    plans = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating tests", total=len(changed))

        for file_path in changed:
            progress.update(task, description=f"Generating tests for [bold]{file_path}[/bold]")

            ctx = gather_file_context(
                file_path=file_path,
                repo_root=root,
                base_branch=base,
                store=store,
                retriever=retriever,
                call_graph=call_graph,
            )

            prompt = build_test_prompt(ctx)
            messages = [
                Message(role="system", content="You are a senior software engineer generating developer test cases for a code change. Use full repository context to reconstruct feature behavior and produce meaningful test scenarios."),
                Message(role="user", content=prompt),
            ]

            try:
                response_text = llm.chat(messages, stream=False)
            except Exception as e:
                response_text = f"Feature Summary:\nLLM error: {e}\n\nUnit Tests:\n- Could not generate tests."

            plans.append(parse_test_response(file_path, response_text))
            progress.advance(task)

    report = format_test_report(changed, plans)

    console.print()
    console.rule("[dim]Generate Test Report[/dim]")
    console.print()
    console.print(report)
    console.rule()


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #

@app.command()
def trace(
    query: Annotated[Optional[str], typer.Argument(help="Symbol name, route, or natural-language description")] = None,
    depth: Annotated[int, typer.Option("--depth", help="Maximum traversal depth")] = 8,
    plain: Annotated[bool, typer.Option("--plain", help="Plain text output (no tree characters)")] = False,
):
    """Trace execution flow through the repository from a symbol or query."""
    if not query:
        query = Prompt.ask("[bold]Trace[/bold]")
    if not query.strip():
        err_console.print("[red]Query cannot be empty.[/red]")
        raise typer.Exit(1)

    root = _repo_root()
    config = load_config(root)

    cg_db = _call_graph_db(config, root)
    if not cg_db.exists():
        err_console.print(
            "[red]No call graph found. Run [bold]repoask index[/bold] first.[/red]"
        )
        raise typer.Exit(1)

    from .tracing.call_graph import CallGraphStore
    from .tracing.entry_point_resolver import EntryPointResolver
    from .tracing.graph_traverser import GraphTraverser
    from .tracing.side_effect_detector import detect_side_effects
    from .tracing.trace_formatter import format_trace

    call_graph = CallGraphStore(cg_db)
    stats = call_graph.stats()
    if stats["edges"] == 0:
        err_console.print(
            "[yellow]Call graph is empty. Re-run [bold]repoask index[/bold] to populate it.[/yellow]"
        )
        raise typer.Exit(1)

    # Build retriever for semantic fallback (optional — skip if store not present)
    retriever = None
    store_dir = _store_dir(config, root)
    if store_dir.exists():
        try:
            from .providers.embeddings.factory import create_embedding_provider
            from .retrieval.retriever import Retriever
            from .store.chroma import VectorStore

            with console.status("[dim]Loading embedding model...[/dim]", spinner="dots"):
                embedder = create_embedding_provider(config.embedding)
            store = VectorStore(store_dir)
            retriever = Retriever(store, embedder)
        except Exception:
            pass  # degrade gracefully to graph-only resolution

    resolver = EntryPointResolver(call_graph, retriever)
    traverser = GraphTraverser(call_graph, max_depth=depth)

    with console.status(f"[dim]Resolving entry point for: {query!r}[/dim]", spinner="dots"):
        entry = resolver.resolve(query)

    if entry is None:
        err_console.print(
            f"[red]Could not resolve an entry point for: {query!r}[/red]\n"
            "[dim]Try using the exact symbol name (e.g. [bold]AuthController.login[/bold]) "
            "or re-index with [bold]repoask index[/bold].[/dim]"
        )
        raise typer.Exit(1)

    with console.status("[dim]Traversing call graph...[/dim]", spinner="dots"):
        root_node = traverser.traverse(entry)
        side_effects = detect_side_effects(root_node)

    output = format_trace(
        query=query,
        entry=entry,
        root=root_node,
        side_effects=side_effects,
        rich=not plain,
    )

    console.print()
    console.rule("[dim]Trace[/dim]")
    console.print()
    console.print(output)
    console.print()
    console.rule()


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

@app.command()
def status():
    """Show index statistics for the current repository."""
    root = _repo_root()
    config = load_config(root)

    store_dir = _store_dir(config, root)
    tracker_db = _tracker_db(config, root)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Repository", str(root))
    table.add_row("Index path", str(root / config.store.path))
    table.add_row("Embedding provider", f"{config.embedding.provider} / {config.embedding.model}")
    table.add_row("LLM provider", f"{config.llm.provider} / {config.llm.model}")

    if store_dir.exists():
        from .store.chroma import VectorStore
        store = VectorStore(store_dir)
        stats = store.stats()
        table.add_row("Chunks in store", str(stats["chunks"]))
        table.add_row("Files indexed", str(stats["files"]))
    else:
        table.add_row("Index", "[yellow]Not built — run repoask index[/yellow]")

    if tracker_db.exists():
        import sqlite3, time
        conn = sqlite3.connect(str(tracker_db))
        row = conn.execute("SELECT MAX(indexed_at) FROM file_hashes").fetchone()
        if row and row[0]:
            import datetime
            ts = datetime.datetime.fromtimestamp(row[0]).strftime("%Y-%m-%d %H:%M:%S")
            table.add_row("Last indexed", ts)

    console.print(table)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@app.command(name="config")
def show_config():
    """View the current configuration."""
    config = get_or_create_global_config()
    console.print(f"[dim]Config file:[/dim] {GLOBAL_CONFIG_FILE}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Section")
    table.add_column("Key")
    table.add_column("Value")

    def masked(val: str) -> str:
        if val and len(val) > 8:
            return val[:4] + "***" + val[-4:]
        return "***" if val else "[dim](not set)[/dim]"

    table.add_row("embedding", "provider", config.embedding.provider)
    table.add_row("embedding", "model", config.embedding.model)
    table.add_row("embedding", "api_key", masked(config.embedding.api_key))
    table.add_row("llm", "provider", config.llm.provider)
    table.add_row("llm", "model", config.llm.model)
    table.add_row("llm", "api_key", masked(config.llm.api_key))
    table.add_row("llm", "temperature", str(config.llm.temperature))
    table.add_row("llm", "max_tokens", str(config.llm.max_tokens))

    console.print(table)
    console.print(f"\n[dim]Edit directly: {GLOBAL_CONFIG_FILE}[/dim]")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    app()


if __name__ == "__main__":
    main()
