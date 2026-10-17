"""``drive-agent`` command-line interface."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from drive_agent import __version__
from drive_agent.core.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_FILE,
    ConfigError,
    Settings,
    extract_folder_id,
    load_settings,
)
from drive_agent.drive.models import SKIP_LABELS
from drive_agent.vectorstores.location import describe_location, namespace_for

if TYPE_CHECKING:
    from drive_agent.core.agent import AgentReply, DriveAgent
    from drive_agent.drive.doctor import CheckResult
    from drive_agent.indexing.loader import FolderLoad

app = typer.Typer(
    name="drive-agent",
    help="Answer questions from a Google Drive folder.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

CONFIG_HEADER = """\
# Drive Knowledge Agent configuration.
# Secrets (API keys, tokens) do NOT go here: put them in .env or the environment.
"""

ENV_TEMPLATE = """\
# Drive Knowledge Agent secrets. Never commit this file.
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
# Service account mode: path to the JSON key
GOOGLE_APPLICATION_CREDENTIALS=
# OAuth mode: path to the OAuth client JSON
GOOGLE_OAUTH_CLIENT_FILE=
"""


def _version(value: bool) -> None:
    if value:
        console.print(f"drive-agent {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to drive-agent.yaml.", dir_okay=False),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    ctx.obj = {"config": config}


def _settings(ctx: typer.Context) -> Settings:
    try:
        return load_settings(ctx.obj["config"])
    except ConfigError as exc:
        err_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _not_implemented(command: str, milestone: str) -> None:
    err_console.print(
        f"[yellow]`drive-agent {command}` is not implemented yet ({milestone}).[/yellow]"
    )
    raise typer.Exit(1)


@app.command()
def init(
    folder: Annotated[str | None, typer.Option(help="Drive folder URL or ID to index.")] = None,
    auth_mode: Annotated[str | None, typer.Option(help="service_account or oauth.")] = None,
    path: Annotated[Path, typer.Option(help="Where to write the config.")] = DEFAULT_CONFIG_FILE,
    force: Annotated[bool, typer.Option(help="Overwrite an existing config file.")] = False,
) -> None:
    """Interactive setup: writes drive-agent.yaml and a .env template."""
    if path.exists() and not force:
        err_console.print(f"[red]{path} already exists.[/red] Use --force to overwrite.")
        raise typer.Exit(1)

    if auth_mode is None:
        auth_mode = typer.prompt("Auth mode (service_account / oauth)", default="service_account")
    if folder is None:
        folder = typer.prompt("Google Drive folder URL or ID")

    try:
        settings = Settings.model_validate(
            {"drive": {"auth_mode": auth_mode, "folder_ids": [extract_folder_id(folder)]}}
        )
    except ValueError as exc:  # includes pydantic.ValidationError
        err_console.print(f"[red]Invalid input:[/red] {exc}")
        raise typer.Exit(1) from exc

    body = yaml.safe_dump(settings.to_yaml_dict(), sort_keys=False)
    path.write_text(CONFIG_HEADER + body, encoding="utf-8")
    console.print(f"[green]Wrote {path}[/green]")

    env_path = path.parent / ".env"
    if not env_path.exists():
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Wrote {env_path}[/green] (fill in your keys)")

    console.print("Next: [bold]drive-agent check[/bold] to verify Drive access.")


@app.command()
def check(
    ctx: typer.Context,
    show_skipped: Annotated[
        bool, typer.Option("--show-skipped", help="List every skipped file and why.")
    ] = False,
) -> None:
    """Verify Drive auth, read-only scope, and folder visibility."""
    from drive_agent.drive import doctor  # deferred: Google client import is slow

    settings = _settings(ctx)
    with console.status("Checking Google Drive access…"):
        result = doctor.run_check(settings)
    _render_check(result, show_skipped=show_skipped)
    raise typer.Exit(0 if result.ok else 1)


def _render_check(result: CheckResult, *, show_skipped: bool) -> None:
    if result.description is None:
        console.print(f"[red]✗ Auth ({result.mode})[/red]  {result.error}")
        return
    who = result.identity or "unknown identity"
    as_who = "" if who in result.description else f" as [bold]{who}[/bold]"
    if result.error:
        console.print(f"[red]✗ Auth[/red]  {result.description}{as_who}")
        console.print(f"  {result.error}")
        return
    console.print(f"[green]✓ Auth[/green]  {result.description}{as_who}")
    console.print("  scope: drive.readonly (read-only)")

    for folder in result.folders:
        console.print()
        if folder.report is None:
            console.print(f"[red]✗ Folder {folder.folder_id}[/red]")
            console.print(f"  {folder.error}")
            continue
        report, stats = folder.report, folder.report.stats
        where = "Shared Drive" if report.root.drive_id else "My Drive"
        console.print(
            f"[green]✓ Folder[/green]  [bold]{report.root.name}[/bold] ({report.root.id}) · {where}"
        )
        kinds = ", ".join(f"{k} {n}" for k, n in sorted(report.kinds().items()))
        console.print(f"  files to index: {len(report.files)}" + (f" ({kinds})" if kinds else ""))
        console.print(f"  sub-folders: {stats.folders} · shortcuts: {stats.shortcuts}")
        if report.skipped:
            reasons = ", ".join(
                f"{n} {SKIP_LABELS[reason]}" for reason, n in report.skip_reasons().most_common()
            )
            console.print(f"  skipped: {reasons}")
        if stats.depth_limited:
            console.print(f"  not entered (max_depth): {stats.depth_limited} sub-folder(s)")
        for warning in folder.warnings:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")
        if show_skipped and report.skipped:
            table = Table("path", "reason", "type")
            for item in report.skipped:
                reason = SKIP_LABELS[item.reason] + (f" ({item.detail})" if item.detail else "")
                table.add_row(item.path, reason, item.mime_type)
            console.print(table)


def _render_load(load: FolderLoad, show_chunks: int) -> None:
    console.print()
    if load.root is None or load.error:
        console.print(f"[red]✗ Folder {load.folder_id}[/red]  {load.error}")
        if load.root is None:
            return
    console.print(f"[bold]{load.root.name}[/bold] ({load.root.id})")
    table = Table("", "file", "chunks", "~tokens", "notes", box=None, pad_edge=False)
    for result in load.files:
        mark = (
            "[red]✗[/red]"
            if result.error
            else "[yellow]⚠[/yellow]"
            if result.warnings
            else "[green]✓[/green]"
        )
        notes = result.error or "; ".join(result.warnings)
        table.add_row(mark, result.file.path, str(len(result.chunks)), f"{result.tokens:,}", notes)
    console.print(table)
    if load.skipped:
        console.print(f"  skipped (see `drive-agent check --show-skipped`): {len(load.skipped)}")
    for result in load.files:
        for chunk in result.chunks[:show_chunks]:
            meta = chunk.metadata
            console.rule(f"{meta['chunk_id']}  {meta['path']}", style="dim")
            console.print(chunk.page_content, markup=False, highlight=False)


def _print_totals(loads: list[FolderLoad]) -> None:
    files = [r for load in loads for r in load.files]
    console.print()
    console.print(
        f"Total: {len(files)} files · {sum(len(r.chunks) for r in files)} chunks · "
        f"~{sum(r.tokens for r in files):,} tokens · {sum(1 for r in files if r.error)} failed · "
        f"{sum(len(load.skipped) for load in loads)} skipped"
    )


def _folder_failed(loads: list[FolderLoad]) -> bool:
    return any(load.root is None or load.error for load in loads)


@app.command()
def sync(
    ctx: typer.Context,
    full: Annotated[
        bool, typer.Option("--full", help="Drop the index and rebuild it from scratch.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Fetch, parse and chunk only. No embeddings, nothing is written."
        ),
    ] = False,
    show_chunks: Annotated[
        int,
        typer.Option("--show-chunks", min=0, help="Print the first N chunks of each file."),
    ] = 0,
) -> None:
    """Index the configured Drive folders: embed new or changed chunks, remove stale ones."""
    from drive_agent.drive.auth import DriveAuthError, connect  # deferred: slow imports
    from drive_agent.indexing.loader import FileResult, load_folder

    settings = _settings(ctx)
    # pypdf logs recoverable structure problems (e.g. "wrong pointing object") as warnings;
    # they don't affect extraction and would clutter the report.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    # An unreachable Qdrant is reported by us; skip qdrant-client's duplicate thread warning.
    warnings.filterwarnings("ignore", message="Failed to obtain server version")

    if not settings.drive.folder_ids:
        err_console.print("[red]No Drive folders configured.[/red] Run `drive-agent init`.")
        raise typer.Exit(1)
    try:
        _auth, service = connect(settings.drive, settings.data_dir)
    except DriveAuthError as exc:
        err_console.print(f"[red]Auth error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if dry_run:
        console.print(
            "[bold]Dry run[/bold]: fetching, parsing and chunking only. Nothing is written."
        )
        loads: list[FolderLoad] = []
        for folder_id in settings.drive.folder_ids:
            with console.status(f"Reading folder {folder_id}…") as status:

                def progress(result: FileResult) -> None:
                    status.update(f"Parsed {result.file.path}")

                loads.append(load_folder(service, settings, folder_id, on_file=progress))
            _render_load(loads[-1], show_chunks)
        _print_totals(loads)
        raise typer.Exit(1 if _folder_failed(loads) else 0)

    from drive_agent.embeddings import build_embeddings
    from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
    from drive_agent.indexing.sync import SyncError, embedding_fingerprint, run_sync
    from drive_agent.vectorstores import VectorStoreError, build_vector_index

    fingerprint = embedding_fingerprint(settings)
    try:
        with console.status(f"Loading embeddings {fingerprint} (first run downloads the model)…"):
            embeddings = build_embeddings(settings)
            vector_index = build_vector_index(settings, embeddings, recreate=full)
    except (ConfigError, VectorStoreError) as exc:
        err_console.print(f"[red]Setup error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"Syncing into [bold]{vector_index.description}[/bold] · embeddings {fingerprint}"
    )
    records = SQLiteRecordManager(vector_index.namespace, settings.data_dir / STATE_DB)
    try:
        with console.status("Reading Drive and embedding…") as status:
            report = run_sync(
                service,
                settings,
                vector_index,
                records,
                full=full,
                on_file=lambda result: status.update(f"Processed {result.file.path}"),
            )
    except SyncError as exc:
        err_console.print(f"[red]Sync stopped:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        records.close()
        vector_index.close()

    for load in report.folders:
        _render_load(load, show_chunks)
    _print_totals(report.folders)
    console.print(
        f"Index: [green]+{report.added} added[/green] · {report.updated} updated · "
        f"{report.unchanged} unchanged · {report.deleted} deleted"
        + (f" ({report.pruned_files} files removed)" if report.pruned_files else "")
        + f" · {report.seconds}s"
    )
    if report.healed:
        console.print("[yellow]⚠ The vector store was empty, so the sync state was reset.[/yellow]")
    if report.prune_skipped:
        console.print(
            "[yellow]⚠ A folder could not be read, so removed files were not pruned.[/yellow]"
        )
    raise typer.Exit(1 if _folder_failed(report.folders) else 0)


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[
        str | None, typer.Argument(help="Your question. Omit it to start an interactive chat.")
    ] = None,
    thread: Annotated[
        str | None,
        typer.Option("--thread", help="Conversation id to continue (default: a new one)."),
    ] = None,
) -> None:
    """Ask questions about the indexed Drive files; answers cite their sources."""
    from drive_agent.core.agent import DriveAgent, open_checkpointer  # deferred: slow imports
    from drive_agent.embeddings import build_embeddings
    from drive_agent.llm import LLMError
    from drive_agent.llm.claude import ClaudeLLM
    from drive_agent.vectorstores import VectorStoreError, build_vector_index

    settings = _settings(ctx)
    warnings.filterwarnings("ignore", message="Failed to obtain server version")
    try:
        with console.status("Loading the index…"):
            vector_index = build_vector_index(settings, build_embeddings(settings))
    except (ConfigError, VectorStoreError) as exc:
        err_console.print(f"[red]Setup error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        if vector_index.count() == 0:
            err_console.print("[yellow]The index is empty.[/yellow] Run `drive-agent sync` first.")
            raise typer.Exit(1)
        llm = ClaudeLLM(settings)
        store = vector_index.store

        def retriever(query: str, k: int) -> list[Any]:
            return store.similarity_search(query, k=k)

        with open_checkpointer(settings) as checkpointer:
            agent = DriveAgent(settings, llm, retriever, checkpointer)
            if question is not None:
                reply = _answer(agent, question, thread)
                console.print(
                    f"[dim]Conversation {reply.thread_id} · continue with "
                    f"--thread {reply.thread_id}[/dim]"
                )
                return
            thread = thread or uuid.uuid4().hex[:12]
            console.print(
                f"[dim]Chatting about your Drive (conversation {thread}). "
                "Empty line or 'exit' to quit.[/dim]"
            )
            while True:
                try:
                    line = console.input("[bold cyan]You:[/bold cyan] ").strip()
                except EOFError:
                    break
                if not line or line.lower() in {"exit", "quit"}:
                    break
                _answer(agent, line, thread)
    except (ConfigError, LLMError) as exc:
        err_console.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        vector_index.close()


def _answer(agent: DriveAgent, question: str, thread: str | None) -> AgentReply:
    started = False

    def on_text(text: str) -> None:
        nonlocal started
        if not started:  # print the prefix only once the answer actually starts
            console.print("[bold green]Agent:[/bold green] ", end="")
            started = True
        console.print(text, end="", markup=False, highlight=False)

    with console.status("Searching your Drive…") as status:

        def first_token(text: str) -> None:
            status.stop()
            on_text(text)

        reply = agent.ask(question, thread_id=thread, on_text=first_token)
    console.print()
    if reply.citations:
        console.print("\n[bold]Sources[/bold]")
        for citation in sorted(reply.citations, key=lambda c: c["number"]):
            where = (
                f", page {citation['page']}"
                if "page" in citation
                else f", slide {citation['slide']}"
                if "slide" in citation
                else f" > {citation['section']}"
                if "section" in citation
                else ""
            )
            link = f"  [dim]{citation['web_url']}[/dim]" if "web_url" in citation else ""
            console.print(
                f"  [{citation['number']}] {citation.get('file_name', '?')}{where}{link}",
                highlight=False,
            )
    console.print()
    return reply


@app.command()
def ui(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8501,
    host: Annotated[
        str, typer.Option(help="Address to bind; use 0.0.0.0 only behind a password.")
    ] = "localhost",
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open a browser tab automatically.")
    ] = False,
) -> None:
    """Launch the Streamlit app (Chat, Index and Setup pages)."""
    settings = _settings(ctx)
    if importlib.util.find_spec("streamlit") is None:
        err_console.print(
            "[red]Streamlit is not installed.[/red] pip install 'drive-knowledge-agent[streamlit]'"
        )
        raise typer.Exit(1)
    if host not in {"localhost", "127.0.0.1"} and settings.secrets.ui_password is None:
        err_console.print(
            "[yellow]⚠ Serving beyond localhost without UI_PASSWORD: anyone who can reach this "
            "address can query your Drive index.[/yellow]"
        )

    from drive_agent.interfaces import streamlit_app

    app_path = Path(streamlit_app.__file__).with_name("app.py")
    env = dict(os.environ)
    if ctx.obj["config"] is not None:
        env[CONFIG_ENV_VAR] = str(Path(ctx.obj["config"]).resolve())
    command = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true" if no_browser else "false",
        "--client.toolbarMode", "minimal",  # hide Streamlit's developer/deploy menu
    ]  # fmt: skip
    raise typer.Exit(subprocess.call(command, env=env))  # noqa: S603 - fixed argv, no shell


@app.command()
def slack(ctx: typer.Context) -> None:
    """Run the Slack bot."""
    _settings(ctx)
    _not_implemented("slack", "M6")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show the resolved configuration and which secrets are set."""
    settings = _settings(ctx)

    table = Table(title="drive-agent status", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("config file", str(settings.source_file or "(defaults only)"))
    table.add_row("auth mode", settings.drive.auth_mode)
    table.add_row("folders", ", ".join(settings.drive.folder_ids) or "[red]none configured[/red]")
    table.add_row("file types", ", ".join(settings.drive.include_mime))
    table.add_row("embeddings", f"{settings.embeddings.provider} / {settings.embeddings.model}")
    table.add_row("vector store", describe_location(settings))
    table.add_row("answer model", settings.llm.answer_model)
    table.add_row("agent mode", settings.agent.mode)
    table.add_row("index", _index_summary(settings))
    console.print(table)

    secrets = settings.secrets.status()
    console.print(
        "secrets set: " + (", ".join(name for name, is_set in secrets.items() if is_set) or "none")
    )


def _index_summary(settings: Settings) -> str:
    from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager

    db = settings.data_dir / STATE_DB
    if not db.is_file():
        return "not built yet (run `drive-agent sync`)"
    records = SQLiteRecordManager(namespace_for(settings), db)
    try:
        last = records.get_meta("last_sync")
        embedding = records.get_meta("embedding")
    finally:
        records.close()
    if not last:
        return "not built yet (run `drive-agent sync`)"
    failed = f", {last['failed']} failed" if last.get("failed") else ""
    return (
        f"{last['chunks']} chunks from {last['files']} files · last sync {last['at']}{failed}"
        f" · {embedding}"
    )
