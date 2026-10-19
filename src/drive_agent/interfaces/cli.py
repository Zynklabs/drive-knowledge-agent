"""``drive-agent`` command-line interface."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import threading
import uuid
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
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
    EXTRA_ENV_VAR,
    ConfigError,
    Settings,
    bot_env_file,
    extract_folder_id,
    load_settings,
)
from drive_agent.drive.models import SKIP_LABELS
from drive_agent.vectorstores.location import describe_location, namespace_for

if TYPE_CHECKING:
    from drive_agent.core.agent import AgentReply, DriveAgent
    from drive_agent.core.knowledge_base import KnowledgeBase
    from drive_agent.drive.doctor import CheckResult
    from drive_agent.indexing.loader import FileResult, FolderLoad

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
    bot: Annotated[
        str | None,
        typer.Option("--bot", "-b", help="Use bots/<name>.env (one bot per Drive folder)."),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="Extra env file layered over .env.", dir_okay=False),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    if bot is not None and env_file is not None:
        err_console.print("[red]Use either --bot or --env-file, not both.[/red]")
        raise typer.Exit(2)
    if bot is not None:
        try:
            env_file = bot_env_file(bot)
        except ConfigError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        if not env_file.is_file():
            err_console.print(
                f"[red]No bot named {bot!r}[/red] ({env_file} not found). "
                f"Create it with `drive-agent bot add {bot} --folder <url>`."
            )
            raise typer.Exit(1)
    ctx.obj = {"config": config, "env_file": env_file}


def _settings(ctx: typer.Context) -> Settings:
    try:
        return load_settings(ctx.obj["config"], extra_env_file=ctx.obj.get("env_file"))
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
    with console.status("Checking Google Drive access…") as status:
        result = doctor.run_check(settings, on_scan=_scan_progress(status))
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


def _watch(ctx: typer.Context, *, conflicting: bool) -> None:
    if conflicting:
        err_console.print("[red]--watch can't be combined with --dry-run or --full.[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    if not settings.ingest.schedule_cron:
        err_console.print("[red]Auto-sync is off.[/red] Set SYNC_SCHEDULE to a cron expression.")
        raise typer.Exit(1)
    kb = _open_knowledge_base(settings)
    try:
        _service_logging()
        _announce_auto_sync(kb)
        console.print(f"Watching {kb.index.description}. Ctrl+C to stop.")
        _wait_forever()
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        kb.close()


def _scan_progress(status: Any) -> Callable[[int, int], None]:
    def update(folders: int, items: int) -> None:
        status.update(f"Scanning Drive… {folders:,} folders · {items:,} items listed")

    return update


def _file_progress(status: Any, verb: str) -> Callable[[FileResult], None]:
    count = 0

    def update(result: FileResult) -> None:
        nonlocal count
        count += 1
        status.update(f"{verb} {count:,} files · {result.file.path}")

    return update


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
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Keep running and re-sync on ingest.schedule_cron (catching up if overdue).",
        ),
    ] = False,
) -> None:
    """Index the configured Drive folders: embed new or changed chunks, remove stale ones."""
    if watch:
        _watch(ctx, conflicting=dry_run or full)
        return
    from drive_agent.drive.auth import DriveAuthError, connect  # deferred: slow imports
    from drive_agent.indexing.loader import load_folder

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
                loads.append(
                    load_folder(
                        service,
                        settings,
                        folder_id,
                        on_file=_file_progress(status, "Parsed"),
                        on_scan=_scan_progress(status),
                    )
                )
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
                on_file=_file_progress(status, "Embedded"),
                on_scan=_scan_progress(status),
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
    from drive_agent.llm import LLMError

    settings = _settings(ctx)
    kb = _open_knowledge_base(settings)
    try:
        status = kb.status()
        if status.chunks == 0:
            err_console.print("[yellow]The index is empty.[/yellow] Run `drive-agent sync` first.")
            raise typer.Exit(1)
        if status.overdue:
            console.print(
                f"[yellow]⚠ The index was last synced {_ago(status.last_sync)} and a scheduled "
                "sync is overdue. Run `drive-agent sync`, or keep `drive-agent sync --watch` "
                "running to sync automatically.[/yellow]"
            )
        agent = kb.agent()
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
        kb.close()


def _open_knowledge_base(settings: Settings) -> KnowledgeBase:
    from drive_agent.core.knowledge_base import KnowledgeBase  # deferred: slow imports
    from drive_agent.vectorstores import VectorStoreError

    warnings.filterwarnings("ignore", message="Failed to obtain server version")
    try:
        with console.status("Loading the index…"):
            return KnowledgeBase.open(settings)
    except (ConfigError, VectorStoreError) as exc:
        err_console.print(f"[red]Setup error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _ago(moment: datetime | None) -> str:
    if moment is None:
        return "never"
    hours = (datetime.now(UTC) - moment).total_seconds() / 3600
    return f"{hours:.0f} hours ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def _utc(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC") if moment else "never"


def _service_logging() -> None:
    """Timestamped logs for long-running commands (bot, watcher); quiet the chatty libraries."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "urllib3", "apscheduler", "slack_sdk.socket_mode"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)


def _wait_forever() -> None:
    threading.Event().wait()


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
    if settings.extra_env_file is not None:
        env[EXTRA_ENV_VAR] = str(settings.extra_env_file.resolve())
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
    """Run the Slack bot (Socket Mode by default: no public URL needed).

    While it runs, the bot keeps its index fresh on ingest.schedule_cron (SYNC_SCHEDULE).
    """
    settings = _settings(ctx)
    if importlib.util.find_spec("slack_bolt") is None:
        err_console.print(
            "[red]Slack support is not installed.[/red] pip install 'drive-knowledge-agent[slack]'"
        )
        raise typer.Exit(1)

    from drive_agent.interfaces.slack_bot import runner
    from drive_agent.llm import LLMError

    try:
        runner.required_tokens(settings)
    except ConfigError as exc:
        err_console.print(f"[red]Setup error:[/red] {exc}")
        raise typer.Exit(1) from exc
    kb = _open_knowledge_base(settings)
    try:
        if kb.status().chunks == 0:
            console.print(
                "[yellow]⚠ The index is empty; auto-sync or `drive-agent sync` fills it.[/yellow]"
            )
        _service_logging()
        _announce_auto_sync(kb)
        console.print(
            f"Slack bot running in {settings.interfaces.slack.mode} mode · "
            f"{kb.index.description}. Ctrl+C to stop."
        )
        runner.run(kb)
    except (ConfigError, LLMError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        kb.close()


def _announce_auto_sync(kb: KnowledgeBase) -> None:
    if kb.start_auto_sync():
        status = kb.status()
        console.print(
            f"Auto-sync: `{status.schedule}` (UTC) · next {_utc(status.next_sync)}"
            + (" · catching up now (overdue)" if status.overdue else "")
        )
    else:
        console.print("Auto-sync: off (SYNC_SCHEDULE=off)")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show the resolved configuration and which secrets are set."""
    settings = _settings(ctx)

    table = Table(title="drive-agent status", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    if settings.extra_env_file is not None:
        table.add_row("bot env file", str(settings.extra_env_file))
    table.add_row("config file", str(settings.source_file or "(defaults only)"))
    table.add_row("auth mode", settings.drive.auth_mode)
    table.add_row("folders", ", ".join(settings.drive.folder_ids) or "[red]none configured[/red]")
    table.add_row("file types", ", ".join(settings.drive.include_mime))
    table.add_row("embeddings", f"{settings.embeddings.provider} / {settings.embeddings.model}")
    table.add_row("vector store", describe_location(settings))
    table.add_row("answer model", settings.llm.answer_model)
    table.add_row("agent mode", settings.agent.mode)
    table.add_row("index", _index_summary(settings))
    table.add_row("auto-sync", _auto_sync_summary(settings))
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


# --- bots: one bot per Drive folder ------------------------------------------

bot_app = typer.Typer(
    help="Manage bots. Each bot = one Drive knowledge base with its own Slack app.",
    no_args_is_help=True,
)
app.add_typer(bot_app, name="bot")

BOT_ENV_TEMPLATE = """\
# Bot "{name}": one Drive knowledge base with its own index and Slack app.
# Layered over the shared .env (Google, Qdrant, LLM settings come from there).
# Holds Slack tokens: never commit this file.
DRIVE_FOLDER_IDS={folders}
DRIVE_AGENT__VECTOR_STORE__COLLECTION={collection}
DRIVE_AGENT__DATA_DIR=.drive-agent/bots/{name}
SLACK_COMMAND={command}

# From the Slack app created with {manifest}:
SLACK_BOT_TOKEN=
SLACK_APP_TOKEN=

# Optional:
# SLACK_ALLOWED_CHANNELS=C0123ABC,C0456DEF
# SLACK_ADMIN_USERS=U0123ABC
"""


@bot_app.command("add")
def bot_add(
    name: Annotated[str, typer.Argument(help="Short name, e.g. hr or engineering.")],
    folder: Annotated[
        list[str], typer.Option("--folder", "-f", help="Drive folder URL or ID (repeatable).")
    ],
    force: Annotated[bool, typer.Option(help="Overwrite an existing bot.")] = False,
) -> None:
    """Create bots/<name>.env and a Slack app manifest for one Drive folder."""
    from drive_agent.interfaces.slack_bot.manifest import render_manifest

    try:
        env_path = bot_env_file(name)
        folder_ids = [extract_folder_id(f) for f in folder]
    except (ConfigError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if env_path.exists() and not force:
        err_console.print(f"[red]{env_path} already exists.[/red] Use --force to overwrite.")
        raise typer.Exit(1)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    command = f"/drive-{name}"[:32]
    manifest_path = env_path.with_name(f"{name}.manifest.yaml")
    content = BOT_ENV_TEMPLATE.format(
        name=name,
        folders=",".join(folder_ids),
        collection="drive_agent_" + name.replace("-", "_"),
        command=command,
        manifest=manifest_path,
    )
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    manifest_path.write_text(
        render_manifest(
            app_name=f"Drive Agent ({name})", display_name=f"drive-{name}", command=command
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Created bot {name!r}[/green]: {env_path} and {manifest_path}")
    console.print(
        "Next:\n"
        f"  1. Create a Slack app from {manifest_path} (api.slack.com/apps → From a manifest),\n"
        f"     then put its SLACK_BOT_TOKEN and SLACK_APP_TOKEN in {env_path}\n"
        f"  2. drive-agent --bot {name} check\n"
        f"  3. drive-agent --bot {name} sync\n"
        f"  4. drive-agent --bot {name} slack"
    )


@bot_app.command("list")
def bot_list() -> None:
    """Show every bot with its folders, index and Slack setup."""
    from drive_agent.core.config import BOTS_DIR

    files = sorted(BOTS_DIR.glob("*.env"))
    if not files:
        console.print("No bots yet. Create one with `drive-agent bot add <name> --folder <url>`.")
        return
    table = Table("bot", "folders", "index", "slack", "command")
    for path in files:
        try:
            settings = load_settings(extra_env_file=path)
        except ConfigError as exc:
            table.add_row(path.stem, f"[red]config error: {exc}[/red]", "", "", "")
            continue
        secrets = settings.secrets
        tokens = (
            "[green]ready[/green]"
            if secrets.slack_bot_token and secrets.slack_app_token
            else "[yellow]tokens missing[/yellow]"
        )
        table.add_row(
            path.stem,
            ", ".join(settings.drive.folder_ids) or "[red]none[/red]",
            _index_summary(settings).split(" · ")[0],
            tokens,
            settings.interfaces.slack.command,
        )
    console.print(table)


def _auto_sync_summary(settings: Settings) -> str:
    from drive_agent.core.knowledge_base import next_due
    from drive_agent.indexing.scheduler import is_overdue
    from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager

    schedule = settings.ingest.schedule_cron
    if not schedule:
        return "off"
    last = None
    db = settings.data_dir / STATE_DB
    if db.is_file():
        records = SQLiteRecordManager(namespace_for(settings), db)
        try:
            at = (records.get_meta("last_sync") or {}).get("at")
        finally:
            records.close()
        last = datetime.fromisoformat(at) if at else None
    due = (
        "[yellow]overdue[/yellow]" if is_overdue(schedule, last) else _utc(next_due(schedule, last))
    )
    return f"`{schedule}` (UTC) · next due {due} · runs inside `slack`, `ui` and `sync --watch`"
