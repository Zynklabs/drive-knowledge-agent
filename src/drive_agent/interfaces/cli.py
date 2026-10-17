"""``drive-agent`` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from drive_agent import __version__
from drive_agent.core.config import (
    DEFAULT_CONFIG_FILE,
    ConfigError,
    Settings,
    extract_folder_id,
    load_settings,
)
from drive_agent.drive.models import SKIP_LABELS

if TYPE_CHECKING:
    from drive_agent.drive.doctor import CheckResult

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


@app.command()
def sync(
    ctx: typer.Context,
    full: Annotated[bool, typer.Option(help="Re-index everything.")] = False,
) -> None:
    """Index or re-index the configured Drive folders."""
    _settings(ctx)
    _not_implemented("sync", "M3")


@app.command()
def ask(ctx: typer.Context, question: Annotated[str, typer.Argument()]) -> None:
    """Ask a one-off question."""
    _settings(ctx)
    _not_implemented("ask", "M4")


@app.command()
def ui(ctx: typer.Context) -> None:
    """Launch the Streamlit app."""
    _settings(ctx)
    _not_implemented("ui", "M5")


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
    table.add_row(
        "vector store", f"{settings.vector_store.provider} / {settings.vector_store.collection}"
    )
    table.add_row("answer model", settings.llm.answer_model)
    table.add_row("agent mode", settings.agent.mode)
    table.add_row("index", "not built yet")
    console.print(table)

    secrets = settings.secrets.status()
    console.print(
        "secrets set: " + (", ".join(name for name, is_set in secrets.items() if is_set) or "none")
    )
