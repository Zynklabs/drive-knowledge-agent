"""Index page: index status and a sync / full-rebuild button with live per-file progress."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from drive_agent.drive import auth
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.loader import FileResult
from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
from drive_agent.indexing.sync import SyncError, SyncReport, embedding_fingerprint, run_sync
from drive_agent.interfaces.streamlit_app.resources import Resources, get_resources


def _records(resources: Resources) -> SQLiteRecordManager:
    return SQLiteRecordManager(resources.index.namespace, resources.settings.data_dir / STATE_DB)


def _short_time(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %H:%M")
    except ValueError:
        return iso


def _status(resources: Resources) -> None:
    records = _records(resources)
    try:
        last = records.get_meta("last_sync") or {}
    finally:
        records.close()
    chunks, files, when = st.columns(3)
    chunks.metric("Chunks in index", resources.index.count())
    files.metric("Files indexed", last.get("files", 0))
    when.metric("Last sync (UTC)", _short_time(last.get("at")))
    st.caption(
        f"{resources.index.description} · embeddings `{embedding_fingerprint(resources.settings)}`"
    )


def _sync(resources: Resources, *, full: bool) -> SyncReport | None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    settings = resources.settings
    if not settings.drive.folder_ids:
        st.error("No Drive folders configured. Add them on the Setup page.")
        return None
    try:
        _drive_auth, service = auth.connect(settings.drive, settings.data_dir, interactive=False)
    except DriveAuthError as exc:
        st.error(f"**Drive access:** {exc}")
        return None

    records = _records(resources)
    label = "Rebuilding the index…" if full else "Syncing…"
    try:
        with st.status(label, expanded=True) as status:

            def on_file(result: FileResult) -> None:
                if result.error:
                    status.write(f"✗ {result.file.path}: {result.error}")
                else:
                    status.write(f"✓ {result.file.path} · {len(result.chunks)} chunks")

            report = run_sync(
                service, settings, resources.index, records, full=full, on_file=on_file
            )
            status.update(label=f"Done in {report.seconds}s", state="complete", expanded=False)
    except SyncError as exc:
        st.error(str(exc))
        return None
    finally:
        records.close()
    return report


def _report(report: SyncReport) -> None:
    st.success(
        f"+{report.added} added · {report.updated} updated · {report.unchanged} unchanged · "
        f"{report.deleted} deleted"
        + (f" ({report.pruned_files} files removed)" if report.pruned_files else "")
    )
    for load in report.folders:
        if load.error:
            st.error(f"Folder {load.folder_id}: {load.error}")
    if report.prune_skipped:
        st.warning("A folder could not be read, so removed files were not pruned.")
    rows: list[dict[str, Any]] = [
        {
            "status": "✗" if r.error else "⚠" if r.warnings else "✓",
            "file": r.file.path,
            "chunks": len(r.chunks),
            "~tokens": r.tokens,
            "notes": r.error or "; ".join(r.warnings),
        }
        for r in report.files
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_index() -> None:
    st.title("Index")
    resources = get_resources()
    if resources is None:
        return
    _status(resources)

    sync_col, full_col = st.columns(2)
    sync = sync_col.button("Sync now", type="primary", use_container_width=True)
    with full_col:
        confirm = st.checkbox("Re-embed everything (full rebuild)")
        full = st.button("Full rebuild", disabled=not confirm, use_container_width=True)
    st.caption(
        "Sync embeds only new or changed chunks and removes stale ones. A full rebuild is "
        "needed after changing the embedding model or the hybrid setting."
    )
    if sync or full:
        report = _sync(resources, full=full)
        if report is not None:
            _report(report)
