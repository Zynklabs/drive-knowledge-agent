"""Index page: index status, auto-sync schedule, and sync / full rebuild with live progress."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from drive_agent.core.knowledge_base import KnowledgeBase
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.loader import FileResult
from drive_agent.indexing.scheduler import SyncBusy
from drive_agent.indexing.sync import SyncError, SyncReport, embedding_fingerprint, summarize
from drive_agent.interfaces.streamlit_app.resources import get_resources


def _short_time(moment: datetime | str | None) -> str:
    if not moment:
        return "never"
    if isinstance(moment, str):
        try:
            parsed = datetime.fromisoformat(moment)
        except ValueError:
            return moment
        return parsed.strftime("%b %d, %H:%M")
    return moment.strftime("%b %d, %H:%M")


def _status(kb: KnowledgeBase) -> None:
    status = kb.status()
    chunks, files, when = st.columns(3)
    chunks.metric("Chunks in index", status.chunks)
    files.metric("Files indexed", status.files)
    when.metric("Last sync (UTC)", _short_time(status.last_sync))
    if status.syncing:
        st.info("A sync is running right now.")
    if status.schedule:
        st.caption(
            f"Auto-sync `{status.schedule}` (UTC) while this app runs · "
            f"next {_short_time(status.next_sync)} UTC"
        )
    else:
        st.caption("Auto-sync is off (SYNC_SCHEDULE=off).")
    st.caption(f"{kb.index.description} · embeddings `{embedding_fingerprint(kb.settings)}`")


def _sync(kb: KnowledgeBase, *, full: bool) -> SyncReport | None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    if not kb.settings.drive.folder_ids:
        st.error("No Drive folders configured. Add them on the Setup page.")
        return None
    label = "Rebuilding the index…" if full else "Syncing…"
    with st.status(label, expanded=True) as status:

        def on_file(result: FileResult) -> None:
            if result.error:
                status.write(f"✗ {result.file.path}: {result.error}")
            elif result.unchanged:
                status.write(f"✓ {result.file.path} · unchanged")
            else:
                status.write(f"✓ {result.file.path} · {len(result.chunks)} chunks")

        try:
            report = kb.sync("web", full=full, on_file=on_file)
        except SyncBusy:
            status.update(label="Already syncing", state="error")
            st.warning("A sync is already running (possibly the automatic one). Try again shortly.")
            return None
        except DriveAuthError as exc:
            status.update(label="Drive access failed", state="error")
            st.error(f"**Drive access:** {exc}")
            return None
        except SyncError as exc:
            status.update(label="Sync stopped", state="error")
            st.error(str(exc))
            return None
        status.update(label=f"Done in {report.seconds}s", state="complete", expanded=False)
    return report


def _report(report: SyncReport) -> None:
    st.success(summarize(report))
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
            "notes": r.error or "; ".join(r.warnings) or ("unchanged" if r.unchanged else ""),
        }
        for r in report.files
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_index() -> None:
    st.title("Index")
    kb = get_resources()
    if kb is None:
        return
    _status(kb)

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
        report = _sync(kb, full=full)
        if report is not None:
            _report(report)
