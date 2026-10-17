"""Setup page: resolved configuration, Drive folders, and the Drive access check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from drive_agent.core.config import (
    DEFAULT_CONFIG_FILE,
    ConfigError,
    Settings,
    extract_folder_id,
    load_settings,
    read_environment,
)
from drive_agent.drive import doctor
from drive_agent.drive.models import SKIP_LABELS
from drive_agent.interfaces.streamlit_app.resources import reset_resources
from drive_agent.vectorstores.location import describe_location


def _config_table(settings: Settings) -> None:
    rows = {
        "Config file": str(settings.source_file or "(defaults + environment)"),
        "Drive auth": settings.drive.auth_mode,
        "File types": ", ".join(settings.drive.include_mime),
        "Embeddings": f"{settings.embeddings.provider} / {settings.embeddings.model}",
        "Vector store": describe_location(settings),
        "Hybrid search": "on (dense + BM25)" if settings.retrieval.hybrid else "off",
        "LLM": f"{settings.llm.provider} · answer {settings.llm.answer_model} · "
        f"fast {settings.llm.fast_model}",
    }
    st.table([{"setting": key, "value": value} for key, value in rows.items()])
    present = [name for name, is_set in settings.secrets.status().items() if is_set]
    st.caption("Secrets set (from .env / environment): " + (", ".join(present) or "none"))


def save_folders(settings: Settings, folder_ids: list[str]) -> Path:
    """Write ``drive.folder_ids`` into the YAML config, keeping its other settings."""
    path = settings.source_file or DEFAULT_CONFIG_FILE
    data: dict[str, Any] = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("drive", {})["folder_ids"] = folder_ids
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _folders(settings: Settings) -> None:
    st.subheader("Drive folders")
    env = read_environment()
    env_override = bool(env.get("DRIVE_FOLDER_IDS") or env.get("DRIVE_FOLDER_ID"))
    text = st.text_area(
        "Folder URLs or IDs, one per line (sub-folders are included)",
        value="\n".join(settings.drive.folder_ids),
        disabled=env_override,
    )
    if env_override:
        st.info("Folders come from DRIVE_FOLDER_IDS in .env / the environment; edit them there.")
        return
    if st.button("Save folders"):
        try:
            ids = [extract_folder_id(line) for line in text.splitlines() if line.strip()]
        except ValueError as exc:
            st.error(str(exc))
            return
        path = save_folders(settings, ids)
        reset_resources()
        st.success(f"Saved {len(ids)} folder(s) to {path}. Run a sync on the Index page.")


def _check(settings: Settings) -> None:
    st.subheader("Drive access")
    st.caption(
        "Read-only access via a service account or OAuth. Step-by-step setup: "
        "`docs/google-drive-access.md` in the repository. OAuth sign-in happens once in a "
        "terminal: `drive-agent check`."
    )
    if not st.button("Check Drive access"):
        return
    with st.spinner("Checking…"):
        result = doctor.run_check(settings, interactive=False)
    if result.description is None or result.error:
        st.error(result.error or "Authentication failed")
        return
    st.success(f"Signed in as {result.identity or result.description} · scope drive.readonly")
    for folder in result.folders:
        if folder.report is None:
            st.error(f"Folder {folder.folder_id}: {folder.error}")
            continue
        report = folder.report
        where = "Shared Drive" if report.root.drive_id else "My Drive"
        with st.container(border=True):
            st.markdown(f"**{report.root.name}** · {where}")
            kinds = ", ".join(f"{k} {n}" for k, n in sorted(report.kinds().items()))
            st.write(
                f"{len(report.files)} files to index ({kinds or 'none'}) · "
                f"{report.stats.folders} sub-folders"
            )
            if report.skipped:
                st.caption(
                    "Skipped: "
                    + ", ".join(
                        f"{n} {SKIP_LABELS[reason]}"
                        for reason, n in report.skip_reasons().most_common()
                    )
                )
            for warning in folder.warnings:
                st.warning(warning)


def render_setup() -> None:
    st.title("Setup")
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"**Configuration error:** {exc}")
        return
    _config_table(settings)
    _folders(settings)
    _check(settings)
