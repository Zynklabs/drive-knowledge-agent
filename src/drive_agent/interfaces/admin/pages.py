"""Admin portal pages.

Read-only monitoring for every bot in ``bots/``: is it up, what is indexed, is a sync running,
how are questions going, and the service logs. It reads files only (see ``data.py``), so it
loads no model and never touches a bot's index.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from drive_agent.core.config import ConfigError, load_settings
from drive_agent.core.monitor import read_activity
from drive_agent.interfaces.admin.data import (
    BotOverview,
    bot_names,
    bot_settings,
    log_files,
    overview,
    parse_time,
    tail,
)

REFRESH = "10s"
HEALTH_ICONS = {
    "running": "🟢",
    "not responding": "🟠",
    "stopped": "🔴",
    "never started": "⚪",
}
SLACK_ICONS = {"connected": "🟢", "unstable": "🟠", "disconnected": "🔴"}


def password_ok() -> bool:
    """ADMIN_PASSWORD (or UI_PASSWORD) gates the portal when set."""
    try:
        secrets = load_settings().secrets
    except ConfigError:
        return True
    secret = secrets.admin_password or secrets.ui_password
    if secret is None or st.session_state.get("admin_ok"):
        return True
    st.title("Drive Agent admin")
    attempt = st.text_input("Password", type="password")
    if attempt and hmac.compare_digest(attempt.encode(), secret.get_secret_value().encode()):
        st.session_state.admin_ok = True
        st.rerun()
    elif attempt:
        st.error("Wrong password.")
    return False


def when(moment: datetime | None) -> str:
    """Local time plus how long ago / from now, e.g. '19 Oct 16:35 (12 min ago)'."""
    if moment is None:
        return "never"
    delta = (moment - datetime.now(UTC)).total_seconds()
    minutes = abs(delta) / 60
    if minutes < 1:
        rel = "just now"
    elif minutes < 90:
        rel = f"{minutes:.0f} min"
    elif minutes < 48 * 60:
        rel = f"{minutes / 60:.1f} h"
    else:
        rel = f"{minutes / 1440:.0f} days"
    if rel != "just now":
        rel = f"in {rel}" if delta > 0 else f"{rel} ago"
    return f"{moment.astimezone():%d %b %H:%M} ({rel})"


def render_bot(bot: BotOverview) -> None:
    with st.container(border=True):
        if bot.config_error or bot.settings is None:
            st.subheader(f"⚠️ {bot.name}")
            st.error(bot.config_error or "Could not load settings.")
            return
        settings = bot.settings
        icon = HEALTH_ICONS[bot.health]
        st.subheader(f"{icon} {bot.name}")
        hb = bot.heartbeat or {}
        details = [f"**{bot.health}**"]
        if bot.health == "running":
            details.append(f"up since {when(parse_time(hb.get('started_at')))}")
        elif hb.get("updated_at"):
            details.append(f"last seen {when(parse_time(hb.get('updated_at')))}")
        details.append(f"`{settings.interfaces.slack.command}`")
        st.markdown(" · ".join(details))
        render_slack(bot)

        index, total = bot.index, bot.total_files
        a, b, c, d = st.columns(4)
        a.metric(
            "Files in folder",
            f"{total:,}" if total is not None else "-",
            help="Supported files found in the Drive folder (known once a sync has listed it).",
        )
        b.metric(
            "Files synced",
            f"{index.synced_files:,}",
            help="Files whose text is in the index now. Updates live while a sync runs.",
        )
        c.metric("Questions, 24 h", bot.last_24h.questions)
        stats = bot.last_24h
        rate = f"{stats.not_found / stats.questions:.0%}" if stats.questions else "-"
        d.metric("Not found, 24 h", rate)
        if total:
            share = min(index.synced_files / total, 1.0)
            st.progress(share, text=f"{share:.0%} of files synced")

        details = [f"{index.chunks:,} chunks"]
        if index.failed:
            details.append(f"{index.failed} failed")
        if index.empty:
            details.append(f"{index.empty} with no readable text")
        if index.skipped:
            details.append(f"{index.skipped} skipped by type or size")
        st.caption(" · ".join(details))

        sync_line = f"Last sync: {when(index.last_sync)}"
        sync_line += (
            f" · auto-sync `{bot.schedule}` UTC, next {when(bot.next_sync)}"
            if bot.schedule
            else " · auto-sync off"
        )
        st.caption(sync_line)
        if stats.questions:
            st.caption(
                f"24 h: {stats.answered} answered · {stats.not_found} not found · "
                f"{stats.errors} errors · {stats.users} people · "
                f"avg {stats.avg_seconds} s per answer"
            )
        render_sync(bot.sync)


def render_slack(bot: BotOverview) -> None:
    """The Slack connection: the process can be up while Slack keeps dropping it."""
    render_slack_restarts(bot.slack_restarts)
    state = bot.slack_health
    if state == "unknown":
        return
    slack = (bot.heartbeat or {}).get("slack") or {}
    drops, errors = slack.get("drops_10m", 0), slack.get("errors_10m", 0)
    last_message = when(parse_time(slack.get("last_message_at")))
    line = f"Slack {SLACK_ICONS[state]} **{state}**"
    if state == "connected":
        line += f" since {when(parse_time(slack.get('connected_since')))}"
    line += f" · {drops} drops, {errors} errors in 10 min · last message from Slack {last_message}"
    if state == "connected":
        st.markdown(line)
        return
    hint = (
        "It keeps reconnecting, so messages may not reach the bot. If this lasts, restart it."
        if state == "unstable"
        else "Messages are not reaching the bot. If this lasts, restart it."
    )
    if slack.get("last_error"):
        line += f" · last error `{slack['last_error']}`"
    st.warning(f"{line}\n\n{hint}")


def render_slack_restarts(restarts: list[dict[str, Any]]) -> None:
    """What the bot's Slack watchdog did about a stuck connection in the last 24 h."""
    if not restarts:
        return
    exits = sum(1 for e in restarts if e.get("action") == "restart")
    count = f"{len(restarts) - exits} reconnects, {exits} restarts"
    last = restarts[0]
    st.caption(
        f"Slack watchdog, 24 h: {count} · last {last.get('action')} "
        f"{when(parse_time(last.get('at')))} ({last.get('reason') or 'no reason given'})"
    )


def render_sync(sync: dict[str, Any] | None) -> None:
    if not sync:
        return
    state = sync.get("state")
    done, total = sync.get("files_done", 0), sync.get("files_total")
    if isinstance(total, int):
        processed = f"{done:,} of {total:,} files checked"
    elif state == "running" and done == 0:
        processed = "listing the folder"
    else:
        processed = f"{done:,} files checked"
    counts = f"{processed} · {sync.get('chunks_added', 0):,} new chunks" + (
        f" · {sync['files_failed']} failed" if sync.get("files_failed") else ""
    )
    if state == "running":
        st.info(
            f"🔄 **Sync running** since {when(parse_time(sync.get('started_at')))}: {counts} "
            f"(updated {when(parse_time(sync.get('updated_at')))})"
        )
    elif state == "failed":
        st.warning(
            f"Last sync stopped {when(parse_time(sync.get('finished_at')))}: "
            f"{sync.get('error', 'unknown error')} ({counts})"
        )


def overview_page() -> None:
    st.title("Bots")
    st.caption(f"Refreshes every {REFRESH}.")

    @st.fragment(run_every=REFRESH)
    def bots() -> None:
        for name in bot_names():
            render_bot(overview(name))

    bots()


def activity_page() -> None:
    st.title("Activity")
    names = bot_names()
    name = st.selectbox("Bot", names)
    try:
        settings = bot_settings(name)
    except ConfigError as exc:
        st.error(str(exc))
        return
    events = [
        e for e in read_activity(settings.data_dir, limit=2000) if e.get("kind") == "question"
    ]
    if not events:
        st.info("No questions yet. Each question a bot answers shows up here.")
        return
    rows = [
        {
            "time": (
                t.astimezone().strftime("%d %b %H:%M:%S") if (t := parse_time(e.get("at"))) else ""
            ),
            "outcome": e.get("outcome", ""),
            "seconds": e.get("seconds"),
            "sources": e.get("sources"),
            "via": e.get("via", ""),
            "user": e.get("user") or "",
            "channel": e.get("channel") or "",
            "question": e.get("question", ""),
        }
        for e in reversed(events)
    ]
    if not any(row["question"] for row in rows):
        for row in rows:
            del row["question"]
        st.caption("Question text is not recorded. Set SLACK_LOG_QUESTIONS=true to include it.")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def logs_page() -> None:
    st.title("Logs")
    try:
        log_dir = load_settings().admin.log_dir
    except ConfigError as exc:
        st.error(str(exc))
        return
    files = log_files(log_dir)
    if not files:
        st.info(
            "No log folder configured. Set LOG_DIR to the folder where your bots' output is "
            "written (for example the service manager's log folder)."
            if log_dir is None
            else f"No .log files in {log_dir}."
        )
        return
    left, mid, right = st.columns([3, 2, 1])
    path: Path = left.selectbox("File", files, format_func=lambda p: p.name)
    contains = mid.text_input("Only lines containing", placeholder="e.g. ERROR")
    lines = right.number_input("Lines", min_value=50, max_value=5000, value=300, step=50)
    live = st.toggle("Follow (refresh every 5 s)", value=True)

    @st.fragment(run_every="5s" if live else None)
    def show() -> None:
        st.caption(
            f"{path} · updated {when(datetime.fromtimestamp(path.stat().st_mtime, UTC))} "
            "· newest first"
        )
        newest_first = reversed(tail(path, int(lines), contains))
        st.code("\n".join(newest_first) or "(empty)", language=None)

    show()
