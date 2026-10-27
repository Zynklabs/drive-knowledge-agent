"""Admin portal (``drive-agent admin`` runs ``streamlit run`` on this file)."""

from __future__ import annotations

import streamlit as st

from drive_agent.interfaces.admin.pages import (
    activity_page,
    logs_page,
    overview_page,
    password_ok,
    syncs_page,
)


def main() -> None:
    st.set_page_config(page_title="Drive Agent admin", page_icon="🛠️", layout="wide")
    if not password_ok():
        return
    st.navigation(
        [
            st.Page(overview_page, title="Bots", icon="🤖", url_path="bots", default=True),
            st.Page(syncs_page, title="Syncs", icon="🔄", url_path="syncs"),
            st.Page(activity_page, title="Activity", icon="📈", url_path="activity"),
            st.Page(logs_page, title="Logs", icon="📜", url_path="logs"),
        ]
    ).run()


main()
