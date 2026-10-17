"""Streamlit entry point (``drive-agent ui`` runs ``streamlit run`` on this file)."""

from __future__ import annotations

import hmac

import streamlit as st

from drive_agent.core.config import ConfigError, load_settings
from drive_agent.interfaces.streamlit_app.chat import render_chat
from drive_agent.interfaces.streamlit_app.index_page import render_index
from drive_agent.interfaces.streamlit_app.setup_page import render_setup


def password_ok() -> bool:
    """Gate the app behind UI_PASSWORD when it is set (recommended whenever it is hosted)."""
    try:
        secret = load_settings().secrets.ui_password
    except ConfigError:
        return True  # the pages report configuration errors themselves
    if secret is None or st.session_state.get("authenticated"):
        return True
    st.title("Drive Knowledge Agent")
    attempt = st.text_input("Password", type="password")
    if attempt and hmac.compare_digest(attempt.encode(), secret.get_secret_value().encode()):
        st.session_state.authenticated = True
        st.rerun()
    elif attempt:
        st.error("Wrong password.")
    return False


def main() -> None:
    st.set_page_config(page_title="Drive Knowledge Agent", page_icon="📁", layout="centered")
    if not password_ok():
        return
    page = st.navigation(
        [
            st.Page(render_chat, title="Chat", icon="💬", url_path="chat", default=True),
            st.Page(render_index, title="Index", icon="🗂️", url_path="index"),
            st.Page(render_setup, title="Setup", icon="⚙️", url_path="setup"),
        ]
    )
    page.run()


main()
