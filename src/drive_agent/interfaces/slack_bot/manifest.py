"""Slack app manifests. Each bot gets its own Slack app (name, display name, slash command)."""

from __future__ import annotations

DEFAULT_APP_NAME = "Drive Knowledge Agent"
DEFAULT_DISPLAY_NAME = "Drive Agent"
DEFAULT_COMMAND = "/drive-agent"

_TEMPLATE = """\
# Slack app manifest for {app_name} (Socket Mode: no public URL needed).
# Create the app at https://api.slack.com/apps → Create New App → From a manifest → paste this.
# Scopes are the minimum the bot needs: it reads only messages that mention it or are sent
# to it directly, and it never reads channel history.
display_information:
  name: {app_name}
  description: Answers questions from your Google Drive files, with sources.
  background_color: "#1f4e79"
features:
  app_home:
    home_tab_enabled: false
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  bot_user:
    display_name: {display_name}
    always_online: true
  slash_commands:
    - command: {command}
      description: Ask your Drive, check index status, or sync
      usage_hint: "[ask] <question> | status | sync | help"
      should_escape: false
oauth_config:
  scopes:
    bot:
      - app_mentions:read # @mentions in channels the bot is invited to
      - chat:write # post and update its own replies
      - commands # {command}
      - im:history # direct messages sent to the bot
      - im:read
      - im:write
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.im
  interactivity:
    is_enabled: true # 👍 / 👎 feedback buttons
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
"""


def render_manifest(
    app_name: str = DEFAULT_APP_NAME,
    display_name: str = DEFAULT_DISPLAY_NAME,
    command: str = DEFAULT_COMMAND,
) -> str:
    return _TEMPLATE.format(app_name=app_name, display_name=display_name, command=command)
