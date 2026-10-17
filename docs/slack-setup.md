# Running the Slack bot

The bot answers questions about your indexed Drive files inside Slack, using the same agent as the CLI and the web app. By default it runs in **Socket Mode**: it opens an outbound connection to Slack, so you don't need a public URL, a tunnel, or open ports.

## 1. Create the Slack app (2 minutes)

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From a manifest**, and pick your workspace.
2. Paste the contents of [`slack/manifest.yaml`](../slack/manifest.yaml) → **Next** → **Create**.
3. **Install App** → **Install to Workspace** → **Allow**. Copy the **Bot User OAuth Token** (`xoxb-…`).
4. **Basic Information** → **App-Level Tokens** → **Generate Token and Scopes**. Name it `socket`, add the scope `connections:write`, then **Generate**. Copy the token (`xapp-…`).

## 2. Configure and run

Add both tokens to `.env`, each on its own line, and never commit them:

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

Then start the bot:

```bash
pip install 'drive-knowledge-agent[slack]'   # or: uv sync --all-extras
drive-agent sync                             # make sure the index is built
drive-agent slack
```

Finally, invite the bot to a channel with `/invite @Drive Agent`.

## 3. Using it

| How | What happens |
|---|---|
| `@Drive Agent who has Kubernetes experience?` | Answers in the message's thread with linked sources and 👍/👎 buttons |
| Reply in that thread and mention `@Drive Agent` again | A follow-up that uses the thread's earlier conversation |
| Direct message to the bot | Same as a mention, in a private conversation |
| `/drive-agent <question>` | A private (ephemeral) answer only you can see; no memory between commands |
| `/drive-agent status` | Index size and last sync |
| `/drive-agent sync` | Re-index now. **Admins only** (see below); the bot also re-syncs automatically |

The bot shows "Searching your Drive…" straight away, then edits that message as the answer streams in.

## 4. One bot per Drive folder

Run a separate bot for each knowledge base, for example HR policies, engineering docs and hiring. Each bot gets its own Drive folder, its own Qdrant collection, its own state (sync records, conversations, feedback), and its own Slack app and slash command. Shared settings (Google service account, Qdrant URL, embedding model, LLM key) stay in the main `.env`.

```bash
drive-agent bot add hr  --folder https://drive.google.com/drive/folders/<HR_FOLDER_ID>
drive-agent bot add eng --folder <ENG_FOLDER_ID> --folder <ANOTHER_FOLDER_ID>
drive-agent bot list
```

`bot add hr` creates:

- **`bots/hr.env`:** the bot's folder(s), collection `drive_agent_hr`, state directory `.drive-agent/bots/hr`, slash command `/drive-hr`, and empty Slack token slots. It's gitignored and private (`0600`) because it will hold tokens.
- **`bots/hr.manifest.yaml`:** a Slack manifest for an app called "Drive Agent (hr)" with `/drive-hr`. Create a Slack app from it (as in step 1) and paste that app's tokens into `bots/hr.env`.

Then use `--bot hr` with any command:

```bash
drive-agent --bot hr check      # verify Drive access to the HR folder
drive-agent --bot hr sync       # build the HR index
drive-agent --bot hr slack      # run the HR bot
drive-agent --bot hr ask "What is the leave policy?"
drive-agent --bot hr ui --port 8502
```

Each bot runs as its own process, so start one `drive-agent --bot <name> slack` per bot (in separate terminals, or as separate services under a process manager or Docker). Values in `bots/<name>.env` override the shared `.env`. An empty value there, such as `SLACK_BOT_TOKEN=`, blanks out the shared one, so bots never share a Slack app by accident.

### Automatic sync

While the bot runs, it keeps its own index fresh: it re-syncs on `SYNC_SCHEDULE` (cron, UTC; default `0 */6 * * *`, every 6 hours) and catches up at startup if a scheduled sync was missed. Each bot can have its own schedule in `bots/<name>.env`, e.g. `SYNC_SCHEDULE=0 * * * *` for hourly, or `SYNC_SCHEDULE=off`. `/drive-agent status` shows the next run. A manual `/drive-agent sync` can't overlap an automatic one: the bot replies that a sync is already running.

## 5. Settings

| Setting (`drive-agent.yaml` → `interfaces.slack`) | Env var | Default | Meaning |
|---|---|---|---|
| `allowed_channels` | `SLACK_ALLOWED_CHANNELS` | empty = all | Channel IDs (e.g. `C0123ABC`) where the bot answers |
| `allow_dms` | `DRIVE_AGENT__INTERFACES__SLACK__ALLOW_DMS` | `true` | Answer direct messages |
| `admin_users` | `SLACK_ADMIN_USERS` | empty = nobody | User IDs (e.g. `U0123ABC`) allowed to run `/drive-agent sync`; `*` = everyone |
| `ingest.schedule_cron` | `SYNC_SCHEDULE` | `0 */6 * * *` | Auto-sync schedule (cron, UTC); `off` disables it |
| `command` | `SLACK_COMMAND` | `/drive-agent` | The slash command; each bot in a workspace needs its own |
| `mode` | `SLACK_MODE` | `socket` | `http` to receive events at a public URL instead |
| `http_port` | `DRIVE_AGENT__INTERFACES__SLACK__HTTP_PORT` | `3000` | Port for HTTP mode |

### HTTP mode (optional)

Use HTTP mode when you deploy behind a public URL and don't want Socket Mode:

1. Set `SLACK_MODE=http` and `SLACK_SIGNING_SECRET` (from **Basic Information**). The app-level token isn't needed.
2. In the app config:
   - turn **Socket Mode** off;
   - set the **Event Subscriptions** request URL to `https://<your-host>/slack/events`;
   - set the same URL for **Interactivity** and for the `/drive-agent` command.

## Privacy

- **Who can see answers:** anyone in a channel can read the bot's answers and their sources. Use `allowed_channels` to keep it to the right audience. Slash-command answers are private to the person who asked.
- **What goes to Claude:** the question and the retrieved passages, as with the CLI.
- **What's stored locally:** thread conversations in `.drive-agent/conversations.db`, and 👍/👎 feedback (with the thread ID, user ID and time) in `.drive-agent/feedback.jsonl`.
- **What the bot can read in Slack:** only messages that mention it or are sent to it. It never reads channel history.
