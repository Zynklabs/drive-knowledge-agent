# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security
- Slack access follows channel membership. Direct messages are now **off by default** (`SLACK_ALLOW_DMS=true` turns them on): before, anyone in the workspace could DM a bot and query its whole folder. Slash-command questions are answered only in channels the bot has been added to (and in DMs when they are on); before, `/drive-agent <question>` worked from any channel or DM.

### Fixed
- A sync no longer crashes on a very large file: chunks are written 64 at a time, under Qdrant's 32 MB request limit. Before, one huge file crashed the sync and every restart crashed on it again.

### Added
- Admin portal (`drive-agent admin`, localhost:8600): each bot's health, files in the folder vs files synced, live sync progress, 24-hour question stats, an activity log (question text only with `SLACK_LOG_QUESTIONS=true`) and live service logs (`LOG_DIR`).
- Syncs save to the index file by file, so a stopped sync resumes where it left off. Local embedding inputs are capped at 1,024 tokens, which keeps memory bounded on badly-tokenizing text.
- Faster Drive walking with identical results:
  - A whole Shared Drive is listed flat in a few requests. On a 692-folder, 3,386-item drive the check went from about 7 min to 17 s.
  - Other trees list each folder with its own query, 8 in parallel: 250 folders went from 144 s to 23 s.
  - Folders are never combined into one `or` query, because Drive silently drops some children from those. That was caught when an early version missed 4 Google Docs; the file sets are now verified identical to a one-folder-at-a-time walk.
- Live progress in `check` and `sync` (folders scanned, items listed, files processed).
- A Shared Drive used as the root now shows its real name instead of "Drive", in output and citation paths.
- Auto-sync, owned by the agent: a new `KnowledgeBase` core object bundles the index, conversation memory, the agent and the index's freshness.
  - Long-running hosts (Slack bot, web app, and the new `drive-agent sync --watch`) re-sync on `ingest.schedule_cron` / `SYNC_SCHEDULE` (cron, UTC, default every 6 hours, `off` to disable). They catch up immediately at startup when a scheduled run was missed.
  - All syncs (scheduled, CLI, web app, `/drive-agent sync`) go through one coordinator, so they never overlap.
  - `status` shows the next due sync, `ask` warns when the index is overdue, and the web app's Index page and the bot's `/drive-agent status` show the schedule.
  - Long-running commands log with timestamps.
- `SLACK_ADMIN_USERS=*` lets everyone run `/drive-agent sync`.

### Changed
- The CLI `ask` / `slack`, the web app and the Slack runner are all built on `KnowledgeBase` instead of wiring the index, LLM and checkpointer themselves.
- Slack bot (M6), run with `drive-agent slack`:
  - Answers @mentions and DMs in-thread. A "Searching your Drive…" placeholder is updated as the answer streams (throttled to respect rate limits), and the final Block Kit reply has mrkdwn text, linked sources and 👍/👎 feedback (logged to `feedback.jsonl`). Each Slack thread is one conversation.
  - The `/drive-agent` command gives ephemeral answers and has `status`, `help` and admin-only `sync`.
  - Duplicate event deliveries are ignored, bot messages are skipped, and a channel allow-list and DM toggle are available.
  - Runs in Socket Mode (default) or HTTP mode, from a minimal-scope Slack app manifest (`slack/manifest.yaml`). Setup guide: `docs/slack-setup.md`.
- One bot per Drive folder: `drive-agent bot add NAME --folder URL` writes `bots/NAME.env` (folder, its own Qdrant collection, its own state directory, its own slash command and empty Slack token slots; mode 0600, gitignored) and a per-bot Slack manifest. `drive-agent bot list` shows all bots. The global `--bot NAME` / `--env-file PATH` options layer that file over the shared `.env` for any command, including `ui`.
- `sync_now()` and `summarize()` are shared by the web app and Slack, and `make_retriever()` is shared by the CLI, web app and Slack.
- Streamlit web app (M5), launched with `drive-agent ui`:
  - **Chat** streams answers, shows citation cards with links to Drive and a retrieved-passages view, and keeps each browser session's conversation in the shared SQLite checkpointer.
  - **Index** shows status and runs sync or a confirmed full rebuild with live per-file progress.
  - **Setup** shows the resolved config and which secrets are set, edits folders (and says when `.env` manages them), and runs the Drive access check.
  - An optional `UI_PASSWORD` gate. The launcher warns when the app is served beyond localhost without one, and Streamlit telemetry and the deploy menu are off.
- `AgentReply.sources` (the passages an answer used) and a thread-safe `create_checkpointer`.
- Question answering (M4): `drive-agent ask` runs a LangGraph RAG graph. It rewrites follow-ups into standalone queries, runs a hybrid search (dense + BM25, RRF inside Qdrant), grades relevance with a corrective retry, then streams a grounded answer from Claude with `[n]` citations linked to Drive files and pages. Conversation memory is kept per `--thread` (SQLite checkpointer), and running `ask` with no question opens an interactive chat.
- Claude via the official `anthropic` SDK. The answer model `claude-opus-5-5` runs at explicit effort with `fallbacks="default"` (server-side refusal fallback). Query rewriting and grading use `claude-haiku-5-5` with structured outputs. Missing credentials, rate limits and network errors get clear messages.
- Hybrid retrieval: the collection stores BM25 sparse vectors (FastEmbed `Qdrant/bm25`) next to the dense vectors. Older collections are detected and need `sync --full`, which now recreates the collection.

- `LLM_PROVIDER=openrouter`: the same Anthropic-SDK requests go through OpenRouter's Anthropic-compatible endpoint (Bearer `OPENROUTER_API_KEY`). Model IDs are mapped to `anthropic/claude-opus-5.5` style, and the Anthropic-only fallback beta is omitted.

### Changed
- Defaults: `llm.answer_model=claude-opus-5-5`, `retrieval.top_k=20`, `agent.mode=rag` (agentic mode is planned). Removed the unimplemented `retrieval.rerank`; added `retrieval.grade`, `llm.answer_effort` and `llm.fast_effort`.
- Indexing (M3): `drive-agent sync` embeds chunks and stores them in Qdrant. The Qdrant server URL is set with `QDRANT_URL`/`QDRANT_API_KEY`; without a URL Qdrant runs embedded on disk, and `:memory:` is used for tests. Default embeddings are open source and local (`nomic-embed-text-v1.5` via FastEmbed, with task prefixes applied); Voyage and OpenAI are optional extras.
- Incremental re-sync through LangChain's indexing API with a SQLite record manager. Unchanged chunks are skipped, changed files have their chunks replaced, and files that vanish or stop yielding text are pruned. A file that fails to fetch keeps its previous chunks, and pruning is skipped if any folder fails to read. Changing the embedding model requires `sync --full`. Chunk IDs are SHA-256 based UUIDs, which Qdrant accepts.
- `drive-agent status` shows index size, last sync time, and the embedding model.

### Changed
- Defaults: `embeddings.provider=local`, `vector_store.provider=qdrant`, `ingest.contextual_chunks=false` (Claude-written chunk context is not implemented yet).
- Fetch, parse and chunk (M2). Google Docs are exported as Markdown (falling back to .docx) and Slides as .pptx; binary files are downloaded with a size cap. There are parsers for Markdown, text, HTML, JSON, PDF (`pypdf`), .docx and .pptx, and each keeps its heading path, page or slide number. A structure-aware chunker (`langchain-text-splitters`) emits LangChain `Document`s with stable `<file_id>:<n>` IDs and citation metadata. PDF cleanup re-attaches bullets and punctuation split onto their own lines. Scanned and password-protected PDFs are reported, and one bad file never stops a run.
- `drive-agent sync --dry-run [--show-chunks N]`: a per-file report of chunks and approximate tokens, without writing anything.
- Google Drive connector (M1): service-account keys, domain-wide delegation, keyless Application Default Credentials, and OAuth sign-in. Read-only `drive.readonly` scope only; cached OAuth tokens with broader scopes are rejected.
- Recursive folder walker covering Shared Drives, shortcuts (with cycle detection), pagination, `max_depth`, `max_file_mb`, and file-type filtering. Spreadsheets and tabular files are skipped as out of scope.
- `drive-agent check` permission doctor with actionable error messages and least-privilege warnings.
- Setup guide: `docs/google-drive-access.md`.
- Repo scaffold: package layout, layered config loader (defaults → YAML → env → overrides), `drive-agent` CLI with `init` and `status`, CI, and pre-commit.
