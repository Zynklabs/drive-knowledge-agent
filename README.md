# Drive Knowledge Agent

An open-source agent that indexes a Google Drive folder (including every sub-folder and Shared Drive) and answers questions about its contents with citations. You can use it from a **Streamlit** web app or a **Slack** bot.

- **Read-only Drive access.** You connect it with a service account (for enterprises) or your own Google sign-in (OAuth).
- **Built on LangGraph + LangChain.** You can swap the vector store (Qdrant today; Chroma, Pinecone and pgvector planned), the embedding provider, or the LLM through config.
- **Open-source embeddings by default.** `nomic-embed-text-v1.5` (Apache-2.0) runs locally via FastEmbed, so document text never leaves your machine during indexing.
- **Claude** is the default answer model.

> **Status: pre-alpha (M6).** Index Drive folders and ask questions with cited answers from the CLI, the Streamlit web app, or Slack, with one bot per folder. See the [roadmap](#roadmap).

## Supported files

Google Docs, Google Slides, PDF, Word (.docx), PowerPoint (.pptx), plain text, Markdown, HTML, JSON.

How each type is read: Google Docs are exported as Markdown (headings kept, falling back to .docx), and Google Slides as .pptx (slide boundaries and speaker notes kept). PDFs are parsed page by page with `pypdf`, which is BSD-licensed; PyMuPDF is avoided because it is AGPL. Scanned PDFs without a text layer are reported but not OCR'd yet. Every chunk keeps its file, path, Drive link, heading path, and page or slide number for citations.

Spreadsheets and tabular files are **out of scope**: Excel (.xlsx/.xls/.xlsm), Google Sheets, and CSV. They can hold very large datasets that bloat the index and split poorly into chunks for Q&A. Sync skips them and lists them in the sync report.

## Quick start (development)

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11+ for you.

```bash
git clone https://github.com/OWNER/drive-knowledge-agent.git
cd drive-knowledge-agent
uv sync
uv run drive-agent init      # writes drive-agent.yaml and a .env template
uv run drive-agent status    # shows resolved config and which secrets are set
uv run drive-agent check     # verifies read-only Drive access and lists what will be indexed
```

## Google Drive access

Access is always **read-only** (`drive.readonly`). Use a service account (share the folder with it as Viewer, use domain-wide delegation, or run keyless on Google Cloud), or sign in with your own Google account via OAuth.

Step-by-step setup: **[docs/google-drive-access.md](docs/google-drive-access.md)**.

## Vector database: bring your own Qdrant

The agent stores embeddings in [Qdrant](https://qdrant.tech). Point it at your instance:

```bash
QDRANT_URL=https://<cluster>.cloud.qdrant.io   # or http://localhost:6333
QDRANT_API_KEY=...                             # if your instance requires one
```

To try it locally, run `docker run -p 6333:6333 qdrant/qdrant:v1.19.2`, or use a free Qdrant Cloud cluster. If `QDRANT_URL` is unset, Qdrant runs **embedded on disk** in `.drive-agent/qdrant`, with no server needed (single process; good for a quick trial).

## Embeddings

| `EMBEDDINGS_PROVIDER` | Model (default) | Notes |
|---|---|---|
| `local` (default) | `nomic-ai/nomic-embed-text-v1.5` | Apache-2.0, 8k context, runs on CPU via FastEmbed; about 0.5 GB downloaded once into `.drive-agent/models` |
| `voyage` | `voyage-3.5` | `pip install 'drive-knowledge-agent[voyage]'` + `VOYAGE_API_KEY` |
| `openai` | `text-embedding-3-small` | `pip install 'drive-knowledge-agent[openai]'` + `OPENAI_API_KEY` |

Other good open-source choices are `ibm-granite/granite-embedding-small-english-r2` (tiny and fast) and `Qwen/Qwen3-Embedding-0.6B` (multilingual, heavier). Set them with `EMBEDDINGS_MODEL`. Changing the model requires `drive-agent sync --full`, and the agent refuses to mix embeddings from different models.

## Asking questions

```bash
drive-agent ask "Who has Kubernetes experience?"          # one-off answer with sources
drive-agent ask "And who led a team?" --thread hiring     # continue a conversation
drive-agent ask                                           # interactive chat
```

Each question runs through a LangGraph flow:

1. **Rewrite:** follow-up questions are turned into standalone search queries using the conversation.
2. **Retrieve:** a hybrid search in Qdrant (dense vectors plus BM25 keywords, fused with RRF) returns the top `retrieval.top_k` chunks.
3. **Grade:** a fast model keeps only the relevant chunks. If none are relevant, it retries once with a reworded query; if that also finds nothing, the answer says so instead of guessing.
4. **Answer:** Claude answers from the sources, streaming the text, and cites them as `[n]`. Each citation links back to the Drive file and page or slide.

Conversations are remembered per `--thread` in `.drive-agent/conversations.db`.

| Setting | Default | Notes |
|---|---|---|
| `llm.answer_model` / `ANSWER_MODEL` | `claude-opus-5-5` | Effort `llm.answer_effort` (default `medium`); server-side refusal fallback enabled |
| `llm.fast_model` / `FAST_MODEL` | `claude-haiku-5-5` | Query rewriting and relevance grading (effort `low`) |

Claude credentials: set `ANTHROPIC_API_KEY` in `.env`, or sign in once with `ant auth login`. To use **OpenRouter**, set `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY`. The same Anthropic-format requests go to OpenRouter, with model IDs mapped automatically (`claude-opus-5-5` → `anthropic/claude-opus-5.5`); only Anthropic's server-side refusal fallback isn't available there.

### What leaves your machine

| Step | Data | Where it goes |
|---|---|---|
| `sync` | All indexed text | Your local embedding model, then your Qdrant (Qdrant stores the chunk text) |
| `ask` | Your question, recent conversation turns, the ~20 retrieved chunks (for grading) and the kept chunks (for the answer) | The Claude API (via OpenRouter if `LLM_PROVIDER=openrouter`) |
| `ask` | Questions and answers | `.drive-agent/conversations.db` on this machine |

## Web app (Streamlit)

```bash
pip install 'drive-knowledge-agent[streamlit]'   # or: uv sync --all-extras
drive-agent ui                                   # http://localhost:8501
```

| Page | What it does |
|---|---|
| **Chat** | Streams answers, shows citation cards with **Open** links to the Drive files, and has a "Retrieved passages" view. Each browser session is its own conversation, and "New conversation" starts a fresh one. |
| **Index** | Shows index size and last sync. **Sync now** streams per-file progress; **Full rebuild** must be confirmed first. |
| **Setup** | Shows the resolved configuration and which secrets are set (names only), lets you edit Drive folders, and has a **Check Drive access** button. |

The app binds to `localhost` by default. To serve it to others (`drive-agent ui --host 0.0.0.0`), set `UI_PASSWORD` first: everyone who can open the app can query everything the agent indexed. Streamlit usage statistics are turned off.

## Keeping the index fresh (auto-sync)

The agent re-syncs Drive by itself on a schedule (`ingest.schedule_cron`, default every 6 hours, UTC) whenever it runs as a long-lived process: the Slack bot, the web app, or the dedicated watcher:

```bash
drive-agent sync --watch                 # keep this index fresh without a bot or web app
SYNC_SCHEDULE="0 * * * *" drive-agent slack   # hourly; SYNC_SCHEDULE=off disables it
```

When one of these starts and a scheduled sync was missed while nothing was running, it catches up immediately. Scheduled and manual syncs (CLI, web app, `/drive-agent sync`) share one lock, so they never overlap. `drive-agent status` shows when the next sync is due, and `drive-agent ask` warns if the index is overdue.

## Slack bot

```bash
pip install 'drive-knowledge-agent[slack]'
drive-agent slack          # Socket Mode: no public URL needed
```

Mention the bot or DM it: it answers in the thread with linked sources and 👍/👎 feedback, and it remembers the thread's conversation. `/drive-agent <question>` gives a private answer. **One bot per Drive folder:** `drive-agent bot add hr --folder <url>` creates a separate bot with its own index and Slack app, and you run it with `drive-agent --bot hr slack`. Full setup, including the Slack app manifest: **[docs/slack-setup.md](docs/slack-setup.md)**.

## Configuration

Settings are layered, lowest to highest priority:

1. Built-in defaults
2. `drive-agent.yaml` (or a file passed with `--config`, or `$DRIVE_AGENT_CONFIG`). See [`drive-agent.example.yaml`](drive-agent.example.yaml).
3. Environment variables and `.env`. Use the friendly names in [`.env.example`](.env.example) (e.g. `VECTOR_STORE=qdrant`), or set any field as `DRIVE_AGENT__<SECTION>__<FIELD>` (e.g. `DRIVE_AGENT__INGEST__CHUNK_TOKENS=600`).

Secrets (API keys, tokens, database URLs) are read **only** from the environment. A config file containing a `secrets:` section is rejected.

## Commands

| Command | Purpose | Available in |
|---|---|---|
| `drive-agent init` | Interactive setup | M0 ✅ |
| `drive-agent status` | Resolved config, secrets set, index stats | M0 ✅ |
| `drive-agent check [--show-skipped]` | Permission doctor for Drive access | M1 ✅ |
| `drive-agent sync --dry-run [--show-chunks N]` | Fetch, parse and chunk; report per file, write nothing | M2 ✅ |
| `drive-agent sync [--full]` | Index into Qdrant; re-runs only embed new or changed chunks and remove stale ones | M3 ✅ |
| `drive-agent ask ["…"] [--thread ID]` | Cited answers; interactive chat without a question | M4 ✅ |
| `drive-agent ui [--port N] [--host H] [--no-browser]` | Streamlit app | M5 ✅ |
| `drive-agent slack` | Slack bot (Socket or HTTP mode) | M6 ✅ |
| `drive-agent bot add NAME --folder URL` / `bot list` | One bot per Drive folder; select one with `--bot NAME` on any command | M6 ✅ |

## Roadmap

| # | Milestone |
|---|---|
| M0 | Repo scaffold: package layout, config loader, CLI skeleton, CI |
| M1 | Drive connector: service account + OAuth, recursive walker, `check` |
| M2 | Parsing & chunking |
| M3 | Embeddings + Chroma, `sync` |
| M4 | LangGraph RAG graph + Claude, `ask` |
| M5 | Streamlit app |
| M6 | Slack bot |
| M7 | Qdrant / Pinecone / pgvector + agentic mode |
| M8 | Incremental sync |
| M9 | Hardening: evals, security review, docs, Docker |
| M10 | Release v0.1.0 |

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
