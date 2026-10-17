# Drive Knowledge Agent

An open-source agent that indexes a Google Drive folder (including every sub-folder and Shared Drive) and answers questions about its contents with citations. You can use it from a **Streamlit** web app or a **Slack** bot.

- **Read-only Drive access.** You connect it with a service account (for enterprises) or your own Google sign-in (OAuth).
- **Built on LangGraph + LangChain.** You can swap the vector store (Chroma, Qdrant, Pinecone, pgvector), the embedding provider, or the LLM through config.
- **Claude** is the default answer model.

> **Status: pre-alpha (M1).** `init`, `status`, and `check` work so far. See the [roadmap](#roadmap).

## Supported files

Google Docs, Google Slides, PDF, Word (.docx), PowerPoint (.pptx), plain text, Markdown, HTML, JSON.

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
| `drive-agent sync [--full]` | Index / re-index | M3 |
| `drive-agent ask "…"` | One-off question | M4 |
| `drive-agent ui` | Streamlit app | M5 |
| `drive-agent slack` | Slack bot | M6 |

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
uv sync
uv run pre-commit install
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
