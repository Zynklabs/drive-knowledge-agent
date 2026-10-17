# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
