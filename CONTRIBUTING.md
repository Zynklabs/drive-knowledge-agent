# Contributing

Thanks for helping build Drive Knowledge Agent.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Before opening a PR

```bash
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest
```

CI runs the same checks on Python 3.11, 3.12, and 3.13.

## Ground rules

- **Read-only Drive access only.** Never add a write scope or a Drive API call that creates, updates, moves, shares, or deletes files.
- **No secrets in code, config files, tests, or fixtures.** Secrets come from environment variables only.
- **New integrations** (vector stores, embedding providers) go through the LangChain factories in `vectorstores/` and `embeddings/`, ship as optional extras, and must pass the shared conformance tests.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `docs:`, …).
- Add a line to `CHANGELOG.md` for user-facing changes.

## Reporting security issues

Please don't open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
