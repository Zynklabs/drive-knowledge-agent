# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Google Drive connector (M1): service-account keys, domain-wide delegation, keyless Application Default Credentials, and OAuth sign-in. Read-only `drive.readonly` scope only; cached OAuth tokens with broader scopes are rejected.
- Recursive folder walker covering Shared Drives, shortcuts (with cycle detection), pagination, `max_depth`, `max_file_mb`, and file-type filtering. Spreadsheets and tabular files are skipped as out of scope.
- `drive-agent check` permission doctor with actionable error messages and least-privilege warnings.
- Setup guide: `docs/google-drive-access.md`.
- Repo scaffold: package layout, layered config loader (defaults → YAML → env → overrides), `drive-agent` CLI with `init` and `status`, CI, and pre-commit.
